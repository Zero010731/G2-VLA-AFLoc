from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.model import AFLocMRSG


@dataclass(frozen=True)
class GeometryTransform:
    horizontal_flip: bool = False
    crop_top: int = 0
    crop_left: int = 0
    crop_height: int | None = None
    crop_width: int | None = None


@dataclass(frozen=True)
class TeacherTarget:
    final_heatmap: torch.Tensor
    query_heatmaps: torch.Tensor
    confidence: torch.Tensor
    route_weights: torch.Tensor


class MRSGTeacher(nn.Module):
    def __init__(self, student: AFLocMRSG, decay: float = 0.99) -> None:
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0.0, 1.0]")
        self.decay = float(decay)
        self.model = deepcopy(student)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: AFLocMRSG) -> None:
        teacher_parameters = dict(self.model.named_parameters())
        student_parameters = dict(student.named_parameters())
        for name, teacher_parameter in teacher_parameters.items():
            student_parameter = student_parameters[name].detach()
            teacher_parameter.mul_(self.decay).add_(
                student_parameter,
                alpha=1.0 - self.decay,
            )

        teacher_buffers = dict(self.model.named_buffers())
        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in teacher_buffers.items():
            teacher_buffer.copy_(student_buffers[name].detach())

    @torch.no_grad()
    def forward(
        self,
        image_features,
        phrase_features,
        patch_mask: torch.Tensor | None = None,
    ) -> TeacherTarget:
        output = self.model(
            image_features,
            phrase_features,
            patch_mask=patch_mask,
        )
        final_heatmap = output.final_heatmap.detach()
        query_heatmaps = output.query_heatmaps.detach()
        route_weights = output.query_route_weights.detach()
        confidence = torch.ones_like(final_heatmap).detach()
        return TeacherTarget(
            final_heatmap=final_heatmap,
            query_heatmaps=query_heatmaps,
            confidence=confidence,
            route_weights=route_weights,
        )


def transform_heatmap(
    heatmap: torch.Tensor,
    transform: GeometryTransform,
    output_size: tuple[int, int] | None = None,
    mode: str = "bilinear",
) -> torch.Tensor:
    aligned = heatmap
    height, width = aligned.shape[-2:]

    top = min(max(int(transform.crop_top), 0), height)
    left = min(max(int(transform.crop_left), 0), width)
    bottom = height if transform.crop_height is None else min(
        top + max(int(transform.crop_height), 0),
        height,
    )
    right = width if transform.crop_width is None else min(
        left + max(int(transform.crop_width), 0),
        width,
    )
    if bottom > top and right > left:
        aligned = aligned[..., top:bottom, left:right]

    if transform.horizontal_flip:
        aligned = aligned.flip(-1)

    if output_size is None or tuple(aligned.shape[-2:]) == tuple(output_size):
        return aligned

    interpolate_kwargs = {}
    if mode not in {"nearest", "area", "nearest-exact"}:
        interpolate_kwargs["align_corners"] = False
    return F.interpolate(aligned, size=output_size, mode=mode, **interpolate_kwargs)


def transform_phrase(phrase: str, transform: GeometryTransform) -> str:
    if not transform.horizontal_flip:
        return phrase

    left_pattern = re.compile(r"\bleft\b", re.IGNORECASE)
    right_pattern = re.compile(r"\bright\b", re.IGNORECASE)
    has_left = bool(left_pattern.search(phrase))
    has_right = bool(right_pattern.search(phrase))
    if has_left == has_right:
        return phrase
    if has_left:
        return left_pattern.sub(lambda match: _match_case(match.group(0), "right"), phrase)
    return right_pattern.sub(lambda match: _match_case(match.group(0), "left"), phrase)


def teacher_confidence(
    *,
    scale_heatmaps: tuple[torch.Tensor, ...],
    weak_heatmap: torch.Tensor,
    strong_heatmap: torch.Tensor,
    teacher_route_weights: torch.Tensor,
    student_route_weights: torch.Tensor,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    reference_heatmap = _normalize_heatmap(weak_heatmap)
    scale_agreement = _mean_heatmap_agreement(scale_heatmaps, reference_heatmap)
    augmentation_agreement = _heatmap_agreement(
        reference_heatmap,
        _normalize_heatmap(strong_heatmap),
    )
    route_stability = _route_agreement(
        teacher_route_weights,
        student_route_weights,
        device=reference_heatmap.device,
        dtype=reference_heatmap.dtype,
    )
    phrase_margin_confidence = _phrase_margin_confidence(
        positive_scores,
        negative_scores,
        margin=margin,
        device=reference_heatmap.device,
        dtype=reference_heatmap.dtype,
    )
    confidence = (
        scale_agreement
        * augmentation_agreement
        * route_stability
        * phrase_margin_confidence
    )
    return confidence.detach().nan_to_num(0.0).clamp(0.0, 1.0)


def _match_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def _normalize_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    normalized = heatmap.detach().nan_to_num(0.0)
    minimum = normalized.amin(dim=(-2, -1), keepdim=True)
    maximum = normalized.amax(dim=(-2, -1), keepdim=True)
    scale = (maximum - minimum).clamp_min(1.0e-6)
    return ((normalized - minimum) / scale).clamp(0.0, 1.0)


def _heatmap_agreement(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (1.0 - (first - second).abs()).clamp(0.0, 1.0)


def _mean_heatmap_agreement(
    scale_heatmaps: tuple[torch.Tensor, ...],
    reference_heatmap: torch.Tensor,
) -> torch.Tensor:
    if len(scale_heatmaps) <= 1:
        return torch.ones_like(reference_heatmap)
    agreements = [
        _heatmap_agreement(reference_heatmap, _normalize_heatmap(heatmap))
        for heatmap in scale_heatmaps
    ]
    return torch.stack(agreements, dim=0).mean(dim=0)


def _route_agreement(
    teacher_route_weights: torch.Tensor,
    student_route_weights: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    teacher = teacher_route_weights.detach().to(device=device, dtype=dtype).nan_to_num(0.0)
    student = student_route_weights.detach().to(device=device, dtype=dtype).nan_to_num(0.0)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    student = student / student.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    agreement = 1.0 - 0.5 * (teacher - student).abs().sum(dim=1, keepdim=True)
    return agreement.clamp(0.0, 1.0)[:, :, None, None]


def _phrase_margin_confidence(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    margin: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positive = positive_scores.detach().to(device=device, dtype=dtype).nan_to_num(0.0)
    negative = negative_scores.detach().to(device=device, dtype=dtype).nan_to_num(0.0)
    negative_max = negative.amax(dim=1)
    denominator = max(float(margin), 1.0e-6)
    confidence = ((positive - negative_max) / denominator).clamp(0.0, 1.0)
    return confidence[:, None, None, None]
