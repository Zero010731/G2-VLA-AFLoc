from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.contracts import MRSGOutput
from anaprior.models.afloc_mrsg.teacher import TeacherTarget


@dataclass(frozen=True)
class MRSGGroupedLoss:
    total: torch.Tensor
    grounding: torch.Tensor
    teacher: torch.Tensor
    mask: torch.Tensor
    query: torch.Tensor
    diagnostics: dict[str, float]


def cross_modal_grounding_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    reconstructed_phrase: torch.Tensor,
    target_phrase: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    _validate_positive_negative_scores(positive_scores, negative_scores)
    _require_finite("reconstructed_phrase", reconstructed_phrase)
    _require_finite("target_phrase", target_phrase)
    if reconstructed_phrase.ndim == 2:
        reconstructed_phrase = reconstructed_phrase.unsqueeze(1)
    if reconstructed_phrase.ndim != 3:
        raise ValueError("reconstructed_phrase must have shape [B,C] or [B,4,C]")
    if target_phrase.ndim != 2:
        raise ValueError("target_phrase must have shape [B,C]")
    if reconstructed_phrase.shape[0] != positive_scores.shape[0]:
        raise ValueError("reconstructed_phrase must share positive_scores batch size")
    if reconstructed_phrase.shape[2] != target_phrase.shape[1]:
        raise ValueError("target_phrase must match reconstructed phrase dimension")
    if target_phrase.shape[0] != positive_scores.shape[0]:
        raise ValueError("target_phrase must share positive_scores batch size")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    positive_prob = positive_scores.clamp(1.0e-6, 1.0 - 1.0e-6)
    mil = F.binary_cross_entropy(positive_prob, torch.ones_like(positive_prob))
    cycle = _cosine_distance(
        reconstructed_phrase,
        target_phrase.detach().unsqueeze(1).expand_as(reconstructed_phrase),
        dim=-1,
    ).mean()
    counterfactual = F.relu(
        margin + negative_scores - positive_scores.unsqueeze(1),
    ).mean()
    return mil + 0.5 * cycle + counterfactual


def teacher_equivariance_loss(
    student_final_heatmap: torch.Tensor,
    student_query_heatmaps: torch.Tensor,
    teacher_target: TeacherTarget | None,
) -> torch.Tensor:
    _validate_student_maps(student_final_heatmap, student_query_heatmaps)
    if teacher_target is None:
        return _zero_like_loss(student_final_heatmap, student_query_heatmaps)

    _validate_teacher_target(teacher_target, student_final_heatmap, student_query_heatmaps)
    confidence = teacher_target.confidence.detach().clamp(0.0, 1.0)
    teacher_final = teacher_target.final_heatmap.detach()
    teacher_queries = teacher_target.query_heatmaps.detach()

    final = _weighted_mse(student_final_heatmap, teacher_final, confidence)
    query = _weighted_mse(
        student_query_heatmaps,
        teacher_queries,
        confidence.expand_as(student_query_heatmaps),
    )
    return final + 0.25 * query


def masked_patch_distillation_loss(
    masked_predictions: Mapping[str, torch.Tensor],
    source_targets: Mapping[str, torch.Tensor],
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    expected_keys = ("l2", "l", "lf")
    if set(masked_predictions.keys()) != set(expected_keys):
        raise ValueError("masked_predictions must contain l2, l, and lf")
    if set(source_targets.keys()) != set(expected_keys):
        raise ValueError("source_targets must contain l2, l, and lf")
    if patch_mask.ndim != 4 or patch_mask.shape[1] != 1:
        raise ValueError("patch_mask must have shape [B,1,H,W]")
    if patch_mask.dtype != torch.bool:
        raise ValueError("patch_mask must be a boolean tensor")

    losses = []
    mask = patch_mask
    for name in expected_keys:
        prediction = masked_predictions[name]
        target = source_targets[name]
        _require_finite(f"masked_predictions[{name}]", prediction)
        _require_finite(f"source_targets[{name}]", target)
        if prediction.ndim != 4:
            raise ValueError(f"masked_predictions[{name}] must have shape [B,C,H,W]")
        if target.shape != prediction.shape:
            raise ValueError(f"source_targets[{name}] must match masked_predictions[{name}]")
        if patch_mask.shape[0] != prediction.shape[0] or patch_mask.shape[-2:] != prediction.shape[-2:]:
            raise ValueError("patch_mask must match prediction batch and spatial shape")
        if not patch_mask.any():
            losses.append(_zero_like_loss(prediction))
            continue
        cosine = _cosine_distance(prediction, target.detach(), dim=1)
        losses.append(cosine.masked_select(mask[:, 0]).mean())
    return torch.stack(losses).mean()


def query_regularization_loss(
    query_heatmaps: torch.Tensor,
    route_weights: torch.Tensor,
) -> torch.Tensor:
    if query_heatmaps.ndim != 4 or query_heatmaps.shape[1] != 4:
        raise ValueError("query_heatmaps must have shape [B,4,H,W]")
    if route_weights.shape != query_heatmaps.shape[:2]:
        raise ValueError("route_weights must have shape [B,4]")
    _require_finite("query_heatmaps", query_heatmaps)
    _require_finite("route_weights", route_weights)

    route_balance = (route_weights.mean(dim=0) - 0.25).square().mean()
    diversity = _pairwise_query_cosine(query_heatmaps).mean()
    noncollapse = F.relu(0.02 - query_heatmaps.var(dim=(-2, -1), unbiased=False)).mean()
    operator_structure = _operator_structure_loss(query_heatmaps)
    return 0.25 * (route_balance + diversity + noncollapse + operator_structure)


def compute_mrsg_loss(
    student: MRSGOutput,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    pyramid: Any,
    teacher_target: TeacherTarget | None,
    config: Any,
    target_phrase: torch.Tensor | None = None,
    margin: float = 0.2,
) -> MRSGGroupedLoss:
    if not isinstance(student, MRSGOutput):
        raise ValueError("student must be an MRSGOutput")
    student.validate()
    if target_phrase is None:
        if student.query_reconstructed_phrase is None:
            raise ValueError("target_phrase is required when query reconstruction is absent")
        target_phrase = student.query_reconstructed_phrase.mean(dim=1).detach()
    if student.query_reconstructed_phrase is None:
        raise ValueError("student.query_reconstructed_phrase is required")

    grounding = cross_modal_grounding_loss(
        positive_scores,
        negative_scores,
        student.query_reconstructed_phrase,
        target_phrase,
        margin=margin,
    )
    teacher = teacher_equivariance_loss(
        student.final_heatmap,
        student.query_heatmaps,
        teacher_target,
    )
    mask = _mask_group_loss(student, pyramid)
    query = query_regularization_loss(student.query_heatmaps, student.query_route_weights)

    total = (
        _top_level_weight(config, "w_ground") * grounding
        + _top_level_weight(config, "w_teacher") * teacher
        + _top_level_weight(config, "w_mask") * mask
        + _top_level_weight(config, "w_query") * query
    )
    _require_finite("total", total)
    return MRSGGroupedLoss(
        total=total,
        grounding=grounding,
        teacher=teacher,
        mask=mask,
        query=query,
        diagnostics={
            "grounding": _float_detached(grounding),
            "teacher": _float_detached(teacher),
            "mask": _float_detached(mask),
            "query": _float_detached(query),
            "positive_negative_margin": _float_detached(
                positive_scores.detach().mean() - negative_scores.detach().mean()
            ),
            "teacher_confident_coverage": _teacher_coverage(teacher_target),
            "query_pairwise_cosine": _float_detached(
                _pairwise_query_cosine(student.query_heatmaps.detach()).mean()
            ),
        },
    )


def _mask_group_loss(student: MRSGOutput, pyramid: Any) -> torch.Tensor:
    masked_predictions = student.masked_predictions
    source_targets = student.source_targets
    patch_mask = student.patch_mask
    if masked_predictions is None and hasattr(pyramid, "masked_prediction"):
        masked_predictions = pyramid.masked_prediction
    if source_targets is None and hasattr(pyramid, "source_targets"):
        source_targets = pyramid.source_targets
    if patch_mask is None and hasattr(pyramid, "patch_mask"):
        patch_mask = pyramid.patch_mask
    if masked_predictions is None or source_targets is None or patch_mask is None:
        return _zero_like_loss(student.final_heatmap)
    return masked_patch_distillation_loss(masked_predictions, source_targets, patch_mask)


def _validate_positive_negative_scores(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> None:
    if positive_scores.ndim != 1:
        raise ValueError("positive_scores must have shape [B]")
    if negative_scores.ndim != 2:
        raise ValueError("negative_scores must have shape [B,N]")
    if negative_scores.shape[0] != positive_scores.shape[0]:
        raise ValueError("negative_scores must share positive_scores batch size")
    if negative_scores.shape[1] == 0:
        raise ValueError("negative_scores must contain at least one negative")
    _require_finite("positive_scores", positive_scores)
    _require_finite("negative_scores", negative_scores)


def _validate_student_maps(
    final_heatmap: torch.Tensor,
    query_heatmaps: torch.Tensor,
) -> None:
    if final_heatmap.ndim != 4 or final_heatmap.shape[1] != 1:
        raise ValueError("student final heatmap must have shape [B,1,H,W]")
    if query_heatmaps.shape != (
        final_heatmap.shape[0],
        4,
        final_heatmap.shape[2],
        final_heatmap.shape[3],
    ):
        raise ValueError("student query heatmaps must have shape [B,4,H,W]")
    _require_finite("student_final_heatmap", final_heatmap)
    _require_finite("student_query_heatmaps", query_heatmaps)


def _validate_teacher_target(
    target: TeacherTarget,
    student_final: torch.Tensor,
    student_queries: torch.Tensor,
) -> None:
    if target.final_heatmap.shape != student_final.shape:
        raise ValueError("teacher final_heatmap must match student final heatmap")
    if target.query_heatmaps.shape != student_queries.shape:
        raise ValueError("teacher query_heatmaps must match student query heatmaps")
    if target.confidence.shape != student_final.shape:
        raise ValueError("teacher confidence must have shape [B,1,H,W]")
    if target.route_weights.shape != (student_final.shape[0], 4):
        raise ValueError("teacher route_weights must have shape [B,4]")
    _require_finite("teacher final_heatmap", target.final_heatmap)
    _require_finite("teacher query_heatmaps", target.query_heatmaps)
    _require_finite("teacher confidence", target.confidence)
    _require_finite("teacher route_weights", target.route_weights)
    if target.route_weights.requires_grad:
        raise ValueError("teacher route_weights must be detached")


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    numerator = ((prediction - target).square() * weight).sum()
    denominator = weight.sum() * (prediction.shape[1] if weight.shape[1] == 1 else 1)
    return numerator / denominator.clamp_min(1.0)


def _cosine_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(prediction, target, dim=dim, eps=1.0e-6)


def _pairwise_query_cosine(query_heatmaps: torch.Tensor) -> torch.Tensor:
    flat = query_heatmaps.flatten(2)
    flat = F.normalize(flat, dim=-1, eps=1.0e-6)
    pairs = []
    for left in range(4):
        for right in range(left + 1, 4):
            pairs.append((flat[:, left] * flat[:, right]).sum(dim=-1).abs())
    return torch.stack(pairs, dim=1)


def _operator_structure_loss(query_heatmaps: torch.Tensor) -> torch.Tensor:
    focal = query_heatmaps[:, 0:1]
    diffuse = query_heatmaps[:, 1:2]
    boundary = query_heatmaps[:, 2:3]
    structural = query_heatmaps[:, 3:4]

    area = focal.mean(dim=(-2, -1))
    focal_area = F.relu(area - 0.35).mean()
    diffuse_smooth = _total_variation(diffuse)
    boundary_continuity = _total_variation(boundary)
    structural_symmetry = (structural - structural.flip(-1)).abs().mean()
    return focal_area + diffuse_smooth + boundary_continuity + structural_symmetry


def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
    horizontal = (tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).abs().mean()
    vertical = (tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).abs().mean()
    return horizontal + vertical


def _top_level_weight(config: Any, name: str) -> float:
    if isinstance(config, Mapping):
        value = config.get(name, 1.0)
    else:
        value = getattr(config, name, 1.0)
    if not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _teacher_coverage(target: TeacherTarget | None) -> float:
    if target is None:
        return 0.0
    return _float_detached(target.confidence.detach().clamp(0.0, 1.0).gt(0.0).float().mean())


def _zero_like_loss(*tensors: torch.Tensor) -> torch.Tensor:
    if not tensors:
        return torch.tensor(0.0)
    return sum(tensor.sum() * 0.0 for tensor in tensors)


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")


def _float_detached(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().item())
