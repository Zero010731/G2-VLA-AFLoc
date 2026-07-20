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


def anchor_patch_grounding_loss(
    student_heatmap: torch.Tensor,
    anchor_heatmap: torch.Tensor,
    anchor_confidence: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Keep learned evidence near reliable AFLoc patches and rank them above the rest."""
    _validate_spatial_pair(student_heatmap, anchor_heatmap, "anchor_heatmap")
    _validate_spatial_pair(student_heatmap, anchor_confidence, "anchor_confidence")
    confidence = anchor_confidence.detach().clamp(0.05, 1.0)
    anchor = anchor_heatmap.detach().clamp(0.0, 1.0)
    alignment = ((student_heatmap - anchor).abs() * confidence).sum() / confidence.sum().clamp_min(1.0)
    positive = (student_heatmap * anchor * confidence).sum() / (anchor * confidence).sum().clamp_min(1.0)
    outside = ((student_heatmap * (1.0 - anchor)) * confidence).sum()
    outside = outside / ((1.0 - anchor) * confidence).sum().clamp_min(1.0)
    ranking = F.relu(float(margin) - positive + outside)
    return alignment + ranking


def cross_view_patch_consistency_loss(
    student_heatmap: torch.Tensor,
    transformed_student_heatmap: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    _validate_spatial_pair(student_heatmap, transformed_student_heatmap, "transformed_student_heatmap")
    _validate_spatial_pair(student_heatmap, confidence, "confidence")
    weight = confidence.detach().clamp(0.05, 1.0)
    return (
        (student_heatmap - transformed_student_heatmap).abs() * weight
    ).sum() / weight.sum().clamp_min(1.0)


def phrase_patch_refinement_loss(
    final_heatmap: torch.Tensor,
    phrase_patch_logits: torch.Tensor,
) -> torch.Tensor:
    """Make the bounded residual follow dense phrase evidence without backpropagating into it."""
    if final_heatmap.ndim != 4 or final_heatmap.shape[1] != 1:
        raise ValueError("final_heatmap must have shape [B,1,H,W]")
    if phrase_patch_logits.ndim != 4:
        raise ValueError("phrase_patch_logits must have shape [B,T,H,W]")
    if phrase_patch_logits.shape[0] != final_heatmap.shape[0]:
        raise ValueError("phrase_patch_logits must share final_heatmap batch size")
    if phrase_patch_logits.shape[-2:] != final_heatmap.shape[-2:]:
        raise ValueError("phrase_patch_logits must share final_heatmap spatial shape")

    logits = phrase_patch_logits.detach()
    valid_tokens = logits.amax(dim=(-2, -1)).gt(-1.0e3)
    valid_count = valid_tokens.sum(dim=1, keepdim=True).clamp_min(1)
    evidence = (
        logits.masked_fill(~valid_tokens[:, :, None, None], 0.0).sum(dim=1, keepdim=True)
        / valid_count[:, :, None, None].to(dtype=logits.dtype)
    )
    evidence_mean = evidence.mean(dim=(-2, -1), keepdim=True)
    evidence_std = evidence.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1.0e-4)
    target = torch.sigmoid(((evidence - evidence_mean) / evidence_std).clamp(-6.0, 6.0))

    alignment = F.smooth_l1_loss(final_heatmap, target)
    final_centered = final_heatmap - final_heatmap.mean(dim=(-2, -1), keepdim=True)
    target_centered = target - target.mean(dim=(-2, -1), keepdim=True)
    rank_alignment = 1.0 - F.cosine_similarity(
        final_centered.flatten(1),
        target_centered.flatten(1),
        dim=1,
        eps=1.0e-6,
    ).mean()
    return 0.5 * (alignment + rank_alignment)


def residual_stability_loss(raw_residual_logits: torch.Tensor, cap: float = 2.0) -> torch.Tensor:
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    _require_finite("raw_residual_logits", raw_residual_logits)
    energy = raw_residual_logits.square().mean()
    saturation = F.relu(raw_residual_logits.abs() - 0.8 * cap).square().mean()
    return 0.1 * energy + saturation


def phrase_swap_correction_contrast_loss(
    positive_correction: torch.Tensor,
    negative_corrections: torch.Tensor,
    negative_mask: torch.Tensor,
    similarity_ceiling: float = 0.5,
) -> torch.Tensor:
    if positive_correction.ndim != 4 or positive_correction.shape[1] != 1:
        raise ValueError("positive_correction must have shape [B,1,H,W]")
    if negative_corrections.ndim != 5 or negative_corrections.shape[2] != 1:
        raise ValueError("negative_corrections must have shape [B,N,1,H,W]")
    if negative_corrections.shape[0] != positive_correction.shape[0]:
        raise ValueError("positive and negative corrections must share batch size")
    if negative_corrections.shape[-2:] != positive_correction.shape[-2:]:
        raise ValueError("positive and negative corrections must share spatial shape")
    if negative_mask.shape != negative_corrections.shape[:2]:
        raise ValueError("negative_mask must have shape [B,N]")
    if not negative_mask.any():
        return _zero_like_loss(positive_correction, negative_corrections)
    positive = positive_correction.flatten(1)
    positive = positive - positive.mean(dim=1, keepdim=True)
    negative = negative_corrections.flatten(2)
    negative = negative - negative.mean(dim=2, keepdim=True)
    similarity = F.cosine_similarity(positive[:, None], negative, dim=2, eps=1.0e-6)
    penalties = F.relu(similarity - similarity_ceiling)
    return penalties.masked_select(negative_mask).mean()


def _validate_spatial_pair(reference: torch.Tensor, value: torch.Tensor, name: str) -> None:
    if value.shape != reference.shape:
        raise ValueError(f"{name} must match student_heatmap shape")
    _require_finite(name, value)


def cross_modal_grounding_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    reconstructed_phrase: torch.Tensor,
    target_phrase: torch.Tensor,
    margin: float = 0.2,
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if negative_mask is None:
        negative_mask = torch.ones_like(negative_scores, dtype=torch.bool)
    _validate_positive_negative_scores(positive_scores, negative_scores, negative_mask)
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
    if target_phrase.shape[0] != positive_scores.shape[0]:
        raise ValueError("target_phrase must share positive_scores batch size")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    positive_prob = positive_scores.clamp(1.0e-6, 1.0 - 1.0e-6)
    mil = F.binary_cross_entropy(positive_prob, torch.ones_like(positive_prob))
    aligned_target_phrase = _align_target_phrase_dimension(
        target_phrase.detach(),
        reconstructed_phrase.shape[2],
    )
    cycle = _cosine_distance(
        reconstructed_phrase,
        aligned_target_phrase.unsqueeze(1).expand_as(reconstructed_phrase),
        dim=-1,
    ).mean()
    counterfactual = _masked_counterfactual_mean(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        negative_mask=negative_mask,
        margin=margin,
    )
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
    final_heatmap: torch.Tensor | None = None,
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
    query_loss = 0.25 * (route_balance + diversity + noncollapse + operator_structure)
    return query_loss


def compute_mrsg_loss(
    student: MRSGOutput,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    pyramid: Any,
    teacher_target: TeacherTarget | None,
    config: Any,
    target_phrase: torch.Tensor | None = None,
    margin: float = 0.2,
    negative_mask: torch.Tensor | None = None,
    anchor_heatmap: torch.Tensor | None = None,
    anchor_confidence: torch.Tensor | None = None,
    cross_view_loss: torch.Tensor | None = None,
    residual_stability: torch.Tensor | None = None,
    phrase_swap_loss: torch.Tensor | None = None,
    w_residual_stability: float = 0.0,
    w_cross_correction: float = 0.0,
    w_phrase_swap: float = 0.0,
) -> MRSGGroupedLoss:
    if not isinstance(student, MRSGOutput):
        raise ValueError("student must be an MRSGOutput")
    student.validate()
    ground_weight = _top_level_weight(config, "w_ground")
    anchor_loss = _zero_like_loss(student.final_heatmap)
    cross_view_value = _zero_like_loss(student.final_heatmap)
    patch_refinement = _zero_like_loss(student.final_heatmap)
    residual_stability_value = _zero_like_loss(student.final_heatmap)
    phrase_swap_value = _zero_like_loss(student.final_heatmap)
    if ground_weight > 0.0:
        if target_phrase is None:
            raise ValueError("target_phrase is required when grounding weight is active")
        if student.query_reconstructed_phrase is None:
            raise ValueError("student.query_reconstructed_phrase is required")
        grounding = cross_modal_grounding_loss(
            positive_scores,
            negative_scores,
            student.query_reconstructed_phrase,
            target_phrase.detach(),
            margin=margin,
            negative_mask=negative_mask,
        )
        patch_refinement = phrase_patch_refinement_loss(
            student.final_heatmap,
            student.phrase_patch_logits,
        )
        grounding = grounding + 0.25 * patch_refinement
        if anchor_heatmap is not None or anchor_confidence is not None:
            if anchor_heatmap is None or anchor_confidence is None:
                raise ValueError("anchor_heatmap and anchor_confidence must be provided together")
            anchor_loss = anchor_patch_grounding_loss(
                student.final_heatmap,
                anchor_heatmap,
                anchor_confidence,
            )
            grounding = grounding + anchor_loss
        if cross_view_loss is not None:
            cross_view_value = cross_view_loss
            grounding = grounding + float(w_cross_correction) * cross_view_loss
        if residual_stability is not None:
            residual_stability_value = residual_stability
            grounding = grounding + float(w_residual_stability) * residual_stability
        if phrase_swap_loss is not None:
            phrase_swap_value = phrase_swap_loss
            grounding = grounding + float(w_phrase_swap) * phrase_swap_loss
    else:
        grounding = _zero_like_loss(
            positive_scores,
            negative_scores,
            student.final_heatmap,
        )
    teacher = teacher_equivariance_loss(
        student.final_heatmap,
        student.query_heatmaps,
        teacher_target,
    )
    mask = _mask_group_loss(student, pyramid)
    query = query_regularization_loss(
        student.query_heatmaps,
        student.query_route_weights,
        final_heatmap=student.final_heatmap,
    )

    total = (
        ground_weight * grounding
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
            "positive_negative_margin": _masked_positive_negative_margin(
                positive_scores,
                negative_scores,
                negative_mask,
            ),
            "teacher_confident_coverage": _teacher_coverage(teacher_target),
            "query_pairwise_cosine": _float_detached(
                _pairwise_query_cosine(student.query_heatmaps.detach()).mean()
            ),
            "anchor_grounding": _float_detached(anchor_loss),
            "cross_view_patch": _float_detached(cross_view_value),
            "phrase_patch_refinement": _float_detached(patch_refinement),
            "residual_stability": _float_detached(residual_stability_value),
            "phrase_swap_contrast": _float_detached(phrase_swap_value),
            "anchor_confidence_mean": (
                _float_detached(anchor_confidence.mean())
                if anchor_confidence is not None
                else 0.0
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
    negative_mask: torch.Tensor,
) -> None:
    if positive_scores.ndim != 1:
        raise ValueError("positive_scores must have shape [B]")
    if negative_scores.ndim != 2:
        raise ValueError("negative_scores must have shape [B,N]")
    if negative_scores.shape[0] != positive_scores.shape[0]:
        raise ValueError("negative_scores must share positive_scores batch size")
    if negative_mask.shape != negative_scores.shape:
        raise ValueError("negative_mask must match negative_scores shape")
    if negative_mask.dtype != torch.bool:
        raise ValueError("negative_mask must be boolean")
    _require_finite("positive_scores", positive_scores)
    _require_finite("negative_scores", negative_scores)


def _masked_counterfactual_mean(
    *,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    negative_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if negative_scores.shape[1] == 0:
        return _zero_like_loss(positive_scores, negative_scores)
    penalties = F.relu(margin + negative_scores - positive_scores.unsqueeze(1))
    masked = penalties * negative_mask.to(dtype=penalties.dtype)
    valid_count = negative_mask.sum()
    if int(valid_count.detach().cpu().item()) == 0:
        return _zero_like_loss(positive_scores, negative_scores)
    return masked.sum() / valid_count.to(dtype=penalties.dtype)


def _align_target_phrase_dimension(
    target_phrase: torch.Tensor,
    reconstructed_dim: int,
) -> torch.Tensor:
    if target_phrase.shape[1] == reconstructed_dim:
        return target_phrase
    resized = F.interpolate(
        target_phrase.unsqueeze(1),
        size=reconstructed_dim,
        mode="linear",
        align_corners=False,
    )
    return resized.squeeze(1)


def _masked_positive_negative_margin(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    negative_mask: torch.Tensor | None,
) -> float:
    if negative_mask is None:
        negative_mask = torch.ones_like(negative_scores, dtype=torch.bool)
    if positive_scores.ndim != 1 or negative_scores.ndim != 2 or negative_mask.shape != negative_scores.shape:
        return 0.0
    if positive_scores.shape[0] == 0 or negative_scores.shape[0] != positive_scores.shape[0]:
        return 0.0
    if negative_scores.shape[1] == 0:
        return 0.0
    valid_count = negative_mask.sum()
    if int(valid_count.detach().cpu().item()) == 0:
        return 0.0
    negative_mean = (
        negative_scores.detach() * negative_mask.to(dtype=negative_scores.dtype)
    ).sum() / valid_count.to(dtype=negative_scores.dtype)
    margin = positive_scores.detach().mean() - negative_mean
    return float(margin.cpu().item())


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
