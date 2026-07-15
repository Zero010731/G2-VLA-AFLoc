from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.contracts import MRSGOutput
from anaprior.models.afloc_mrsg.teacher import TeacherTarget

HEATMAP_STD_THRESHOLD = 1.0e-3
MAX_ROUTE_UTILIZATION_THRESHOLD = 0.90
QUERY_PAIRWISE_COSINE_THRESHOLD = 0.95
TEACHER_CONFIDENCE_THRESHOLD = 0.5
TEACHER_COVERAGE_MIN = 0.02
TEACHER_COVERAGE_MAX = 0.95


@dataclass(frozen=True)
class PhaseGateDecision:
    phase: str
    passed: bool
    reasons: tuple[str, ...]
    diagnostics: dict[str, float]


def collect_mrsg_diagnostics(
    *,
    output: MRSGOutput,
    positive_scores: torch.Tensor | None = None,
    negative_scores: torch.Tensor | None = None,
    teacher_target: TeacherTarget | None = None,
    masked_reconstruction_loss: torch.Tensor | float | None = None,
    untrained_masked_reconstruction_loss: torch.Tensor | float | None = None,
    model: nn.Module | None = None,
    phase_b_positive_negative_margin: torch.Tensor | float | None = None,
) -> dict[str, float]:
    output.validate()

    heatmap_std = _safe_stat(output.final_heatmap, "std", default=0.0)
    heatmap_variance = _safe_stat(output.final_heatmap, "var", default=0.0)
    heatmap = output.final_heatmap.detach()
    heatmap_mean = float(heatmap.mean().cpu().item())
    heatmap_min = float(heatmap.min().cpu().item())
    heatmap_max = float(heatmap.max().cpu().item())
    probability = heatmap.clamp(1.0e-6, 1.0 - 1.0e-6)
    heatmap_entropy = float(
        (
            -probability * probability.log()
            - (1.0 - probability) * (1.0 - probability).log()
        ).mean().div(math.log(2.0)).cpu().item()
    )
    active_area_ratio = float(heatmap.ge(0.5).float().mean().cpu().item())
    max_route_utilization, route_entropy = _route_utilization_diagnostics(
        output.query_route_weights.detach()
    )
    query_pairwise_cosine = _query_pairwise_cosine(output.query_heatmaps.detach())
    positive_negative_margin = _positive_negative_margin(positive_scores, negative_scores)
    teacher_confident_coverage = _teacher_confident_coverage(teacher_target)
    grad_diagnostics = _gradient_norm_diagnostics(model)

    masked_loss, masked_loss_finite = _finite_scalar(masked_reconstruction_loss)
    untrained_loss, untrained_loss_finite = _finite_scalar(untrained_masked_reconstruction_loss)
    locality_ratio = 0.0
    if masked_loss_finite and untrained_loss_finite and untrained_loss > 0.0:
        locality_ratio = masked_loss / untrained_loss

    phase_b_margin, phase_b_margin_finite = _finite_scalar(phase_b_positive_negative_margin)
    consistency_margin_delta = 0.0
    if phase_b_margin_finite:
        consistency_margin_delta = positive_negative_margin - phase_b_margin

    diagnostics = {
        "heatmap_std": heatmap_std,
        "heatmap_variance": heatmap_variance,
        "heatmap_mean": heatmap_mean,
        "heatmap_min": heatmap_min,
        "heatmap_max": heatmap_max,
        "heatmap_entropy": heatmap_entropy,
        "active_area_ratio": active_area_ratio,
        "max_route_utilization": max_route_utilization,
        "route_utilization_entropy": route_entropy,
        "query_pairwise_cosine": query_pairwise_cosine,
        "query_diversity": max(0.0, 1.0 - query_pairwise_cosine),
        "positive_negative_margin": positive_negative_margin,
        "teacher_confident_coverage": teacher_confident_coverage,
        "masked_reconstruction_loss": masked_loss,
        "masked_reconstruction_finite": 1.0 if masked_loss_finite else 0.0,
        "untrained_masked_reconstruction_loss": untrained_loss,
        "untrained_masked_reconstruction_finite": 1.0 if untrained_loss_finite else 0.0,
        "locality_reconstruction_ratio": locality_ratio,
        "phase_b_positive_negative_margin": phase_b_margin,
        "phase_b_positive_negative_margin_finite": 1.0 if phase_b_margin_finite else 0.0,
        "consistency_margin_delta_vs_phase_b": consistency_margin_delta,
    }
    diagnostics.update(grad_diagnostics)
    return diagnostics


def evaluate_phase_gate(
    phase: str,
    diagnostics: Mapping[str, float],
) -> PhaseGateDecision:
    normalized_phase = phase.strip().lower()
    if normalized_phase not in {"locality", "grounding", "consistency"}:
        raise ValueError("phase must be one of locality, grounding, or consistency")

    reasons: list[str] = []
    heatmap_std, heatmap_std_reason = _gate_diagnostic(diagnostics, "heatmap_std")
    if heatmap_std_reason is not None:
        reasons.append(heatmap_std_reason)
    elif heatmap_std <= HEATMAP_STD_THRESHOLD:
        reasons.append("heatmap_collapse")

    max_route_utilization, max_route_reason = _gate_diagnostic(
        diagnostics,
        "max_route_utilization",
    )
    if max_route_reason is not None:
        reasons.append(max_route_reason)
    elif max_route_utilization >= MAX_ROUTE_UTILIZATION_THRESHOLD:
        reasons.append("route_collapse")

    query_pairwise_cosine, query_pairwise_reason = _gate_diagnostic(
        diagnostics,
        "query_pairwise_cosine",
    )
    if query_pairwise_reason is not None:
        reasons.append(query_pairwise_reason)
    elif query_pairwise_cosine >= QUERY_PAIRWISE_COSINE_THRESHOLD:
        reasons.append("query_collapse")

    positive_negative_margin, positive_negative_reason = _gate_diagnostic(
        diagnostics,
        "positive_negative_margin",
    )
    if positive_negative_reason is not None:
        reasons.append(positive_negative_reason)
    elif positive_negative_margin <= 0.0:
        reasons.append("nonpositive_phrase_margin")

    all_module_gradient_norms_finite, gradient_reason = _gate_diagnostic(
        diagnostics,
        "all_module_gradient_norms_finite",
    )
    if gradient_reason is not None:
        reasons.append(gradient_reason)
    elif all_module_gradient_norms_finite < 1.0:
        reasons.append("nonfinite_gradient_norms")

    if normalized_phase == "locality":
        masked_finite, masked_finite_reason = _gate_flag_or_scalar_finite(
            diagnostics,
            flag_key="masked_reconstruction_finite",
            value_key="masked_reconstruction_loss",
        )
        untrained_finite, untrained_finite_reason = _gate_flag_or_scalar_finite(
            diagnostics,
            flag_key="untrained_masked_reconstruction_finite",
            value_key="untrained_masked_reconstruction_loss",
        )
        if masked_finite_reason is not None:
            reasons.append(masked_finite_reason)
        if untrained_finite_reason is not None:
            reasons.append(untrained_finite_reason)
        if masked_finite_reason is not None or untrained_finite_reason is not None:
            pass
        elif not masked_finite or not untrained_finite:
            reasons.append("nonfinite_masked_reconstruction")
        else:
            masked_loss, masked_loss_reason = _gate_diagnostic(
                diagnostics,
                "masked_reconstruction_loss",
            )
            untrained_loss, untrained_loss_reason = _gate_diagnostic(
                diagnostics,
                "untrained_masked_reconstruction_loss",
            )
            locality_ratio, locality_ratio_reason = _gate_diagnostic(
                diagnostics,
                "locality_reconstruction_ratio",
            )
            for reason in (
                masked_loss_reason,
                untrained_loss_reason,
                locality_ratio_reason,
            ):
                if reason is not None:
                    reasons.append(reason)
            if all(
                reason is None
                for reason in (
                    masked_loss_reason,
                    untrained_loss_reason,
                    locality_ratio_reason,
                )
            ) and (
                untrained_loss <= 0.0
                or masked_loss >= untrained_loss
                or locality_ratio >= 1.0
            ):
                reasons.append("locality_reconstruction_not_improved")

    if normalized_phase == "consistency":
        teacher_confident_coverage, teacher_coverage_reason = _gate_diagnostic(
            diagnostics,
            "teacher_confident_coverage",
        )
        if teacher_coverage_reason is not None:
            reasons.append(teacher_coverage_reason)
        elif not TEACHER_COVERAGE_MIN <= teacher_confident_coverage <= TEACHER_COVERAGE_MAX:
            reasons.append("teacher_confident_coverage_out_of_range")

        phase_b_margin_finite, phase_b_margin_reason = _gate_flag_or_scalar_finite(
            diagnostics,
            flag_key="phase_b_positive_negative_margin_finite",
            value_key="phase_b_positive_negative_margin",
        )
        if phase_b_margin_reason is not None:
            reasons.append(phase_b_margin_reason)
        elif not phase_b_margin_finite:
            reasons.append("nonfinite_diagnostic:phase_b_positive_negative_margin")
        else:
            phase_b_margin, phase_b_margin_value_reason = _gate_diagnostic(
                diagnostics,
                "phase_b_positive_negative_margin",
            )
            if phase_b_margin_value_reason is not None:
                reasons.append(phase_b_margin_value_reason)
            elif positive_negative_reason is None and positive_negative_margin < phase_b_margin:
                reasons.append("consistency_margin_below_phase_b")

    return PhaseGateDecision(
        phase=normalized_phase,
        passed=not reasons,
        reasons=tuple(reasons),
        diagnostics=dict(diagnostics),
    )


def _safe_stat(tensor: torch.Tensor, stat: str, *, default: float) -> float:
    detached = tensor.detach()
    if detached.numel() == 0 or not torch.isfinite(detached).all():
        return default
    if stat == "std":
        return float(detached.std(unbiased=False).cpu().item())
    if stat == "var":
        return float(detached.var(unbiased=False).cpu().item())
    raise ValueError(f"unsupported stat {stat}")


def _route_utilization_diagnostics(route_weights: torch.Tensor) -> tuple[float, float]:
    if route_weights.ndim != 2 or route_weights.shape[1] != 4:
        return 1.0, 0.0
    if route_weights.numel() == 0 or not torch.isfinite(route_weights).all():
        return 1.0, 0.0

    mean_route = route_weights.detach().mean(dim=0).clamp_min(0.0)
    total = float(mean_route.sum().cpu().item())
    if total <= 0.0:
        return 1.0, 0.0
    normalized = mean_route / total
    entropy = -(normalized * normalized.clamp_min(1.0e-12).log()).sum()
    return (
        float(normalized.max().cpu().item()),
        float(entropy.cpu().item()),
    )


def _query_pairwise_cosine(query_heatmaps: torch.Tensor) -> float:
    if query_heatmaps.ndim != 4 or query_heatmaps.shape[1] != 4:
        return 1.0
    if query_heatmaps.numel() == 0 or not torch.isfinite(query_heatmaps).all():
        return 1.0

    flat = query_heatmaps.flatten(2)
    flat = F.normalize(flat, dim=-1, eps=1.0e-6)
    pairs = []
    for left in range(4):
        for right in range(left + 1, 4):
            pairs.append((flat[:, left] * flat[:, right]).sum(dim=-1).abs())
    return float(torch.stack(pairs, dim=1).mean().cpu().item())


def _positive_negative_margin(
    positive_scores: torch.Tensor | None,
    negative_scores: torch.Tensor | None,
) -> float:
    if positive_scores is None or negative_scores is None:
        return 0.0
    if positive_scores.ndim != 1 or negative_scores.ndim != 2:
        return 0.0
    if positive_scores.shape[0] == 0 or negative_scores.shape[0] != positive_scores.shape[0]:
        return 0.0
    if negative_scores.shape[1] == 0:
        return 0.0
    if not torch.isfinite(positive_scores).all() or not torch.isfinite(negative_scores).all():
        return 0.0
    margin = positive_scores.detach().mean() - negative_scores.detach().mean()
    return float(margin.cpu().item())


def _teacher_confident_coverage(target: TeacherTarget | None) -> float:
    if target is None:
        return 0.0
    confidence = target.confidence.detach()
    if confidence.numel() == 0 or not torch.isfinite(confidence).all():
        return 0.0
    return float(
        confidence.clamp(0.0, 1.0).ge(TEACHER_CONFIDENCE_THRESHOLD).float().mean().cpu().item()
    )


def _gradient_norm_diagnostics(model: nn.Module | None) -> dict[str, float]:
    if model is None:
        return {"all_module_gradient_norms_finite": 1.0}

    module_sums: dict[str, float] = {}
    finite = True
    saw_parameter = False
    for name, parameter in model.named_parameters():
        module_name = name.split(".", 1)[0] if "." in name else name or "__root__"
        module_sums.setdefault(module_name, 0.0)
        grad = parameter.grad
        if grad is None:
            continue
        saw_parameter = True
        detached = grad.detach()
        if not torch.isfinite(detached).all():
            module_sums[module_name] = math.inf
            finite = False
            continue
        module_sums[module_name] += float(detached.square().sum().cpu().item())

    diagnostics = {
        f"grad_norm_{module_name}": (
            math.sqrt(value) if math.isfinite(value) else math.inf
        )
        for module_name, value in sorted(module_sums.items())
    }
    if not saw_parameter:
        finite = True
    diagnostics["all_module_gradient_norms_finite"] = 1.0 if finite else 0.0
    return diagnostics


def _finite_scalar(value: torch.Tensor | float | None) -> tuple[float, bool]:
    if value is None:
        return 0.0, False
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() != 1 or not torch.isfinite(detached).all():
            return 0.0, False
        return float(detached.cpu().item()), True
    scalar = float(value)
    if not math.isfinite(scalar):
        return 0.0, False
    return scalar, True


def _gate_diagnostic(
    diagnostics: Mapping[str, float],
    key: str,
) -> tuple[float, str | None]:
    if key not in diagnostics:
        return 0.0, f"missing_diagnostic:{key}"
    scalar = _coerce_float(diagnostics.get(key))
    if not math.isfinite(scalar):
        return 0.0, f"nonfinite_diagnostic:{key}"
    return scalar, None


def _gate_flag_or_scalar_finite(
    diagnostics: Mapping[str, float],
    *,
    flag_key: str,
    value_key: str,
) -> tuple[bool, str | None]:
    if flag_key in diagnostics:
        flag_value, flag_reason = _gate_diagnostic(diagnostics, flag_key)
        if flag_reason is not None:
            return False, flag_reason
        return flag_value >= 1.0, None
    if value_key not in diagnostics:
        return False, f"missing_diagnostic:{value_key}"
    value = _coerce_float(diagnostics.get(value_key))
    if not math.isfinite(value):
        return False, f"nonfinite_diagnostic:{value_key}"
    return True, None


def _coerce_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


__all__ = [
    "PhaseGateDecision",
    "collect_mrsg_diagnostics",
    "evaluate_phase_gate",
]
