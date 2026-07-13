from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from anaprior.models.afloc_mrsg.contracts import MRSGOutput
from anaprior.models.afloc_mrsg.diagnostics import (
    PhaseGateDecision,
    collect_mrsg_diagnostics,
    evaluate_phase_gate,
)
from anaprior.models.afloc_mrsg.teacher import TeacherTarget
from tests.mrsg_test_utils import healthy_grounding_diagnostics


def _student_output() -> MRSGOutput:
    query_heatmaps = torch.tensor(
        [
            [
                [[0.90, 0.75], [0.20, 0.10]],
                [[0.10, 0.25], [0.70, 0.80]],
                [[0.15, 0.85], [0.80, 0.20]],
                [[0.70, 0.30], [0.25, 0.90]],
            ],
            [
                [[0.80, 0.65], [0.25, 0.15]],
                [[0.15, 0.20], [0.75, 0.70]],
                [[0.20, 0.75], [0.85, 0.25]],
                [[0.65, 0.35], [0.20, 0.85]],
            ],
        ],
        dtype=torch.float32,
    )
    return MRSGOutput(
        final_heatmap=torch.tensor(
            [
                [[[0.85, 0.20], [0.15, 0.55]]],
                [[[0.75, 0.25], [0.10, 0.60]]],
            ],
            dtype=torch.float32,
        ),
        query_heatmaps=query_heatmaps,
        query_route_weights=torch.tensor(
            [
                [0.50, 0.20, 0.20, 0.10],
                [0.30, 0.30, 0.20, 0.20],
            ],
            dtype=torch.float32,
        ),
        query_reliability=torch.full((2, 4), 0.5),
        phrase_patch_logits=torch.zeros(2, 3, 2, 2),
    )


def test_phase_gate_rejects_constant_maps_and_single_route_collapse() -> None:
    diagnostics = {
        "heatmap_std": 0.0,
        "max_route_utilization": 0.99,
        "query_pairwise_cosine": 1.0,
        "positive_negative_margin": 0.0,
        "teacher_confident_coverage": 0.2,
        "all_module_gradient_norms_finite": 1.0,
    }

    decision = evaluate_phase_gate("grounding", diagnostics)

    assert decision == PhaseGateDecision(
        phase="grounding",
        passed=False,
        reasons=(
            "heatmap_collapse",
            "route_collapse",
            "query_collapse",
            "nonpositive_phrase_margin",
        ),
        diagnostics=diagnostics,
    )


def test_phase_gate_accepts_noncollapsed_grounding_diagnostics() -> None:
    diagnostics = {
        **healthy_grounding_diagnostics(),
        "all_module_gradient_norms_finite": 1.0,
    }

    assert evaluate_phase_gate("grounding", diagnostics).passed is True


def test_locality_gate_requires_reconstruction_improvement_and_finite_gradients() -> None:
    diagnostics = {
        **healthy_grounding_diagnostics(),
        "all_module_gradient_norms_finite": 0.0,
        "masked_reconstruction_loss": 0.8,
        "untrained_masked_reconstruction_loss": 0.7,
        "locality_reconstruction_ratio": 0.8 / 0.7,
    }

    decision = evaluate_phase_gate("locality", diagnostics)

    assert decision.passed is False
    assert decision.reasons == (
        "nonfinite_gradient_norms",
        "locality_reconstruction_not_improved",
    )


def test_consistency_gate_rejects_teacher_coverage_out_of_range_and_margin_regression() -> None:
    diagnostics = {
        **healthy_grounding_diagnostics(),
        "all_module_gradient_norms_finite": 1.0,
        "teacher_confident_coverage": 0.0,
        "phase_b_positive_negative_margin": 0.35,
        "positive_negative_margin": 0.20,
        "consistency_margin_delta_vs_phase_b": -0.15,
    }

    decision = evaluate_phase_gate("consistency", diagnostics)

    assert decision.passed is False
    assert decision.reasons == (
        "teacher_confident_coverage_out_of_range",
        "consistency_margin_below_phase_b",
    )


def test_collect_mrsg_diagnostics_returns_anti_collapse_metrics() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    loss = model(torch.tensor([[0.2, 0.1, 0.3]], dtype=torch.float32)).sum()
    loss.backward()

    teacher_target = TeacherTarget(
        final_heatmap=torch.ones(2, 1, 2, 2),
        query_heatmaps=torch.ones(2, 4, 2, 2),
        confidence=torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [1.0, 0.0]]],
            ],
            dtype=torch.float32,
        ),
        route_weights=torch.full((2, 4), 0.25),
    )

    diagnostics = collect_mrsg_diagnostics(
        output=_student_output(),
        positive_scores=torch.tensor([0.9, 0.6], dtype=torch.float32),
        negative_scores=torch.tensor([[0.2, 0.3], [0.1, 0.2]], dtype=torch.float32),
        teacher_target=teacher_target,
        masked_reconstruction_loss=torch.tensor(0.4),
        untrained_masked_reconstruction_loss=torch.tensor(0.8),
        model=model,
        phase_b_positive_negative_margin=0.15,
    )

    assert diagnostics["heatmap_std"] > 1.0e-3
    assert diagnostics["heatmap_variance"] > 0.0
    assert diagnostics["max_route_utilization"] == pytest.approx(0.4)
    assert diagnostics["route_utilization_entropy"] > 0.0
    assert diagnostics["query_pairwise_cosine"] < 0.95
    assert diagnostics["query_diversity"] == pytest.approx(1.0 - diagnostics["query_pairwise_cosine"])
    assert diagnostics["positive_negative_margin"] == pytest.approx(0.55)
    assert diagnostics["teacher_confident_coverage"] == pytest.approx(0.25)
    assert diagnostics["all_module_gradient_norms_finite"] == 1.0
    assert diagnostics["locality_reconstruction_ratio"] == pytest.approx(0.5)
    assert diagnostics["consistency_margin_delta_vs_phase_b"] == pytest.approx(0.40)
    assert diagnostics["grad_norm_0"] > 0.0
    assert diagnostics["grad_norm_2"] > 0.0


def test_collect_mrsg_diagnostics_marks_nonfinite_gradients_and_empty_negatives_as_failures() -> None:
    model = nn.Sequential(nn.Linear(2, 2))
    model[0].weight.grad = torch.full_like(model[0].weight, float("inf"))
    model[0].bias.grad = torch.zeros_like(model[0].bias)

    diagnostics = collect_mrsg_diagnostics(
        output=_student_output(),
        positive_scores=torch.tensor([0.4], dtype=torch.float32),
        negative_scores=torch.empty(1, 0),
        teacher_target=TeacherTarget(
            final_heatmap=torch.ones(2, 1, 2, 2),
            query_heatmaps=torch.ones(2, 4, 2, 2),
            confidence=torch.zeros(2, 1, 2, 2),
            route_weights=torch.full((2, 4), 0.25),
        ),
        masked_reconstruction_loss=float("nan"),
        untrained_masked_reconstruction_loss=1.0,
        model=model,
    )

    assert diagnostics["positive_negative_margin"] == 0.0
    assert diagnostics["teacher_confident_coverage"] == 0.0
    assert diagnostics["all_module_gradient_norms_finite"] == 0.0
    assert math.isinf(diagnostics["grad_norm_0"])
    assert diagnostics["masked_reconstruction_loss"] == 0.0
    assert diagnostics["locality_reconstruction_ratio"] == 0.0
