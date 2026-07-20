from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from anaprior.models.afloc_mrsg.contracts import MRSGOutput
from anaprior.models.afloc_mrsg import losses, teacher
from anaprior.models.afloc_mrsg.losses import (
    anchor_patch_grounding_loss,
    cross_view_patch_consistency_loss,
    MRSGGroupedLoss,
    _pairwise_query_cosine,
    _total_variation,
    TeacherTarget,
    compute_mrsg_loss,
    cross_modal_grounding_loss,
    masked_patch_distillation_loss,
    phrase_patch_refinement_loss,
    residual_stability_loss,
    phrase_swap_correction_contrast_loss,
    query_regularization_loss,
    teacher_equivariance_loss,
)


def test_phrase_patch_refinement_drives_final_heatmap_without_target_gradients() -> None:
    final = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
    phrase_logits = torch.tensor(
        [[[[3.0, 2.0], [-2.0, -3.0]], [[2.0, 1.0], [-1.0, -2.0]]]],
        requires_grad=True,
    )

    loss = phrase_patch_refinement_loss(final, phrase_logits)
    loss.backward()

    assert loss.item() > 0.0
    assert final.grad is not None
    assert final.grad.abs().sum().item() > 0.0
    assert phrase_logits.grad is None


def test_residual_stability_penalizes_energy_and_saturation() -> None:
    calm = torch.tensor([[[[0.1, -0.1], [0.2, -0.2]]]], requires_grad=True)
    saturated = torch.tensor([[[[0.1, 4.0], [-5.0, -0.2]]]], requires_grad=True)

    calm_loss = residual_stability_loss(calm, cap=2.0)
    saturated_loss = residual_stability_loss(saturated, cap=2.0)
    saturated_loss.backward()

    assert saturated_loss > calm_loss
    assert saturated.grad is not None
    assert saturated.grad.abs().sum().item() > 0.0


def test_phrase_swap_contrast_penalizes_identical_corrections() -> None:
    positive = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
    identical = positive[:, None].clone().requires_grad_(True)
    distinct = positive.flip(-1)[:, None].clone().requires_grad_(True)
    mask = torch.tensor([[True]])

    identical_loss = phrase_swap_correction_contrast_loss(positive, identical, mask)
    distinct_loss = phrase_swap_correction_contrast_loss(positive, distinct, mask)

    assert identical_loss > distinct_loss


def test_anchor_patch_grounding_prefers_anchor_supported_region() -> None:
    anchor = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    confidence = torch.ones_like(anchor)
    good = torch.tensor([[[[0.9, 0.8], [0.1, 0.1]]]])
    bad = torch.tensor([[[[0.1, 0.1], [0.9, 0.8]]]])

    assert anchor_patch_grounding_loss(good, anchor, confidence) < anchor_patch_grounding_loss(bad, anchor, confidence)


def test_cross_view_patch_consistency_uses_confidence_weight() -> None:
    first = torch.zeros(1, 1, 2, 2)
    second = torch.ones_like(first)
    low_confidence = torch.zeros_like(first)
    high_confidence = torch.ones_like(first)

    assert cross_view_patch_consistency_loss(first, second, low_confidence) < cross_view_patch_consistency_loss(first, second, high_confidence)
from tests.mrsg_test_utils import uniform_routes


@dataclass(frozen=True)
class LossWeights:
    w_ground: float = 1.0
    w_teacher: float = 1.0
    w_mask: float = 1.0
    w_query: float = 1.0


def _student_output(batch: int = 2, height: int = 4, width: int = 4) -> MRSGOutput:
    final_heatmap = torch.rand(batch, 1, height, width, requires_grad=True)
    query_heatmaps = torch.rand(batch, 4, height, width, requires_grad=True)
    return MRSGOutput(
        final_heatmap=final_heatmap,
        query_heatmaps=query_heatmaps,
        query_route_weights=torch.softmax(
            torch.randn(batch, 4, requires_grad=True),
            dim=-1,
        ),
        query_reliability=torch.rand(batch, 4, requires_grad=True),
        phrase_patch_logits=torch.randn(batch, 3, height, width, requires_grad=True),
        masked_predictions={
            name: torch.randn(batch, 5, height, width, requires_grad=True)
            for name in ("l2", "l", "lf")
        },
        source_targets={
            name: torch.randn(batch, 5, height, width)
            for name in ("l2", "l", "lf")
        },
        patch_mask=torch.ones(batch, 1, height, width, dtype=torch.bool),
        query_reconstructed_phrase=torch.randn(batch, 4, 6, requires_grad=True),
        query_patch_gates=torch.rand(batch, 4, height, width, requires_grad=True),
    )


def _teacher_target(batch: int = 2, height: int = 4, width: int = 4) -> TeacherTarget:
    return TeacherTarget(
        final_heatmap=torch.rand(batch, 1, height, width, requires_grad=True),
        query_heatmaps=torch.rand(batch, 4, height, width, requires_grad=True),
        confidence=torch.ones(batch, 1, height, width, requires_grad=True),
        route_weights=torch.full((batch, 4), 0.25),
    )


def test_teacher_target_is_canonical_between_teacher_and_losses() -> None:
    assert teacher.TeacherTarget is losses.TeacherTarget


def test_grounding_loss_rewards_positive_phrase_margin() -> None:
    good = cross_modal_grounding_loss(
        positive_scores=torch.tensor([0.8, 0.7]),
        negative_scores=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
        reconstructed_phrase=torch.eye(2).unsqueeze(1).expand(2, 4, 2),
        target_phrase=torch.eye(2),
        margin=0.2,
    )
    bad = cross_modal_grounding_loss(
        positive_scores=torch.tensor([0.2, 0.2]),
        negative_scores=torch.tensor([[0.5, 0.4], [0.6, 0.5]]),
        reconstructed_phrase=torch.zeros(2, 4, 2),
        target_phrase=torch.eye(2),
        margin=0.2,
    )
    assert good < bad


def test_query_regularization_detects_constant_and_identical_maps() -> None:
    collapsed = torch.ones(4, 4, 8, 8) * 0.5
    diverse = torch.zeros(4, 4, 8, 8)
    diverse[:, 0, 3:5, 3:5] = 1.0
    diverse[:, 1] = torch.linspace(0.2, 0.8, 8).view(1, 1, 8).expand(4, 8, 8)
    diverse[:, 2, :, :4] = 0.2
    diverse[:, 2, :, 4:] = 0.8
    diverse[:, 3] = (
        torch.tensor([0.1, 0.3, 0.6, 0.9, 0.9, 0.6, 0.3, 0.1])
        .view(1, 1, 8)
        .expand(4, 8, 8)
    )
    assert query_regularization_loss(
        collapsed,
        uniform_routes(4),
    ) > query_regularization_loss(diverse, uniform_routes(4))


def test_query_group_does_not_pull_final_heatmap_toward_query_maps() -> None:
    student = _student_output(batch=1)
    final_logits = torch.full((1, 1, 4, 4), 4.0, requires_grad=True)
    collapsed_final = torch.sigmoid(final_logits)
    student = MRSGOutput(
        final_heatmap=collapsed_final,
        query_heatmaps=student.query_heatmaps,
        query_route_weights=student.query_route_weights,
        query_reliability=student.query_reliability,
        phrase_patch_logits=student.phrase_patch_logits,
        masked_predictions=student.masked_predictions,
        source_targets=student.source_targets,
        patch_mask=student.patch_mask,
        query_reconstructed_phrase=student.query_reconstructed_phrase,
        query_patch_gates=student.query_patch_gates,
    )

    loss = compute_mrsg_loss(
        student=student,
        positive_scores=torch.tensor([0.7]),
        negative_scores=torch.tensor([[0.2]]),
        pyramid=student,
        teacher_target=None,
        config=LossWeights(w_ground=0.0, w_teacher=0.0, w_mask=0.0, w_query=1.0),
    )
    loss.total.backward()

    assert final_logits.grad is not None
    assert final_logits.grad.abs().sum().item() == 0.0


def test_compute_mrsg_loss_exposes_exactly_four_top_level_groups() -> None:
    student = _student_output()
    loss = compute_mrsg_loss(
        student=student,
        positive_scores=torch.rand(2, requires_grad=True),
        negative_scores=torch.rand(2, 2, requires_grad=True),
        pyramid=student,
        teacher_target=_teacher_target(),
        config=LossWeights(),
        target_phrase=torch.rand(2, 6),
    )

    assert isinstance(loss, MRSGGroupedLoss)
    assert loss.total.shape == ()
    assert set(loss.diagnostics) >= {
        "grounding",
        "teacher",
        "mask",
        "query",
        "positive_negative_margin",
        "teacher_confident_coverage",
    }
    assert not any(key.startswith("w_") for key in loss.diagnostics)


def test_compute_mrsg_loss_requires_explicit_target_when_grounding_weight_active() -> None:
    student = _student_output(batch=1)

    with pytest.raises(ValueError, match="target_phrase"):
        compute_mrsg_loss(
            student=student,
            positive_scores=torch.tensor([0.7], requires_grad=True),
            negative_scores=torch.tensor([[0.2]], requires_grad=True),
            pyramid=student,
            teacher_target=None,
            config=LossWeights(w_teacher=0.0, w_mask=0.0, w_query=0.0),
        )


def test_explicit_afloc_target_changes_grounding_loss_and_reconstruction_gradients() -> None:
    positive = torch.tensor([0.75], requires_grad=True)
    negative = torch.tensor([[0.20, 0.35]], requires_grad=True)
    target_a = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    target_b = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

    def build_student() -> MRSGOutput:
        student = _student_output(batch=1)
        query_reconstructed_phrase = torch.tensor(
            [[[0.8, 0.1, 0.1, 0.0, 0.0, 0.0]] * 4],
            dtype=torch.float32,
            requires_grad=True,
        )
        return MRSGOutput(
            final_heatmap=student.final_heatmap,
            query_heatmaps=student.query_heatmaps,
            query_route_weights=student.query_route_weights,
            query_reliability=student.query_reliability,
            phrase_patch_logits=student.phrase_patch_logits,
            masked_predictions=student.masked_predictions,
            source_targets=student.source_targets,
            patch_mask=student.patch_mask,
            query_reconstructed_phrase=query_reconstructed_phrase,
            query_patch_gates=student.query_patch_gates,
        )

    student_a = build_student()
    loss_a = compute_mrsg_loss(
        student=student_a,
        positive_scores=positive,
        negative_scores=negative,
        pyramid=student_a,
        teacher_target=None,
        config=LossWeights(w_teacher=0.0, w_mask=0.0, w_query=0.0),
        target_phrase=target_a,
    )
    loss_a.total.backward()
    grad_a = student_a.query_reconstructed_phrase.grad.detach().clone()

    positive_b = positive.detach().clone().requires_grad_(True)
    negative_b = negative.detach().clone().requires_grad_(True)
    student_b = build_student()
    loss_b = compute_mrsg_loss(
        student=student_b,
        positive_scores=positive_b,
        negative_scores=negative_b,
        pyramid=student_b,
        teacher_target=None,
        config=LossWeights(w_teacher=0.0, w_mask=0.0, w_query=0.0),
        target_phrase=target_b,
    )
    loss_b.total.backward()
    grad_b = student_b.query_reconstructed_phrase.grad.detach().clone()

    assert not torch.isclose(loss_a.total.detach(), loss_b.total.detach())
    assert not torch.allclose(grad_a, grad_b)


def test_query_regularization_uses_exact_equal_component_ratio() -> None:
    query_maps = torch.tensor(
        [
            [
                [[0.90, 0.80], [0.70, 0.60]],
                [[0.10, 0.40], [0.80, 0.20]],
                [[0.30, 0.90], [0.50, 0.70]],
                [[0.20, 0.60], [0.80, 0.40]],
            ],
            [
                [[0.60, 0.55], [0.50, 0.45]],
                [[0.90, 0.30], [0.20, 0.70]],
                [[0.40, 0.10], [0.80, 0.60]],
                [[0.75, 0.25], [0.35, 0.95]],
            ],
        ],
        dtype=torch.float32,
    )
    route_weights = torch.tensor(
        [
            [0.55, 0.15, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )

    route_balance = (route_weights.mean(dim=0) - 0.25).square().mean()
    diversity = _pairwise_query_cosine(query_maps).mean()
    noncollapse = torch.relu(0.02 - query_maps.var(dim=(-2, -1), unbiased=False)).mean()

    focal = query_maps[:, 0:1]
    diffuse = query_maps[:, 1:2]
    boundary = query_maps[:, 2:3]
    structural = query_maps[:, 3:4]
    structure = (
        torch.relu(focal.mean(dim=(-2, -1)) - 0.35).mean()
        + _total_variation(diffuse)
        + _total_variation(boundary)
        + (structural - structural.flip(-1)).abs().mean()
    )

    expected = 0.25 * (route_balance + diversity + noncollapse + structure)
    actual = query_regularization_loss(query_maps, route_weights)

    assert torch.isclose(actual, expected)


def test_group_losses_produce_finite_gradients_without_target_gradients() -> None:
    positive = torch.tensor([0.3, 0.7], requires_grad=True)
    negative = torch.tensor([[0.6, 0.5], [0.2, 0.1]], requires_grad=True)
    reconstructed = torch.randn(2, 4, 6, requires_grad=True)
    target_phrase = torch.randn(2, 6, requires_grad=True)
    grounding = cross_modal_grounding_loss(
        positive,
        negative,
        reconstructed,
        target_phrase,
    )

    student_final = torch.rand(2, 1, 4, 4, requires_grad=True)
    student_queries = torch.rand(2, 4, 4, 4, requires_grad=True)
    teacher = _teacher_target()
    teacher_loss = teacher_equivariance_loss(student_final, student_queries, teacher)

    predictions = {
        name: torch.randn(2, 3, 4, 4, requires_grad=True)
        for name in ("l2", "l", "lf")
    }
    targets = {
        name: torch.randn(2, 3, 4, 4, requires_grad=True)
        for name in ("l2", "l", "lf")
    }
    mask_loss = masked_patch_distillation_loss(
        predictions,
        targets,
        torch.ones(2, 1, 4, 4, dtype=torch.bool),
    )

    query_maps = torch.rand(2, 4, 4, 4, requires_grad=True)
    route_logits = torch.randn(2, 4, requires_grad=True)
    route_weights = torch.softmax(route_logits, dim=-1)
    query_loss = query_regularization_loss(query_maps, route_weights)

    total = grounding + teacher_loss + mask_loss + query_loss
    total.backward()

    for tensor in (positive, negative, reconstructed, student_final, student_queries, query_maps):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum().item() > 0.0
    assert route_logits.grad is not None
    assert torch.isfinite(route_logits.grad).all()
    assert route_logits.grad.abs().sum().item() > 0.0

    assert target_phrase.grad is None
    for tensor in teacher.__dict__.values():
        assert tensor.grad is None
    for tensor in targets.values():
        assert tensor.grad is None


def test_grounding_loss_uses_negative_mask_for_padding_invariant_counterfactuals() -> None:
    reconstructed = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]], requires_grad=True)
    target_phrase = torch.tensor([[1.0, 0.0]], requires_grad=True)
    positive = torch.tensor([0.8], requires_grad=True)

    compact = cross_modal_grounding_loss(
        positive_scores=positive,
        negative_scores=torch.tensor([[0.4]], requires_grad=True),
        negative_mask=torch.tensor([[True]]),
        reconstructed_phrase=reconstructed,
        target_phrase=target_phrase,
        margin=0.2,
    )
    padded = cross_modal_grounding_loss(
        positive_scores=positive.detach().clone().requires_grad_(True),
        negative_scores=torch.tensor([[0.4, 0.95, 0.05]], requires_grad=True),
        negative_mask=torch.tensor([[True, False, False]]),
        reconstructed_phrase=reconstructed.detach().clone().requires_grad_(True),
        target_phrase=target_phrase,
        margin=0.2,
    )

    assert torch.isclose(compact.detach(), padded.detach())


def test_grounding_loss_handles_empty_negatives_without_breaking_autograd() -> None:
    positive = torch.tensor([0.8], requires_grad=True)
    reconstructed = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    loss = cross_modal_grounding_loss(
        positive_scores=positive,
        negative_scores=torch.empty(1, 0, requires_grad=True),
        negative_mask=torch.empty(1, 0, dtype=torch.bool),
        reconstructed_phrase=reconstructed,
        target_phrase=torch.tensor([[0.0, 1.0]], requires_grad=True),
        margin=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss.detach())
    assert positive.grad is not None
    assert reconstructed.grad is not None
    assert positive.grad.abs().sum().item() > 0.0
    assert reconstructed.grad.abs().sum().item() > 0.0


def test_absent_and_zero_confidence_teacher_are_backward_safe() -> None:
    student_final = torch.rand(2, 1, 4, 4, requires_grad=True)
    student_queries = torch.rand(2, 4, 4, 4, requires_grad=True)

    absent = teacher_equivariance_loss(student_final, student_queries, None)
    absent.backward(retain_graph=True)
    assert student_final.grad is not None
    assert student_final.grad.abs().sum().item() == 0.0

    student_final.grad.zero_()
    zero_conf = TeacherTarget(
        final_heatmap=torch.rand(2, 1, 4, 4, requires_grad=True),
        query_heatmaps=torch.rand(2, 4, 4, 4, requires_grad=True),
        confidence=torch.zeros(2, 1, 4, 4, requires_grad=True),
        route_weights=torch.full((2, 4), 0.25),
    )
    loss = teacher_equivariance_loss(student_final, student_queries, zero_conf)
    loss.backward()
    assert student_final.grad.abs().sum().item() == 0.0
    assert zero_conf.final_heatmap.grad is None
    assert zero_conf.query_heatmaps.grad is None
    assert zero_conf.confidence.grad is None


def test_empty_mask_distillation_is_backward_safe_and_zero() -> None:
    predictions = {
        name: torch.randn(2, 3, 4, 4, requires_grad=True)
        for name in ("l2", "l", "lf")
    }
    targets = {
        name: torch.randn(2, 3, 4, 4, requires_grad=True)
        for name in ("l2", "l", "lf")
    }

    loss = masked_patch_distillation_loss(
        predictions,
        targets,
        torch.zeros(2, 1, 4, 4, dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    for tensor in predictions.values():
        assert tensor.grad is not None
        assert tensor.grad.abs().sum().item() == 0.0
    for tensor in targets.values():
        assert tensor.grad is None


def test_loss_validation_rejects_bad_shapes_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="positive_scores"):
        cross_modal_grounding_loss(
            torch.rand(2, 1),
            torch.rand(2, 2),
            torch.rand(2, 4, 3),
            torch.rand(2, 3),
        )

    with pytest.raises(ValueError, match="finite"):
        query_regularization_loss(
            torch.full((2, 4, 4, 4), float("nan")),
            uniform_routes(2),
        )

    with pytest.raises(ValueError, match="route_weights must have shape \\[B,4\\]"):
        teacher_equivariance_loss(
            torch.rand(2, 1, 4, 4),
            torch.rand(2, 4, 4, 4),
            TeacherTarget(
                final_heatmap=torch.rand(2, 1, 4, 4),
                query_heatmaps=torch.rand(2, 4, 4, 4),
                confidence=torch.ones(2, 1, 4, 4),
                route_weights=torch.ones(2, 3),
            ),
        )

    with pytest.raises(ValueError, match="route_weights must contain finite values"):
        teacher_equivariance_loss(
            torch.rand(2, 1, 4, 4),
            torch.rand(2, 4, 4, 4),
            TeacherTarget(
                final_heatmap=torch.rand(2, 1, 4, 4),
                query_heatmaps=torch.rand(2, 4, 4, 4),
                confidence=torch.ones(2, 1, 4, 4),
                route_weights=torch.tensor(
                    [[float("nan"), 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]]
                ),
            ),
        )

    with pytest.raises(ValueError, match="route_weights must be detached"):
        teacher_equivariance_loss(
            torch.rand(2, 1, 4, 4),
            torch.rand(2, 4, 4, 4),
            TeacherTarget(
                final_heatmap=torch.rand(2, 1, 4, 4),
                query_heatmaps=torch.rand(2, 4, 4, 4),
                confidence=torch.ones(2, 1, 4, 4),
                route_weights=torch.full((2, 4), 0.25, requires_grad=True),
            ),
        )
