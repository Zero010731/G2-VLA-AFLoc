from __future__ import annotations

import torch
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.model import AFLocMRSG
from anaprior.models.afloc_mrsg.teacher import (
    GeometryTransform,
    MRSGTeacher,
    TeacherTarget,
    teacher_confidence,
    transform_heatmap,
    transform_phrase,
)
from tests.mrsg_test_utils import (
    add_to_parameters,
    clone_parameters,
    fake_image_features,
    fake_phrase_features,
    parameters_changed,
    test_config as make_test_config,
)


def test_ema_updates_teacher_without_gradients_and_uses_exact_formula() -> None:
    torch.manual_seed(11)
    student = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))
    teacher = MRSGTeacher(student, decay=0.9)
    before = clone_parameters(teacher.model)

    add_to_parameters(student, 1.0)
    student_after = clone_parameters(student)
    teacher.update(student)

    assert parameters_changed(before, teacher.model)
    for old_teacher, new_teacher, new_student in zip(
        before,
        teacher.model.parameters(),
        student_after,
    ):
        expected = old_teacher * 0.9 + new_student * 0.1
        assert torch.allclose(new_teacher, expected)
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())


def test_teacher_state_is_isolated_and_buffers_are_copied() -> None:
    torch.manual_seed(13)
    student = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))
    student.register_buffer("ema_probe", torch.tensor([1.0]))
    teacher = MRSGTeacher(student, decay=0.5)

    for student_param, teacher_param in zip(student.parameters(), teacher.model.parameters()):
        assert student_param.data_ptr() != teacher_param.data_ptr()

    add_to_parameters(student, 2.0)
    assert not torch.equal(next(student.parameters()), next(teacher.model.parameters()))

    student.ema_probe.fill_(7.0)
    teacher.update(student)
    assert torch.equal(teacher.model.ema_probe, torch.tensor([7.0]))


def test_teacher_forward_detaches_targets_and_keeps_mrsg_only_parameters() -> None:
    torch.manual_seed(17)
    student = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))
    teacher = MRSGTeacher(student, decay=0.95)

    output = teacher(fake_image_features(), fake_phrase_features())

    assert not output.final_heatmap.requires_grad
    assert not output.query_heatmaps.requires_grad
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())
    named_parameters = dict(teacher.named_parameters())
    assert named_parameters
    assert not any("afloc" in name.lower() for name in named_parameters)
    assert not any("confidence" in name.lower() for name in named_parameters)


def test_horizontal_flip_aligns_heatmap_and_swaps_unilateral_text() -> None:
    transform = GeometryTransform(horizontal_flip=True)
    heatmap = torch.arange(16).view(1, 1, 4, 4).float()

    assert torch.equal(transform_heatmap(heatmap, transform), heatmap.flip(-1))
    assert (
        transform_phrase("small left apical pneumothorax", transform)
        == "small right apical pneumothorax"
    )
    assert transform_phrase("right lower opacity", transform) == "left lower opacity"
    assert transform_phrase("left and right opacities", transform) == "left and right opacities"
    assert transform_phrase("upright portable chest", transform) == "upright portable chest"


def test_crop_resize_flip_alignment_handles_non_square_crops() -> None:
    heatmap = torch.arange(6 * 8).view(1, 1, 6, 8).float()
    transform = GeometryTransform(
        horizontal_flip=True,
        crop_top=1,
        crop_left=2,
        crop_height=3,
        crop_width=4,
    )
    cropped_then_flipped = heatmap[:, :, 1:4, 2:6].flip(-1)

    aligned_crop = transform_heatmap(heatmap, transform, output_size=(3, 4))
    assert torch.equal(aligned_crop, cropped_then_flipped)

    aligned_resized = transform_heatmap(
        heatmap,
        transform,
        output_size=(6, 8),
        mode="nearest",
    )
    expected = F.interpolate(cropped_then_flipped, size=(6, 8), mode="nearest")
    assert torch.equal(aligned_resized, expected)


def test_teacher_confidence_is_detached_product_of_agreement_terms() -> None:
    base = torch.tensor([[[[0.1, 0.8], [0.3, 0.6]]]], requires_grad=True)
    confidence = teacher_confidence(
        scale_heatmaps=(base, base.clone(), base.clone()),
        weak_heatmap=base,
        strong_heatmap=base.clone(),
        teacher_route_weights=torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
        student_route_weights=torch.tensor([[0.7, 0.1, 0.1, 0.1]]),
        positive_scores=torch.tensor([0.9]),
        negative_scores=torch.tensor([[0.1, 0.2]]),
        margin=0.2,
    )

    assert confidence.shape == (1, 1, 2, 2)
    assert not confidence.requires_grad
    assert torch.allclose(confidence, torch.ones_like(confidence))


def test_route_confidence_requires_finite_nonzero_route_weights() -> None:
    aligned = torch.tensor([[[[0.1, 0.8], [0.3, 0.6]]]])
    invalid_routes = (
        torch.zeros(1, 4),
        torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
        torch.tensor([[float("inf"), 0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
    )

    for bad_routes in invalid_routes:
        confidence = teacher_confidence(
            scale_heatmaps=(aligned, aligned),
            weak_heatmap=aligned,
            strong_heatmap=aligned,
            teacher_route_weights=bad_routes,
            student_route_weights=torch.full((1, 4), 0.25),
            positive_scores=torch.tensor([0.9]),
            negative_scores=torch.tensor([[0.1, 0.2]]),
            margin=0.2,
        )
        assert torch.equal(confidence, torch.zeros_like(confidence))

    valid_confidence = teacher_confidence(
        scale_heatmaps=(aligned, aligned),
        weak_heatmap=aligned,
        strong_heatmap=aligned,
        teacher_route_weights=torch.full((1, 4), 0.25),
        student_route_weights=torch.full((1, 4), 0.25),
        positive_scores=torch.tensor([0.9]),
        negative_scores=torch.tensor([[0.1, 0.2]]),
        margin=0.2,
    )
    assert torch.allclose(valid_confidence, torch.ones_like(valid_confidence))


def test_opposing_maps_and_negative_phrase_margins_lower_confident_coverage() -> None:
    aligned = torch.tensor([[[[0.05, 0.95], [0.15, 0.85]]]])
    aligned_confidence = teacher_confidence(
        scale_heatmaps=(aligned, aligned, aligned),
        weak_heatmap=aligned,
        strong_heatmap=aligned,
        teacher_route_weights=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        student_route_weights=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        positive_scores=torch.tensor([0.9]),
        negative_scores=torch.tensor([[0.1, 0.2]]),
        margin=0.2,
    )
    opposing_confidence = teacher_confidence(
        scale_heatmaps=(aligned, 1.0 - aligned),
        weak_heatmap=aligned,
        strong_heatmap=1.0 - aligned,
        teacher_route_weights=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        student_route_weights=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        positive_scores=torch.tensor([0.2]),
        negative_scores=torch.tensor([[0.6, 0.4]]),
        margin=0.2,
    )

    assert (aligned_confidence > 0.5).float().mean() > (
        opposing_confidence > 0.5
    ).float().mean()
    assert opposing_confidence.max().item() == 0.0


def test_teacher_confidence_handles_zero_and_nan_inputs_safely() -> None:
    heatmap = torch.tensor([[[[0.0, float("nan")], [0.5, 1.0]]]])
    confidence = teacher_confidence(
        scale_heatmaps=(heatmap, heatmap),
        weak_heatmap=heatmap,
        strong_heatmap=heatmap,
        teacher_route_weights=torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        student_route_weights=torch.tensor([[0.25, 0.25, 0.25, 0.25]]),
        positive_scores=torch.tensor([0.3]),
        negative_scores=torch.tensor([[0.5]]),
        margin=0.2,
    )
    target = TeacherTarget(
        final_heatmap=heatmap.nan_to_num(0.0),
        query_heatmaps=heatmap.nan_to_num(0.0).expand(1, 4, 2, 2),
        confidence=confidence,
        route_weights=torch.full((1, 4), 0.25),
    )

    assert torch.isfinite(confidence).all()
    assert confidence.max().item() == 0.0
    assert target.confidence.shape == (1, 1, 2, 2)
