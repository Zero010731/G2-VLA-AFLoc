from __future__ import annotations

import inspect

import pytest
import torch

from anaprior.models.afloc_mrsg.query_operators import (
    MorphologyQueryBank,
    straight_through_topk_gate,
)


def make_inputs(batch: int = 2, channels: int = 16, height: int = 12, width: int = 12):
    yy = torch.linspace(0.0, 1.0, height).view(1, 1, height, 1)
    xx = torch.linspace(0.0, 1.0, width).view(1, 1, 1, width)
    pyramid = torch.rand(batch, channels, height, width) + yy + xx
    edge_features = torch.zeros(batch, 3, height, width)
    edge_features[:, 0, :, width // 2 :] = 3.0
    edge_features[:, 1, height // 2 :, :] = 2.0
    edge_features[:, 2, :, :] = (xx - 0.5).abs()
    phrase_vector = torch.rand(batch, channels)
    return pyramid, edge_features, phrase_vector


def test_query_bank_returns_four_distinct_operator_outputs() -> None:
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4, focal_slots=3)
    outputs = bank(*make_inputs())

    assert [item.name for item in outputs] == ["focal", "diffuse", "boundary", "structural"]
    assert all(item.features.shape == (2, 16, 12, 12) for item in outputs)
    assert all(item.heatmap_logits.shape == (2, 1, 12, 12) for item in outputs)
    assert all(item.reliability.shape == (2, 1) for item in outputs)
    assert all(torch.isfinite(item.heatmap_logits).all() for item in outputs)
    assert len({id(module.output_head.weight) for module in bank.operators}) == 4
    assert not torch.allclose(outputs[0].heatmap_logits, outputs[1].heatmap_logits)
    assert not torch.allclose(outputs[1].features, outputs[2].features)


def test_focal_straight_through_gate_is_sparse_and_differentiable() -> None:
    logits = torch.randn(2, 1, 10, 10, requires_grad=True)
    gate = straight_through_topk_gate(logits, fraction=0.1)

    assert int((gate.detach() > 0.5).sum()) == 20
    gate.sum().backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0.0


def test_focal_operator_exposes_sparse_slot_gates_with_input_gradients() -> None:
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4, focal_slots=4, focal_fraction=0.125)
    pyramid, edge_features, phrase_vector = make_inputs(height=8, width=8)
    pyramid.requires_grad_()

    output = bank(pyramid, edge_features, phrase_vector)[0]
    slot_gates = output.auxiliary["slot_gates"]

    assert slot_gates.shape == (2, 4, 1, 8, 8)
    assert int((slot_gates.detach() > 0.5).sum()) == 2 * 4 * 8
    (output.heatmap_logits.mean() + output.features.mean()).backward()
    assert pyramid.grad is not None
    assert pyramid.grad.abs().sum().item() > 0.0


def test_diffuse_operator_has_wider_support_than_focal_operator() -> None:
    torch.manual_seed(3)
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4, focal_slots=2, focal_fraction=0.05)
    pyramid = torch.zeros(1, 16, 16, 16)
    pyramid[:, :, 8, 8] = 8.0
    edge_features = torch.zeros(1, 3, 16, 16)
    phrase_vector = torch.rand(1, 16)

    focal, diffuse, _, _ = bank(pyramid, edge_features, phrase_vector)
    focal_support = (focal.auxiliary["support"] > 0.05).sum()
    diffuse_support = (diffuse.auxiliary["support"] > 0.05).sum()

    assert diffuse_support > focal_support


def test_boundary_operator_responds_to_edge_cues_and_reports_continuity_pairs() -> None:
    torch.manual_seed(4)
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4)
    pyramid, edge_features, phrase_vector = make_inputs(batch=1, height=10, width=10)
    no_edges = torch.zeros_like(edge_features)

    boundary_with_edges = bank(pyramid, edge_features, phrase_vector)[2]
    boundary_without_edges = bank(pyramid, no_edges, phrase_vector)[2]

    assert not torch.allclose(boundary_with_edges.heatmap_logits, boundary_without_edges.heatmap_logits)
    assert set(boundary_with_edges.auxiliary) >= {"horizontal_pairs", "vertical_pairs", "edge_response"}
    assert boundary_with_edges.auxiliary["horizontal_pairs"].shape == (1, 1, 10, 9)
    assert boundary_with_edges.auxiliary["vertical_pairs"].shape == (1, 1, 9, 10)


def test_structural_operator_uses_nonlocal_left_right_relations() -> None:
    torch.manual_seed(9)
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4)
    pyramid, edge_features, phrase_vector = make_inputs(batch=1, height=8, width=12)
    changed_right = pyramid.clone()
    changed_right[:, :, :, 8:] = torch.flip(pyramid[:, :, :, :4], dims=(-1,)) + 5.0

    base = bank(pyramid, edge_features, phrase_vector)[3]
    changed = bank(changed_right, edge_features, phrase_vector)[3]
    left_delta = (base.heatmap_logits[:, :, :, :6] - changed.heatmap_logits[:, :, :, :6]).abs().mean()

    assert left_delta.item() > 1e-4
    assert base.auxiliary["relation_tokens"].shape[:3] == (1, 8, 6)


@pytest.mark.parametrize(("height", "width"), [(1, 1), (2, 1)])
def test_structural_operator_handles_degenerate_width_with_finite_gradients(
    height: int,
    width: int,
) -> None:
    torch.manual_seed(13)
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4)
    pyramid, edge_features, phrase_vector = make_inputs(batch=1, height=height, width=width)
    pyramid.requires_grad_()

    structural = bank(pyramid, edge_features, phrase_vector)[3]

    assert structural.features.shape == (1, 16, height, width)
    assert structural.heatmap_logits.shape == (1, 1, height, width)
    assert torch.isfinite(structural.features).all()
    assert torch.isfinite(structural.heatmap_logits).all()
    assert torch.isfinite(structural.reliability).all()

    (structural.heatmap_logits.mean() + structural.features.mean()).backward()

    assert pyramid.grad is not None
    assert torch.isfinite(pyramid.grad).all()
    assert pyramid.grad.abs().sum().item() > 0.0


def test_invalid_shapes_raise_value_errors() -> None:
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4)
    pyramid, edge_features, phrase_vector = make_inputs()

    with pytest.raises(ValueError, match="pyramid"):
        bank(pyramid[:, :15], edge_features, phrase_vector)
    with pytest.raises(ValueError, match="edge_features"):
        bank(pyramid, edge_features[:, :2], phrase_vector)
    with pytest.raises(ValueError, match="phrase_vector"):
        bank(pyramid, edge_features, phrase_vector[:, :15])
    with pytest.raises(ValueError, match="batch"):
        bank(pyramid, edge_features[:1], phrase_vector)


def test_query_bank_signature_excludes_spatial_supervision() -> None:
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4)
    pyramid, edge_features, phrase_vector = make_inputs()

    assert tuple(inspect.signature(bank.forward).parameters) == (
        "pyramid",
        "edge_features",
        "phrase_vector",
    )
    with pytest.raises(TypeError, match="unexpected keyword argument 'boxes'"):
        bank(pyramid, edge_features, phrase_vector, boxes=torch.zeros(2, 4))
