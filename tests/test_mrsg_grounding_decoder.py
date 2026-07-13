from __future__ import annotations

import inspect

import pytest
import torch

from anaprior.models.afloc_mrsg.dense_decoder import StandaloneDenseDecoder
from anaprior.models.afloc_mrsg.grounding_transformer import (
    MultiQuerySparsePhrasePatchGrounder,
)
from anaprior.models.afloc_mrsg.query_operators import QueryOperatorOutput


def make_query_features(
    batch: int = 2,
    queries: int = 4,
    channels: int = 16,
    height: int = 8,
    width: int = 8,
) -> torch.Tensor:
    return torch.rand(batch, queries, channels, height, width, requires_grad=True)


def make_query_outputs(query_features: torch.Tensor) -> tuple[QueryOperatorOutput, ...]:
    outputs = []
    for index, name in enumerate(("focal", "diffuse", "boundary", "structural")):
        features = query_features[:, index]
        outputs.append(
            QueryOperatorOutput(
                name=name,
                features=features,
                heatmap_logits=features.mean(dim=1, keepdim=True),
                reliability=features.mean(dim=(1, 2, 3), keepdim=True).flatten(1),
                auxiliary={},
            )
        )
    return tuple(outputs)


def test_sparse_grounding_aligns_every_query_with_real_phrase_tokens() -> None:
    module = MultiQuerySparsePhrasePatchGrounder(feature_dim=16, num_heads=4)
    output = module(
        query_features=torch.rand(2, 4, 16, 8, 8),
        word_features=torch.rand(2, 6, 16),
        attention_mask=torch.ones(2, 6, dtype=torch.bool),
    )

    assert output.query_phrase_patch_logits.shape == (2, 4, 6, 8, 8)
    assert output.query_patch_gates.shape == (2, 4, 8, 8)
    assert output.query_reconstructed_phrase.shape == (2, 4, 16)
    assert output.query_gated_features.shape == (2, 4, 16, 8, 8)


def test_sparse_grounding_topk_gates_have_cardinality_and_st_gradients() -> None:
    torch.manual_seed(7)
    module = MultiQuerySparsePhrasePatchGrounder(
        feature_dim=16,
        num_heads=4,
        topk_fraction=0.125,
    )
    query_features = make_query_features(height=8, width=8)
    word_features = torch.rand(2, 6, 16, requires_grad=True)

    output = module(query_features, word_features, torch.ones(2, 6, dtype=torch.bool))

    assert int((output.query_patch_gates.detach() > 0.5).sum()) == 2 * 4 * 8
    loss = (
        output.query_gated_features.mean()
        + output.query_reconstructed_phrase.mean()
        + output.query_phrase_patch_logits.mean()
    )
    loss.backward()

    assert query_features.grad is not None
    assert word_features.grad is not None
    assert query_features.grad.abs().sum().item() > 0.0
    assert word_features.grad.abs().sum().item() > 0.0
    assert module.word_projection.weight.grad is not None
    assert module.word_projection.weight.grad.abs().sum().item() > 0.0


def test_sparse_grounding_rejects_invalid_masks_and_shapes() -> None:
    module = MultiQuerySparsePhrasePatchGrounder(feature_dim=16, num_heads=4)
    query_features = torch.rand(2, 4, 16, 8, 8)
    word_features = torch.rand(2, 6, 16)

    with pytest.raises(ValueError, match="query_features"):
        module(query_features[:, :3], word_features, torch.ones(2, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="word_features"):
        module(query_features, word_features[:, :, :15], torch.ones(2, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="attention_mask"):
        module(query_features, word_features, torch.ones(2, 5, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        module(query_features, word_features, torch.zeros(2, 6, dtype=torch.bool))


def test_sparse_grounding_each_query_depends_on_phrase_tokens() -> None:
    torch.manual_seed(11)
    module = MultiQuerySparsePhrasePatchGrounder(feature_dim=16, num_heads=4)
    query_features = make_query_features(batch=1, height=6, width=6)
    word_features = torch.rand(1, 5, 16, requires_grad=True)

    output = module(query_features, word_features, torch.ones(1, 5, dtype=torch.bool))
    per_query = output.query_gated_features.mean(dim=(2, 3, 4)).flatten()

    for query_index in range(4):
        grad = torch.autograd.grad(
            per_query[query_index],
            word_features,
            retain_graph=True,
            allow_unused=False,
        )[0]
        assert grad.abs().sum().item() > 0.0


def test_decoder_output_is_independent_of_dcem_or_base_heatmap() -> None:
    signature = inspect.signature(StandaloneDenseDecoder.forward)
    assert "base_hmap" not in signature.parameters
    assert "dcem" not in signature.parameters
    assert "fallback" not in signature.parameters


def test_decoder_returns_direct_normalized_heatmap_with_gradients() -> None:
    torch.manual_seed(17)
    decoder = StandaloneDenseDecoder(feature_dim=16)
    pyramid = torch.rand(2, 16, 8, 8, requires_grad=True)
    query_features = make_query_features()
    query_outputs = make_query_outputs(query_features)
    route_weights = torch.softmax(torch.rand(2, 4, requires_grad=True), dim=-1)
    query_patch_gates = torch.rand(2, 4, 8, 8, requires_grad=True)

    heatmap = decoder(pyramid, query_outputs, route_weights, query_patch_gates)

    assert heatmap.shape == (2, 1, 8, 8)
    assert torch.isfinite(heatmap).all()
    assert heatmap.min().item() >= 0.0
    assert heatmap.max().item() <= 1.0

    heatmap.mean().backward()
    assert pyramid.grad is not None
    assert query_features.grad is not None
    assert query_patch_gates.grad is not None
    assert decoder.output_head.weight.grad is not None
    assert pyramid.grad.abs().sum().item() > 0.0
    assert query_features.grad.abs().sum().item() > 0.0
    assert query_patch_gates.grad.abs().sum().item() > 0.0


def test_decoder_is_non_identity_and_has_no_base_fallback_behavior() -> None:
    torch.manual_seed(23)
    decoder = StandaloneDenseDecoder(feature_dim=16)
    pyramid = torch.rand(1, 16, 8, 8)
    query_features = torch.rand(1, 4, 16, 8, 8)
    query_outputs = make_query_outputs(query_features)
    route_weights = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    query_patch_gates = torch.rand(1, 4, 8, 8)

    heatmap = decoder(pyramid, query_outputs, route_weights, query_patch_gates)
    first_query = torch.sigmoid(query_outputs[0].heatmap_logits)
    route_only = (
        torch.stack([torch.sigmoid(item.heatmap_logits[:, 0]) for item in query_outputs], dim=1)
        * route_weights.view(1, 4, 1, 1)
    ).sum(dim=1, keepdim=True)

    assert not torch.allclose(heatmap, first_query)
    assert not torch.allclose(heatmap, route_only)


def test_decoder_rejects_invalid_shapes() -> None:
    decoder = StandaloneDenseDecoder(feature_dim=16)
    pyramid = torch.rand(2, 16, 8, 8)
    query_outputs = make_query_outputs(torch.rand(2, 4, 16, 8, 8))
    route_weights = torch.full((2, 4), 0.25)
    query_patch_gates = torch.rand(2, 4, 8, 8)

    with pytest.raises(ValueError, match="pyramid"):
        decoder(pyramid[:, :15], query_outputs, route_weights, query_patch_gates)
    with pytest.raises(ValueError, match="query_outputs"):
        decoder(pyramid, query_outputs[:3], route_weights, query_patch_gates)
    with pytest.raises(ValueError, match="route_weights"):
        decoder(pyramid, query_outputs, route_weights[:, :3], query_patch_gates)
    with pytest.raises(ValueError, match="query_patch_gates"):
        decoder(pyramid, query_outputs, route_weights, query_patch_gates[:, :, :7])
