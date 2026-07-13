from __future__ import annotations

import inspect

import pytest
import torch

from anaprior.models.afloc_mrsg.contracts import PhraseFeatureBatch
from anaprior.models.afloc_mrsg.phrase_router import (
    PhraseMorphologyRouter,
    route_balance_loss,
)
from tests.mrsg_test_utils import fake_phrase_features


def make_trainable_phrase_features(
    batch: int = 2,
    tokens: int = 7,
    text_dim: int = 24,
) -> PhraseFeatureBatch:
    return PhraseFeatureBatch(
        word_embeddings=torch.rand(batch, tokens, text_dim, requires_grad=True),
        sentence_embedding=torch.rand(batch, text_dim, requires_grad=True),
        disease_description_embedding=torch.rand(batch, text_dim, requires_grad=True),
        attention_mask=torch.ones(batch, tokens, dtype=torch.bool),
    )


def test_router_uses_text_features_without_category_ids() -> None:
    router = PhraseMorphologyRouter(text_dim=24, feature_dim=16, num_heads=4)

    output = router(fake_phrase_features(batch=3, tokens=7, text_dim=24))

    assert output.projected_words.shape == (3, 7, 16)
    assert output.phrase_vector.shape == (3, 16)
    assert output.route_logits.shape == (3, 4)
    assert output.route_weights.shape == (3, 4)
    assert torch.allclose(output.route_weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_router_route_balance_penalizes_single_query_collapse() -> None:
    collapsed = torch.tensor([[0.99, 0.003, 0.003, 0.004]]).repeat(8, 1)
    balanced = torch.full((8, 4), 0.25)

    assert route_balance_loss(collapsed) > route_balance_loss(balanced)


def test_masked_tokens_do_not_affect_outputs() -> None:
    torch.manual_seed(5)
    router = PhraseMorphologyRouter(text_dim=24, feature_dim=16, num_heads=4)
    base = fake_phrase_features(batch=1, tokens=6, text_dim=24)
    attention_mask = torch.tensor([[True, True, True, False, False, False]])
    changed = PhraseFeatureBatch(
        word_embeddings=base.word_embeddings.clone(),
        sentence_embedding=base.sentence_embedding.clone(),
        disease_description_embedding=base.disease_description_embedding.clone(),
        attention_mask=attention_mask,
    )
    changed.word_embeddings[:, 3:, :] += 1000.0

    router.eval()
    with torch.no_grad():
        base_output = router(
            PhraseFeatureBatch(
                word_embeddings=base.word_embeddings.clone(),
                sentence_embedding=base.sentence_embedding.clone(),
                disease_description_embedding=base.disease_description_embedding.clone(),
                attention_mask=attention_mask,
            )
        )
        changed_output = router(changed)

    assert torch.allclose(base_output.projected_words[:, :3], changed_output.projected_words[:, :3])
    assert torch.allclose(base_output.phrase_vector, changed_output.phrase_vector)
    assert torch.allclose(base_output.route_logits, changed_output.route_logits)
    assert torch.allclose(base_output.route_weights, changed_output.route_weights)


def test_only_router_parameters_receive_gradients() -> None:
    router = PhraseMorphologyRouter(text_dim=24, feature_dim=16, num_heads=4)
    features = make_trainable_phrase_features()

    output = router(features)
    loss = output.phrase_vector.square().mean() + output.route_weights.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert features.word_embeddings.grad is None
    assert features.sentence_embedding.grad is None
    assert features.disease_description_embedding.grad is None
    assert router.word_projection.weight.grad is not None
    assert router.route_head.weight.grad is not None
    assert router.route_head.weight.grad.abs().sum().item() > 0.0


def test_invalid_shapes_raise_value_errors() -> None:
    router = PhraseMorphologyRouter(text_dim=24, feature_dim=16, num_heads=4)

    with pytest.raises(ValueError, match="word_embeddings"):
        router(
            PhraseFeatureBatch(
                word_embeddings=torch.rand(2, 24),
                sentence_embedding=torch.rand(2, 24),
                disease_description_embedding=torch.rand(2, 24),
                attention_mask=torch.ones(2, 7, dtype=torch.bool),
            )
        )

    with pytest.raises(ValueError, match="attention_mask"):
        router(
            PhraseFeatureBatch(
                word_embeddings=torch.rand(2, 7, 24),
                sentence_embedding=torch.rand(2, 24),
                disease_description_embedding=torch.rand(2, 24),
                attention_mask=torch.ones(2, 6, dtype=torch.bool),
            )
        )


def test_router_signature_excludes_disease_ids_and_spatial_vectors() -> None:
    router = PhraseMorphologyRouter(text_dim=24, feature_dim=16, num_heads=4)

    assert tuple(inspect.signature(router.forward).parameters) == ("phrase_features",)
    with pytest.raises(TypeError, match="unexpected keyword argument 'disease_ids'"):
        router(fake_phrase_features(), disease_ids=torch.ones(2, dtype=torch.long))
