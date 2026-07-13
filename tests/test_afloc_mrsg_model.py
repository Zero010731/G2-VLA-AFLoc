from __future__ import annotations

import inspect

import pytest
import torch

from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch, PhraseFeatureBatch
from tests.mrsg_test_utils import (
    fake_image_features,
    fake_phrase_features,
    test_config as make_test_config,
)


def model_class():
    from anaprior.models.afloc_mrsg import AFLocMRSG

    return AFLocMRSG


def test_afloc_mrsg_forward_returns_direct_nonconstant_heatmap() -> None:
    torch.manual_seed(31)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))

    output = model(fake_image_features(), fake_phrase_features())

    output.validate()
    assert output.final_heatmap.min().item() >= 0.0
    assert output.final_heatmap.max().item() <= 1.0
    assert output.final_heatmap.detach().std().item() > 0.0
    assert output.query_heatmaps.shape == (2, 4, 16, 16)
    assert output.query_route_weights.shape == (2, 4)
    assert output.query_reliability.shape == (2, 4)
    assert output.phrase_patch_logits.shape == (2, 7, 16, 16)


def test_afloc_mrsg_has_no_dcem_region_or_disease_id_inputs() -> None:
    AFLocMRSG = model_class()
    parameters = inspect.signature(AFLocMRSG.forward).parameters
    forbidden = {
        "base_hmap",
        "dcem_hmap",
        "region_maps",
        "region_scores",
        "disease_id",
        "boxes",
        "masks",
    }

    assert forbidden.isdisjoint(parameters)
    assert tuple(parameters) == ("self", "image_features", "phrase_features", "patch_mask")


def test_afloc_mrsg_routes_all_four_query_branches_into_loss_gradients() -> None:
    torch.manual_seed(37)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))

    output = model(fake_image_features(), fake_phrase_features())
    output.final_heatmap.mean().backward()

    for index, operator in enumerate(model.query_bank.operators):
        grad = operator.output_head.weight.grad
        assert grad is not None, f"query operator {index} did not receive gradients"
        assert grad.abs().sum().item() > 0.0

    assert model.phrase_router.route_head.weight.grad is not None
    assert model.grounder.word_projection.weight.grad is not None
    assert model.decoder.output_head.weight.grad is not None


def test_afloc_mrsg_keeps_afloc_inputs_frozen_at_parameter_boundary() -> None:
    torch.manual_seed(41)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))
    image_features = AFLocFeatureBatch(
        img_emb_l2=torch.rand(2, 32, 16, 16, requires_grad=True),
        img_emb_l=torch.rand(2, 64, 8, 8, requires_grad=True),
        img_emb_lf=torch.rand(2, 128, 4, 4, requires_grad=True),
        image_gray=torch.rand(2, 1, 224, 224, requires_grad=True),
    )
    phrase_features = PhraseFeatureBatch(
        word_embeddings=torch.rand(2, 7, 24, requires_grad=True),
        sentence_embedding=torch.rand(2, 24, requires_grad=True),
        disease_description_embedding=torch.rand(2, 24, requires_grad=True),
        attention_mask=torch.ones(2, 7, dtype=torch.bool),
    )

    output = model(image_features, phrase_features)
    output.final_heatmap.mean().backward()

    assert image_features.img_emb_l2.grad is None
    assert image_features.img_emb_l.grad is None
    assert image_features.img_emb_lf.grad is None
    assert image_features.image_gray.grad is None
    assert phrase_features.word_embeddings.grad is None
    assert phrase_features.sentence_embedding.grad is None
    assert phrase_features.disease_description_embedding.grad is None
    assert model.feature_pyramid.projections["l2"].weight.grad is not None


def test_afloc_mrsg_phrase_patch_logits_are_route_weighted_across_queries() -> None:
    torch.manual_seed(43)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))

    output = model(fake_image_features(), fake_phrase_features())
    internals = model.last_forward_debug
    expected = (
        internals["query_phrase_patch_logits"]
        * output.query_route_weights[:, :, None, None, None]
    ).sum(dim=1)

    assert torch.allclose(output.phrase_patch_logits, expected)


def test_afloc_mrsg_rejects_invalid_patch_masks_and_batch_mismatch() -> None:
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))

    with pytest.raises(ValueError, match="patch_mask"):
        model(
            fake_image_features(),
            fake_phrase_features(),
            patch_mask=torch.zeros(2, 1, 15, 16, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="same batch"):
        model(fake_image_features(batch=2), fake_phrase_features(batch=1))
