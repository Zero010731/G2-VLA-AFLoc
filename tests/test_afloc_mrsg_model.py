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


def _masked_prediction_loss(output) -> torch.Tensor:
    assert output.masked_predictions is not None
    assert output.source_targets is not None
    assert output.patch_mask is not None
    mask = output.patch_mask.to(dtype=output.final_heatmap.dtype)
    masked_positions = mask.sum().clamp_min(1.0)
    losses = []
    for name in ("l2", "l", "lf"):
        prediction = output.masked_predictions[name]
        target = output.source_targets[name]
        losses.append(
            ((prediction - target).square() * mask).sum()
            / (masked_positions * target.shape[1])
        )
    return torch.stack(losses).mean()


def _official_anchor(batch: int = 2) -> torch.Tensor:
    return torch.rand(batch, 1, 16, 16).clamp(1.0e-4, 1.0 - 1.0e-4)


def test_afloc_mrsg_forward_returns_anchor_preserving_nonconstant_heatmap() -> None:
    torch.manual_seed(31)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))

    anchor = _official_anchor()
    output = model(fake_image_features(), fake_phrase_features(), official_anchor=anchor)

    output.validate()
    assert output.final_heatmap.min().item() >= 0.0
    assert output.final_heatmap.max().item() <= 1.0
    assert output.final_heatmap.detach().std().item() > 0.0
    extreme_query_fraction = (
        (output.query_heatmaps.detach().lt(1.0e-3) | output.query_heatmaps.detach().gt(1.0 - 1.0e-3))
        .float()
        .mean()
    )
    assert extreme_query_fraction < 0.8
    assert output.query_heatmaps.shape == (2, 4, 16, 16)
    assert output.query_route_weights.shape == (2, 4)
    assert output.query_reliability.shape == (2, 4)
    assert output.phrase_patch_logits.shape == (2, 7, 16, 16)
    assert output.masked_predictions is not None
    assert output.source_targets is not None
    assert output.patch_mask is not None
    assert output.query_reconstructed_phrase is not None
    assert output.query_patch_gates is not None
    assert torch.allclose(output.anchor_heatmap, anchor)
    assert output.residual_logits is not None
    assert output.bounded_correction is not None


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
    assert tuple(parameters) == (
        "self",
        "image_features",
        "phrase_features",
        "official_anchor",
        "patch_mask",
    )


def test_afloc_mrsg_routes_all_four_query_branches_into_loss_gradients() -> None:
    torch.manual_seed(37)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))

    output = model(
        fake_image_features(),
        fake_phrase_features(),
        official_anchor=_official_anchor(),
    )
    (
        output.final_heatmap.mean()
        + output.query_heatmaps.mean()
        + output.phrase_patch_logits.mean()
    ).backward()

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

    output = model(image_features, phrase_features, official_anchor=_official_anchor())
    output.final_heatmap.mean().backward()

    assert image_features.img_emb_l2.grad is None
    assert image_features.img_emb_l.grad is None
    assert image_features.img_emb_lf.grad is None
    assert image_features.image_gray.grad is None
    assert phrase_features.word_embeddings.grad is None
    assert phrase_features.sentence_embedding.grad is None
    assert phrase_features.disease_description_embedding.grad is None
    assert model.feature_pyramid.projections["l2"].weight.grad is not None


def test_afloc_mrsg_combined_task10_loss_preserves_auxiliary_gradients() -> None:
    torch.manual_seed(47)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))
    patch_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    patch_mask[:, :, 3:10, 4:12] = True
    image_features = AFLocFeatureBatch(
        img_emb_l2=torch.rand(2, 32, 16, 16, requires_grad=True),
        img_emb_l=torch.rand(2, 64, 8, 8, requires_grad=True),
        img_emb_lf=torch.rand(2, 128, 4, 4, requires_grad=True),
        image_gray=torch.rand(2, 1, 224, 224, requires_grad=True),
    )

    output = model(
        image_features,
        fake_phrase_features(),
        official_anchor=_official_anchor(),
        patch_mask=patch_mask,
    )
    assert output.query_reconstructed_phrase is not None
    loss = (
        output.final_heatmap.mean()
        + _masked_prediction_loss(output)
        + output.query_reconstructed_phrase.square().mean()
        + (
            output.query_heatmaps
            * output.query_route_weights[:, :, None, None]
        ).square().mean()
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert image_features.img_emb_l2.grad is None
    assert image_features.img_emb_l.grad is None
    assert image_features.img_emb_lf.grad is None
    assert image_features.image_gray.grad is None
    assert model.feature_pyramid.mask_token.grad is not None
    assert model.feature_pyramid.mask_token.grad.abs().sum().item() > 0.0
    for predictor in model.feature_pyramid.predictors.values():
        grad = predictor[-1].weight.grad
        assert grad is not None
        assert grad.abs().sum().item() > 0.0
    assert model.grounder.output_norm.weight.grad is not None
    assert model.grounder.output_norm.weight.grad.abs().sum().item() > 0.0
    assert model.phrase_router.route_head.weight.grad is not None
    assert model.phrase_router.route_head.weight.grad.abs().sum().item() > 0.0
    assert model.decoder.output_head.weight.grad is not None
    assert model.decoder.output_head.weight.grad.abs().sum().item() > 0.0
    for index, operator in enumerate(model.query_bank.operators):
        grad = operator.output_head.weight.grad
        assert grad is not None, f"query operator {index} did not receive gradients"
        assert grad.abs().sum().item() > 0.0


def test_afloc_mrsg_phrase_patch_logits_are_route_weighted_across_queries() -> None:
    torch.manual_seed(43)
    AFLocMRSG = model_class()
    model = AFLocMRSG(make_test_config(), image_channels=(32, 64, 128))

    output = model(
        fake_image_features(),
        fake_phrase_features(),
        official_anchor=_official_anchor(),
    )
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
            official_anchor=_official_anchor(),
            patch_mask=torch.zeros(2, 1, 15, 16, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="same batch"):
        model(
            fake_image_features(batch=2),
            fake_phrase_features(batch=1),
            official_anchor=_official_anchor(),
        )
