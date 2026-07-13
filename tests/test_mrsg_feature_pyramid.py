from __future__ import annotations

import inspect

import pytest
import torch

from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch
from anaprior.models.afloc_mrsg.feature_pyramid import (
    LocalityAlignedFeaturePyramid,
    masked_patch_distillation_loss,
)
from tests.mrsg_test_utils import fake_image_features


def make_mismatched_features(batch: int = 2) -> AFLocFeatureBatch:
    return AFLocFeatureBatch(
        img_emb_l2=torch.rand(batch, 32, 17, 15),
        img_emb_l=torch.rand(batch, 64, 9, 7),
        img_emb_lf=torch.rand(batch, 128, 5, 4),
        image_gray=torch.rand(batch, 1, 241, 219),
    )


def make_patch_mask(batch: int = 2, height: int = 17, width: int = 15) -> torch.Tensor:
    patch_mask = torch.zeros(batch, 1, height, width, dtype=torch.bool)
    patch_mask[:, :, 4:10, 3:9] = True
    return patch_mask


def make_trainable_features(batch: int = 2) -> AFLocFeatureBatch:
    return AFLocFeatureBatch(
        img_emb_l2=torch.rand(batch, 32, 16, 16, requires_grad=True),
        img_emb_l=torch.rand(batch, 64, 8, 8, requires_grad=True),
        img_emb_lf=torch.rand(batch, 128, 4, 4, requires_grad=True),
        image_gray=torch.rand(batch, 1, 224, 224, requires_grad=True),
    )


def test_feature_pyramid_fuses_three_scales_at_mismatched_resolutions() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16, num_heads=4)
    output = module(make_mismatched_features(), patch_mask=make_patch_mask())

    assert output.fused.shape == (2, 16, 17, 15)
    assert output.edge_features.shape == (2, 3, 17, 15)
    assert set(output.source_targets) == {"l2", "l", "lf"}
    assert set(output.masked_prediction) == {"l2", "l", "lf"}
    assert output.source_targets["l"].shape == (2, 16, 17, 15)
    assert output.masked_prediction["lf"].shape == (2, 16, 17, 15)


def test_only_trainable_pyramid_parameters_receive_gradients() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16, num_heads=4)
    features = make_trainable_features()
    patch_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    patch_mask[:, :, 4:12, 4:12] = True

    output = module(features, patch_mask=patch_mask)
    loss = masked_patch_distillation_loss(output, patch_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert output.source_targets["l2"].requires_grad is False
    assert features.img_emb_l2.grad is None
    assert features.img_emb_l.grad is None
    assert features.img_emb_lf.grad is None
    assert features.image_gray.grad is None
    assert module.predictors["l2"][-1].weight.grad is not None
    assert module.predictors["l2"][-1].weight.grad.abs().sum().item() > 0.0


def test_fused_patches_keep_nonconstant_local_variation() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16, num_heads=4)
    output = module(fake_image_features(), patch_mask=torch.zeros(2, 1, 16, 16, dtype=torch.bool))

    assert output.fused.std(dim=(-2, -1)).mean().item() > 0.0
    assert output.edge_features.std(dim=(-2, -1)).mean().item() > 0.0


def test_masked_reconstruction_loss_is_zero_without_mask_and_positive_with_mask() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16, num_heads=4)
    empty_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    filled_mask = empty_mask.clone()
    filled_mask[:, :, 5:11, 5:11] = True

    empty_output = module(fake_image_features(), patch_mask=empty_mask)
    filled_output = module(fake_image_features(), patch_mask=filled_mask)

    assert masked_patch_distillation_loss(empty_output, empty_mask).item() == 0.0
    assert masked_patch_distillation_loss(filled_output, filled_mask).item() > 0.0


def test_eval_forward_is_deterministic() -> None:
    torch.manual_seed(7)
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16, num_heads=4)
    features = fake_image_features()
    patch_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    patch_mask[:, :, 2:6, 8:12] = True

    module.eval()
    with torch.no_grad():
        first = module(features, patch_mask=patch_mask)
        second = module(features, patch_mask=patch_mask)

    assert torch.allclose(first.fused, second.fused)
    assert torch.allclose(first.edge_features, second.edge_features)
    for key in ("l2", "l", "lf"):
        assert torch.allclose(first.source_targets[key], second.source_targets[key])
        assert torch.allclose(first.masked_prediction[key], second.masked_prediction[key])


def test_invalid_feature_shapes_and_channels_raise_value_errors() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16, num_heads=4)

    with pytest.raises(ValueError, match="img_emb_l2"):
        module(
            AFLocFeatureBatch(
                img_emb_l2=torch.rand(2, 31, 16, 16),
                img_emb_l=torch.rand(2, 64, 8, 8),
                img_emb_lf=torch.rand(2, 128, 4, 4),
                image_gray=torch.rand(2, 1, 224, 224),
            ),
            patch_mask=torch.zeros(2, 1, 16, 16, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="patch_mask"):
        module(
            fake_image_features(),
            patch_mask=torch.zeros(2, 1, 15, 16, dtype=torch.bool),
        )


def test_feature_pyramid_does_not_accept_spatial_supervision_inputs() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16, num_heads=4)
    patch_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)

    assert tuple(inspect.signature(module.forward).parameters) == ("image_features", "patch_mask")
    with pytest.raises(TypeError, match="unexpected keyword argument 'boxes'"):
        module(fake_image_features(), patch_mask=patch_mask, boxes=torch.zeros(2, 4))
