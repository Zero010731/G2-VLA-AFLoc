import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from anaprior.models.region_abnormality_predictor import RegionAbnormalityPredictor


def test_predictor_returns_one_logit_per_region() -> None:
    model = RegionAbnormalityPredictor(
        feature_dim=4,
        num_findings=2,
        finding_embedding_dim=3,
        hidden_dim=5,
    )
    region_features = torch.randn(2, 6, 4)
    finding_ids = torch.tensor([0, 1])

    logits = model(region_features, finding_ids)

    assert logits.shape == (2, 6)


def test_predictor_supports_region_finding_pairs() -> None:
    model = RegionAbnormalityPredictor(
        feature_dim=4,
        num_findings=2,
        finding_embedding_dim=3,
        hidden_dim=5,
    )
    region_features = torch.randn(7, 4)
    finding_ids = torch.tensor([0, 1, 0, 1, 0, 1, 0])

    logits = model.score_pairs(region_features, finding_ids)

    assert logits.shape == (7,)


def test_loss_masks_invalid_regions() -> None:
    model = RegionAbnormalityPredictor(
        feature_dim=2,
        num_findings=1,
        finding_embedding_dim=2,
        hidden_dim=4,
    )
    logits = torch.tensor([[0.0, 0.0, 10.0]])
    labels = torch.tensor([[0.0, 1.0, 1.0]])
    valid_mask = torch.tensor([[True, True, False]])

    loss = model.loss(logits, labels, valid_mask=valid_mask)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.allclose(loss, expected)

