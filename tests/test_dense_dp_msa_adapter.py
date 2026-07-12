import torch

from anaprior.eval.disease_properties import DISEASE_PROPERTY_NAMES, disease_property_matrix
from anaprior.eval.phrase_subtype import PHRASE_SUBTYPES
from anaprior.models.dense_dp_msa_adapter import (
    DenseDPMultiScaleSpatialAdapter,
    dense_region_ranking_loss,
)
from anaprior.models.dp_msa_adapter import normalize_heatmaps


def _inputs(batch: int = 2):
    torch.manual_seed(11)
    base_hmap = torch.rand(batch, 1, 8, 8)
    region_maps = torch.rand(batch, 4, 8, 8)
    region_scores = torch.tensor([[0.9, 0.2, 0.0, 0.4], [0.1, 0.7, 0.3, 0.0]], dtype=torch.float32)[:batch]
    disease_properties = disease_property_matrix(["Pneumothorax", "Pneumonia"])[:batch]
    subtype_ids = torch.tensor([2, 5], dtype=torch.long)[:batch]
    spatial_features = torch.rand(batch, 10, 4, 4)
    return base_hmap, region_maps, region_scores, disease_properties, subtype_ids, spatial_features


def test_dense_dp_msa_adapter_outputs_dense_clipseg_style_refinement() -> None:
    model = DenseDPMultiScaleSpatialAdapter(
        num_subtypes=len(PHRASE_SUBTYPES),
        num_regions=4,
        num_disease_properties=len(DISEASE_PROPERTY_NAMES),
        spatial_channels=10,
        hidden_channels=8,
        condition_dim=6,
        lambda_weight=0.1,
    )
    base_hmap, region_maps, region_scores, disease_properties, subtype_ids, spatial_features = _inputs()

    out = model(
        base_hmap=base_hmap,
        region_maps=region_maps,
        region_scores=region_scores,
        disease_properties=disease_properties,
        subtype_ids=subtype_ids,
        spatial_features=spatial_features,
    )

    assert out.final_heatmap.shape == base_hmap.shape
    assert out.residual_map.shape == base_hmap.shape
    assert out.dense_match.shape == base_hmap.shape
    assert out.anatomy_attention.shape == base_hmap.shape
    assert out.branch_weights.shape == (2, 4)
    assert torch.allclose(out.branch_weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(out.final_heatmap).all()


def test_dense_dp_msa_lambda_zero_returns_normalized_baseline() -> None:
    model = DenseDPMultiScaleSpatialAdapter(
        num_subtypes=len(PHRASE_SUBTYPES),
        num_regions=4,
        num_disease_properties=len(DISEASE_PROPERTY_NAMES),
        spatial_channels=10,
        lambda_weight=0.0,
    )
    base_hmap, region_maps, region_scores, disease_properties, subtype_ids, spatial_features = _inputs()

    out = model(base_hmap, region_maps, region_scores, disease_properties, subtype_ids, spatial_features)

    assert torch.allclose(out.final_heatmap, normalize_heatmaps(base_hmap), atol=1e-6)


def test_dense_region_ranking_loss_penalizes_positive_below_negative_regions() -> None:
    dense_match = torch.tensor([[[[0.1, 0.1], [0.9, 0.9]]]], dtype=torch.float32)
    region_maps = torch.tensor(
        [[
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]],
        dtype=torch.float32,
    )
    region_scores = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    loss = dense_region_ranking_loss(dense_match, region_maps, region_scores, margin=0.2)

    assert loss.ndim == 0
    assert float(loss.item()) > 0.0
