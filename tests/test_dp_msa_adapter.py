import pytest
import torch

from anaprior.models.dp_msa_adapter import DPMultiScaleSpatialAdapter, normalize_heatmaps


def _inputs(batch: int = 2):
    base_hmap = torch.rand(batch, 1, 8, 8)
    region_maps = torch.rand(batch, 4, 8, 8)
    region_scores = torch.tensor([[0.9, 0.2, 0.0, 0.4], [0.1, 0.7, 0.3, 0.0]], dtype=torch.float32)[:batch]
    disease_ids = torch.tensor([0, 1], dtype=torch.long)[:batch]
    subtype_ids = torch.tensor([2, 3], dtype=torch.long)[:batch]
    return base_hmap, region_maps, region_scores, disease_ids, subtype_ids


def test_normalize_heatmaps_scales_each_sample_to_unit_range() -> None:
    values = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]], [[[5.0, 5.0], [5.0, 5.0]]]])

    out = normalize_heatmaps(values)

    assert torch.allclose(out[0].min(), torch.tensor(0.0))
    assert torch.allclose(out[0].max(), torch.tensor(1.0))
    assert torch.allclose(out[1], torch.zeros_like(out[1]))


def test_dp_msa_adapter_outputs_heatmap_residual_and_branch_weights() -> None:
    model = DPMultiScaleSpatialAdapter(
        num_diseases=3,
        num_subtypes=5,
        num_regions=4,
        hidden_channels=8,
        embedding_dim=6,
        lambda_weight=0.25,
    )
    inputs = _inputs()

    out = model(*inputs)

    assert out.final_heatmap.shape == inputs[0].shape
    assert out.residual_map.shape == inputs[0].shape
    assert out.dense_match.shape == inputs[0].shape
    assert out.branch_weights.shape == (2, 3)
    assert torch.allclose(out.branch_weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(out.final_heatmap).all()
    assert float(out.final_heatmap.detach().min()) >= 0.0
    assert float(out.final_heatmap.detach().max()) <= 1.0


def test_dp_msa_lambda_zero_returns_normalized_baseline() -> None:
    model = DPMultiScaleSpatialAdapter(
        num_diseases=3,
        num_subtypes=5,
        num_regions=4,
        hidden_channels=8,
        embedding_dim=6,
        lambda_weight=0.0,
    )
    base_hmap, region_maps, region_scores, disease_ids, subtype_ids = _inputs()

    out = model(base_hmap, region_maps, region_scores, disease_ids, subtype_ids)

    assert torch.allclose(out.final_heatmap, normalize_heatmaps(base_hmap), atol=1e-6)


def test_dp_msa_adapter_validates_region_count() -> None:
    model = DPMultiScaleSpatialAdapter(num_diseases=3, num_subtypes=5, num_regions=4)
    base_hmap, region_maps, region_scores, disease_ids, subtype_ids = _inputs()

    with pytest.raises(ValueError, match="num_regions"):
        model(base_hmap, region_maps[:, :3], region_scores[:, :3], disease_ids, subtype_ids)
