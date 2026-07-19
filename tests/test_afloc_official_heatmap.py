from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from anaprior.features.afloc_official_heatmap import compute_official_afloc_heatmap


def test_official_heatmap_matches_repository_localization_pipeline() -> None:
    local = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) / 50.0
    report = torch.tensor([[0.2, -0.1, 0.3, 0.4, -0.2]], dtype=torch.float32)

    patch_similarity = local.reshape(-1, 5) @ report.transpose(0, 1)
    expected_grid = ndimage.gaussian_filter(
        patch_similarity.reshape(3, 4).numpy(),
        sigma=(1.5, 1.5),
        order=0,
    )
    expected = F.interpolate(
        torch.from_numpy(expected_grid).reshape(1, 1, 3, 4),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    actual = compute_official_afloc_heatmap(local, report)

    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=1.0e-6)


def test_official_heatmap_accepts_encoder_channel_first_batch() -> None:
    local = torch.randn(1, 7, 3, 4)
    report = torch.randn(1, 7)

    actual = compute_official_afloc_heatmap(local, report, output_size=(32, 40))

    assert actual.shape == (32, 40)
    assert torch.isfinite(actual).all()
