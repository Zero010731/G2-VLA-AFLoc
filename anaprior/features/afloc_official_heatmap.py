"""Exact heatmap construction used by the repository's AFLoc baseline."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F


def compute_official_afloc_heatmap(
    local_embeddings: torch.Tensor,
    report_embeddings: torch.Tensor,
    *,
    output_size: Tuple[int, int] = (224, 224),
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return AFLoc's local-image/global-report similarity map without normalization."""
    local = local_embeddings.detach()
    if local.ndim == 4:
        if local.shape[0] != 1:
            raise ValueError("official AFLoc heatmap inference requires batch size 1")
        local = local[0].permute(1, 2, 0).contiguous()
    if local.ndim != 3:
        raise ValueError("local_embeddings must have shape [H,W,D] or [1,D,H,W]")

    report = report_embeddings.detach()
    if report.ndim == 1:
        report = report.unsqueeze(0)
    if report.ndim != 2 or report.shape[0] != 1:
        raise ValueError("report_embeddings must have shape [1,D]")
    if local.shape[-1] != report.shape[-1]:
        raise ValueError("local and report embedding dimensions must match")

    height, width, feature_dim = local.shape
    similarity = local.reshape(-1, feature_dim) @ report.transpose(0, 1)
    similarity = similarity.reshape(height, width).cpu().numpy()
    smoothed = ndimage.gaussian_filter(similarity, sigma=(sigma, sigma), order=0)
    tensor = torch.from_numpy(np.asarray(smoothed)).reshape(1, 1, height, width)
    resized = F.interpolate(
        tensor,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0]
