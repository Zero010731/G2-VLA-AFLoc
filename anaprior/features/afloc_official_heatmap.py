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


def compute_official_afloc_anchor_batch(
    local_embeddings: torch.Tensor,
    report_embeddings: torch.Tensor,
    *,
    output_size: Tuple[int, int],
    sigma: float = 1.5,
    epsilon: float = 1.0e-4,
) -> torch.Tensor:
    """Build detached, logit-safe official AFLoc anchors for a phrase batch."""
    if local_embeddings.ndim != 4:
        raise ValueError("local_embeddings must have shape [B,D,H,W]")
    if report_embeddings.ndim != 2:
        raise ValueError("report_embeddings must have shape [B,D]")
    if local_embeddings.shape[:2] != report_embeddings.shape:
        raise ValueError("local and report embedding batch/channel dimensions must match")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0,0.5)")

    device = local_embeddings.device
    dtype = local_embeddings.dtype
    raw_maps = [
        compute_official_afloc_heatmap(
            local_embeddings[index : index + 1],
            report_embeddings[index : index + 1],
            output_size=output_size,
            sigma=sigma,
        )
        for index in range(local_embeddings.shape[0])
    ]
    raw = torch.stack(raw_maps, dim=0).unsqueeze(1).to(device=device, dtype=dtype)
    low = raw.amin(dim=(-2, -1), keepdim=True)
    high = raw.amax(dim=(-2, -1), keepdim=True)
    normalized = (raw - low) / (high - low).clamp_min(1.0e-6)
    return normalized.clamp(epsilon, 1.0 - epsilon).detach()
