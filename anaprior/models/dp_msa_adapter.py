"""DP-MSA-v0: disease-phrase multi-scale spatial adapter.

This module is intentionally small. It repairs an already-computed AFLoc
heatmap with anatomy-conditioned residual maps and does not modify AFLoc
encoders.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DPMSAOutput:
    final_heatmap: torch.Tensor
    residual_map: torch.Tensor
    branch_weights: torch.Tensor
    dense_match: torch.Tensor


def normalize_heatmaps(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Min-max normalize each `[C,H,W]` sample independently."""

    if values.ndim != 4:
        raise ValueError("values must have shape [B,C,H,W]")
    flat = values.flatten(start_dim=1)
    lo = flat.amin(dim=1).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1).view(-1, 1, 1, 1)
    span = hi - lo
    out = torch.where(span > eps, (values - lo) / span.clamp_min(eps), torch.zeros_like(values))
    return out.clamp(0.0, 1.0)


class DPMultiScaleSpatialAdapter(nn.Module):
    """Small residual adapter over baseline heatmap and anatomy region maps."""

    def __init__(
        self,
        num_diseases: int,
        num_subtypes: int,
        num_regions: int,
        hidden_channels: int = 16,
        embedding_dim: int = 32,
        lambda_weight: float = 0.1,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if num_diseases <= 0:
            raise ValueError("num_diseases must be positive")
        if num_subtypes <= 0:
            raise ValueError("num_subtypes must be positive")
        if num_regions <= 0:
            raise ValueError("num_regions must be positive")
        self.num_regions = int(num_regions)
        self.lambda_weight = float(lambda_weight)
        self.residual_scale = float(residual_scale)

        self.disease_embedding = nn.Embedding(int(num_diseases), int(embedding_dim))
        self.subtype_embedding = nn.Embedding(int(num_subtypes), int(embedding_dim))
        self.conditioner = nn.Sequential(
            nn.Linear(int(embedding_dim) * 2, int(hidden_channels) * 2),
            nn.ReLU(inplace=True),
        )
        self.film = nn.Linear(int(hidden_channels) * 2, int(hidden_channels) * 2)
        self.branch_head = nn.Linear(int(hidden_channels) * 2, 3)

        self.anatomy_proj = nn.Sequential(
            nn.Conv2d(int(num_regions) + 2, int(hidden_channels), kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.diffuse_branch = nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.focal_branch = nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.match_projection = nn.Linear(int(hidden_channels) * 2, int(hidden_channels))
        self.residual_head = nn.Conv2d(int(hidden_channels), 1, kernel_size=1)

    def _validate(
        self,
        base_hmap: torch.Tensor,
        region_maps: torch.Tensor,
        region_scores: torch.Tensor,
        disease_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
    ) -> None:
        if base_hmap.ndim != 4 or base_hmap.shape[1] != 1:
            raise ValueError("base_hmap must have shape [B,1,H,W]")
        if region_maps.ndim != 4:
            raise ValueError("region_maps must have shape [B,R,H,W]")
        if region_maps.shape[1] != self.num_regions:
            raise ValueError(f"region_maps num_regions must be {self.num_regions}")
        if region_scores.shape != region_maps.shape[:2]:
            raise ValueError("region_scores must have shape [B,R] matching region_maps")
        if region_maps.shape[0] != base_hmap.shape[0] or region_maps.shape[2:] != base_hmap.shape[2:]:
            raise ValueError("base_hmap and region_maps must have matching batch and spatial shape")
        if disease_ids.shape[0] != base_hmap.shape[0] or subtype_ids.shape[0] != base_hmap.shape[0]:
            raise ValueError("disease_ids and subtype_ids must match batch size")

    def forward(
        self,
        base_hmap: torch.Tensor,
        region_maps: torch.Tensor,
        region_scores: torch.Tensor,
        disease_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
    ) -> DPMSAOutput:
        self._validate(base_hmap, region_maps, region_scores, disease_ids, subtype_ids)
        base = base_hmap.float()
        maps = region_maps.float()
        scores = region_scores.float().clamp_min(0.0)
        disease_ids = disease_ids.long()
        subtype_ids = subtype_ids.long()

        weighted_region = torch.sum(maps * scores[:, :, None, None], dim=1, keepdim=True)
        anatomy_input = torch.cat([maps, weighted_region, normalize_heatmaps(base)], dim=1)
        anatomy_features = self.anatomy_proj(anatomy_input)

        condition = self.conditioner(
            torch.cat([self.disease_embedding(disease_ids), self.subtype_embedding(subtype_ids)], dim=-1)
        )
        gamma_beta = self.film(condition)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated = anatomy_features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        match_vector = self.match_projection(condition)
        dense_match = torch.sum(modulated * match_vector[:, :, None, None], dim=1, keepdim=True)
        dense_match = normalize_heatmaps(dense_match)

        branch_weights = torch.softmax(self.branch_head(condition), dim=-1)
        local = self.local_branch(modulated)
        diffuse = self.diffuse_branch(modulated)
        focal = self.focal_branch(modulated * (1.0 + dense_match))
        branches = torch.stack([local, diffuse, focal], dim=1)
        fused = torch.sum(branches * branch_weights[:, :, None, None, None], dim=1)

        residual = torch.tanh(self.residual_head(fused)) * self.residual_scale
        final = normalize_heatmaps(base + (self.lambda_weight * residual))
        return DPMSAOutput(
            final_heatmap=final,
            residual_map=residual,
            branch_weights=branch_weights,
            dense_match=dense_match,
        )
