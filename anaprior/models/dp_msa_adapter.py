"""DP-MSA-v0: disease-phrase multi-scale spatial adapter.

This module is intentionally small. It repairs an already-computed AFLoc
heatmap with anatomy-conditioned residual maps and does not modify AFLoc
encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        num_disease_properties: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_diseases <= 0:
            raise ValueError("num_diseases must be positive")
        if num_subtypes <= 0:
            raise ValueError("num_subtypes must be positive")
        if num_regions <= 0:
            raise ValueError("num_regions must be positive")
        if num_disease_properties is not None and num_disease_properties <= 0:
            raise ValueError("num_disease_properties must be positive when provided")
        self.num_regions = int(num_regions)
        self.num_disease_properties = None if num_disease_properties is None else int(num_disease_properties)
        self.lambda_weight = float(lambda_weight)
        self.residual_scale = float(residual_scale)

        self.disease_embedding = nn.Embedding(int(num_diseases), int(embedding_dim))
        self.subtype_embedding = nn.Embedding(int(num_subtypes), int(embedding_dim))
        self.property_projection = (
            nn.Linear(self.num_disease_properties, int(embedding_dim))
            if self.num_disease_properties is not None
            else None
        )
        self.conditioner = nn.Sequential(
            nn.Linear(int(embedding_dim) * 2, int(hidden_channels) * 2),
            nn.ReLU(inplace=True),
        )
        self.film = nn.Linear(int(hidden_channels) * 2, int(hidden_channels) * 2)
        self.branch_head = nn.Linear(int(hidden_channels) * 2, 3)
        self.property_branch_head = nn.Linear(int(hidden_channels) * 2, 5)

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
        self.pleural_branch = nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.structural_branch = nn.Sequential(
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
        disease_properties: Optional[torch.Tensor] = None,
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
        if disease_properties is not None:
            if self.num_disease_properties is None or self.property_projection is None:
                raise ValueError("num_disease_properties must be configured to use disease_properties")
            expected = (base_hmap.shape[0], self.num_disease_properties)
            if tuple(disease_properties.shape) != expected:
                raise ValueError(f"disease_properties must have shape {expected}")

    def _condition(
        self,
        disease_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
        disease_properties: Optional[torch.Tensor],
    ) -> torch.Tensor:
        subtype = self.subtype_embedding(subtype_ids)
        if disease_properties is None:
            disease = self.disease_embedding(disease_ids)
        else:
            if self.property_projection is None:
                raise ValueError("property_projection is not configured")
            disease = self.property_projection(disease_properties.float())
        return self.conditioner(torch.cat([disease, subtype], dim=-1))

    def forward(
        self,
        base_hmap: torch.Tensor,
        region_maps: torch.Tensor,
        region_scores: torch.Tensor,
        disease_ids: torch.Tensor,
        subtype_ids: torch.Tensor,
        *,
        disease_properties: Optional[torch.Tensor] = None,
    ) -> DPMSAOutput:
        self._validate(base_hmap, region_maps, region_scores, disease_ids, subtype_ids, disease_properties)
        base = base_hmap.float()
        maps = region_maps.float()
        scores = region_scores.float().clamp_min(0.0)
        disease_ids = disease_ids.long()
        subtype_ids = subtype_ids.long()

        weighted_region = torch.sum(maps * scores[:, :, None, None], dim=1, keepdim=True)
        anatomy_input = torch.cat([maps, weighted_region, normalize_heatmaps(base)], dim=1)
        anatomy_features = self.anatomy_proj(anatomy_input)

        condition = self._condition(disease_ids, subtype_ids, disease_properties)
        gamma_beta = self.film(condition)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated = anatomy_features * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        match_vector = self.match_projection(condition)
        dense_match = torch.sum(modulated * match_vector[:, :, None, None], dim=1, keepdim=True)
        dense_match = normalize_heatmaps(dense_match)

        local = self.local_branch(modulated)
        diffuse = self.diffuse_branch(modulated)
        focal = self.focal_branch(modulated * (1.0 + dense_match))
        if disease_properties is None:
            branch_weights = torch.softmax(self.branch_head(condition), dim=-1)
            branches = torch.stack([local, diffuse, focal], dim=1)
        else:
            branch_weights = torch.softmax(self.property_branch_head(condition), dim=-1)
            pleural = self.pleural_branch(modulated * (1.0 + dense_match))
            structural = self.structural_branch(modulated)
            branches = torch.stack([local, diffuse, focal, pleural, structural], dim=1)
        fused = torch.sum(branches * branch_weights[:, :, None, None, None], dim=1)

        residual = torch.tanh(self.residual_head(fused)) * self.residual_scale
        final = normalize_heatmaps(base + (self.lambda_weight * residual))
        return DPMSAOutput(
            final_heatmap=final,
            residual_map=residual,
            branch_weights=branch_weights,
            dense_match=dense_match,
        )
