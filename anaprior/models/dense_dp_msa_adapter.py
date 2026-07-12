"""Dense DP-MSA adapter inspired by DenseCLIP and CLIPSeg.

This module is a native AnaPrior component. It borrows mechanisms, not code:
dense phrase-feature matching, prompt-conditioned dense decoding, anatomy
attention, and region-ranking supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from anaprior.models.dp_msa_adapter import normalize_heatmaps


@dataclass(frozen=True)
class DenseDPMSAOutput:
    final_heatmap: torch.Tensor
    residual_map: torch.Tensor
    dense_match: torch.Tensor
    anatomy_attention: torch.Tensor
    branch_weights: torch.Tensor


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        padding = int(dilation)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=int(dilation)),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class DenseDPMultiScaleSpatialAdapter(nn.Module):
    """Patch-level disease/phrase-conditioned residual refinement adapter."""

    def __init__(
        self,
        num_subtypes: int,
        num_regions: int,
        num_disease_properties: int,
        spatial_channels: int,
        hidden_channels: int = 32,
        condition_dim: int = 64,
        lambda_weight: float = 0.05,
        residual_scale: float = 0.20,
    ) -> None:
        super().__init__()
        if num_subtypes <= 0:
            raise ValueError("num_subtypes must be positive")
        if num_regions <= 0:
            raise ValueError("num_regions must be positive")
        if num_disease_properties <= 0:
            raise ValueError("num_disease_properties must be positive")
        if spatial_channels <= 0:
            raise ValueError("spatial_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        self.num_regions = int(num_regions)
        self.num_disease_properties = int(num_disease_properties)
        self.lambda_weight = float(lambda_weight)
        self.residual_scale = float(residual_scale)

        self.spatial_proj = nn.Sequential(
            nn.Conv2d(int(spatial_channels), int(hidden_channels), kernel_size=1),
            nn.GroupNorm(1, int(hidden_channels)),
            nn.SiLU(inplace=True),
        )
        self.fallback_proj = nn.Sequential(
            nn.Conv2d(int(num_regions) + 2, int(hidden_channels), kernel_size=1),
            nn.GroupNorm(1, int(hidden_channels)),
            nn.SiLU(inplace=True),
        )
        self.anatomy_proj = nn.Sequential(
            nn.Conv2d(int(num_regions) + 2, int(hidden_channels), kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.anatomy_attention_head = nn.Conv2d(int(hidden_channels), 1, kernel_size=1)

        self.property_projection = nn.Linear(int(num_disease_properties), int(condition_dim))
        self.subtype_embedding = nn.Embedding(int(num_subtypes), int(condition_dim))
        self.conditioner = nn.Sequential(
            nn.Linear(int(condition_dim) * 2, int(hidden_channels)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_channels), int(hidden_channels)),
            nn.SiLU(inplace=True),
        )
        self.film = nn.Linear(int(hidden_channels), int(hidden_channels) * 2)
        self.match_projection = nn.Linear(int(hidden_channels), int(hidden_channels))
        self.branch_head = nn.Linear(int(hidden_channels), 4)

        self.local_branch = _ConvBlock(int(hidden_channels), dilation=1)
        self.dilated_branch = _ConvBlock(int(hidden_channels), dilation=2)
        self.focal_branch = nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.global_branch = nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            _ConvBlock(int(hidden_channels), dilation=1),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1),
        )

    def _validate(
        self,
        base_hmap: torch.Tensor,
        region_maps: torch.Tensor,
        region_scores: torch.Tensor,
        disease_properties: torch.Tensor,
        subtype_ids: torch.Tensor,
        spatial_features: Optional[torch.Tensor],
    ) -> None:
        if base_hmap.ndim != 4 or base_hmap.shape[1] != 1:
            raise ValueError("base_hmap must have shape [B,1,H,W]")
        if region_maps.ndim != 4:
            raise ValueError("region_maps must have shape [B,R,H,W]")
        if region_maps.shape[1] != self.num_regions:
            raise ValueError(f"region_maps num_regions must be {self.num_regions}")
        if region_scores.shape != region_maps.shape[:2]:
            raise ValueError("region_scores must have shape [B,R]")
        if tuple(region_maps.shape[2:]) != tuple(base_hmap.shape[2:]):
            raise ValueError("region_maps and base_hmap must have the same spatial size")
        if disease_properties.shape != (base_hmap.shape[0], self.num_disease_properties):
            raise ValueError("disease_properties must have shape [B,P]")
        if subtype_ids.shape[0] != base_hmap.shape[0]:
            raise ValueError("subtype_ids must match batch size")
        if spatial_features is not None and spatial_features.ndim != 4:
            raise ValueError("spatial_features must have shape [B,C,H,W]")

    def forward(
        self,
        base_hmap: torch.Tensor,
        region_maps: torch.Tensor,
        region_scores: torch.Tensor,
        disease_properties: torch.Tensor,
        subtype_ids: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> DenseDPMSAOutput:
        self._validate(base_hmap, region_maps, region_scores, disease_properties, subtype_ids, spatial_features)
        base = base_hmap.float()
        maps = region_maps.float()
        scores = region_scores.float().clamp_min(0.0)
        disease_properties = disease_properties.float()
        subtype_ids = subtype_ids.long()

        weighted_region = torch.sum(maps * scores[:, :, None, None], dim=1, keepdim=True)
        anatomy_input = torch.cat([maps, weighted_region, normalize_heatmaps(base)], dim=1)
        anatomy_features = self.anatomy_proj(anatomy_input)
        anatomy_attention = torch.sigmoid(self.anatomy_attention_head(anatomy_features))

        if spatial_features is None:
            spatial = self.fallback_proj(anatomy_input)
        else:
            spatial_raw = spatial_features.float()
            if tuple(spatial_raw.shape[2:]) != tuple(base.shape[2:]):
                spatial_raw = F.interpolate(spatial_raw, size=base.shape[2:], mode="bilinear", align_corners=False)
            spatial = self.spatial_proj(spatial_raw)

        condition = self.conditioner(
            torch.cat([self.property_projection(disease_properties), self.subtype_embedding(subtype_ids)], dim=-1)
        )
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        conditioned = spatial * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        conditioned = conditioned + anatomy_features

        match_vector = self.match_projection(condition)
        dense_match = torch.sum(conditioned * match_vector[:, :, None, None], dim=1, keepdim=True)
        dense_match = normalize_heatmaps(dense_match)

        gated = conditioned * (1.0 + anatomy_attention) * (1.0 + dense_match)
        local = self.local_branch(gated)
        dilated = self.dilated_branch(gated)
        focal = self.focal_branch(gated * (1.0 + dense_match))
        global_context = F.adaptive_avg_pool2d(gated, output_size=1).expand_as(gated)
        global_branch = self.global_branch(global_context)

        branch_weights = torch.softmax(self.branch_head(condition), dim=-1)
        branches = torch.stack([local, dilated, focal, global_branch], dim=1)
        fused = torch.sum(branches * branch_weights[:, :, None, None, None], dim=1)

        residual = torch.tanh(self.decoder(fused)) * self.residual_scale
        final = normalize_heatmaps(base + self.lambda_weight * residual)
        return DenseDPMSAOutput(
            final_heatmap=final,
            residual_map=residual,
            dense_match=dense_match,
            anatomy_attention=anatomy_attention,
            branch_weights=branch_weights,
        )


def dense_region_ranking_loss(
    dense_match: torch.Tensor,
    region_maps: torch.Tensor,
    region_scores: torch.Tensor,
    margin: float = 0.10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rank dense evidence in high-score regions above low-score regions."""

    if dense_match.ndim != 4 or dense_match.shape[1] != 1:
        raise ValueError("dense_match must have shape [B,1,H,W]")
    if region_maps.ndim != 4:
        raise ValueError("region_maps must have shape [B,R,H,W]")
    if region_scores.shape != region_maps.shape[:2]:
        raise ValueError("region_scores must have shape [B,R]")
    if tuple(region_maps.shape[2:]) != tuple(dense_match.shape[2:]):
        raise ValueError("region_maps and dense_match must have the same spatial size")

    maps = region_maps.float()
    match = dense_match.float()
    scores = region_scores.float()
    region_area = maps.flatten(start_dim=2).sum(dim=2).clamp_min(float(eps))
    region_mean = (maps * match).flatten(start_dim=2).sum(dim=2) / region_area

    pos_weights = scores.clamp_min(0.0)
    neg_weights = (scores <= 0.0).float()
    pos_valid = pos_weights.sum(dim=1) > float(eps)
    neg_valid = neg_weights.sum(dim=1) > float(eps)
    valid = pos_valid & neg_valid
    if not bool(valid.any()):
        return dense_match.sum() * 0.0

    pos_score = (region_mean * pos_weights).sum(dim=1) / pos_weights.sum(dim=1).clamp_min(float(eps))
    neg_score = (region_mean * neg_weights).sum(dim=1) / neg_weights.sum(dim=1).clamp_min(float(eps))
    losses = F.relu(float(margin) - pos_score + neg_score)
    return losses[valid].mean()
