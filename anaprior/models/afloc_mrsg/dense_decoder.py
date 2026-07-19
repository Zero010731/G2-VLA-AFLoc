from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from anaprior.models.afloc_mrsg.query_operators import QueryOperatorOutput


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.layers(features))


@dataclass(frozen=True)
class BoundedResidualOutput:
    final_heatmap: torch.Tensor
    residual_logits: torch.Tensor
    bounded_correction: torch.Tensor
    correction_bound: torch.Tensor


class AnchorBoundedResidualDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int | None = None,
        residual_logit_bound: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0.0 < residual_logit_bound <= 2.0:
            raise ValueError("residual_logit_bound must be in (0,2]")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or feature_dim
        self.residual_logit_bound = float(residual_logit_bound)
        input_channels = feature_dim * 6 + 9
        self.input_projection = nn.Conv2d(input_channels, self.hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            _ResidualConvBlock(self.hidden_dim),
            _ResidualConvBlock(self.hidden_dim),
        )
        self.output_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        pyramid: torch.Tensor,
        query_outputs: Sequence[QueryOperatorOutput],
        route_weights: torch.Tensor,
        query_patch_gates: torch.Tensor,
        official_anchor: torch.Tensor,
    ) -> BoundedResidualOutput:
        self._validate(
            pyramid,
            query_outputs,
            route_weights,
            query_patch_gates,
            official_anchor,
        )
        query_features = torch.stack([output.features for output in query_outputs], dim=1)
        query_heatmaps = torch.cat(
            [torch.sigmoid(output.heatmap_logits) for output in query_outputs],
            dim=1,
        )
        query_reliability = torch.cat([output.reliability for output in query_outputs], dim=1)

        weighted_query_maps = query_heatmaps * route_weights[:, :, None, None]
        reliability_maps = query_reliability[:, :, None, None].expand_as(weighted_query_maps)
        global_sparse_gate = query_patch_gates.mean(dim=1, keepdim=True)
        fused_query_features = (query_features * route_weights[:, :, None, None, None]).sum(dim=1)

        decoder_input = torch.cat(
            (
                pyramid,
                fused_query_features,
                query_features.flatten(1, 2),
                weighted_query_maps,
                reliability_maps,
                global_sparse_gate,
            ),
            dim=1,
        )
        residual_logits = self.output_head(self.blocks(self.input_projection(decoder_input)))
        correction_bound = torch.full_like(residual_logits, self.residual_logit_bound)
        bounded_correction = correction_bound * torch.tanh(residual_logits)
        anchor_logits = torch.logit(official_anchor.detach())
        final_heatmap = torch.sigmoid(anchor_logits + bounded_correction)
        return BoundedResidualOutput(
            final_heatmap=final_heatmap,
            residual_logits=residual_logits,
            bounded_correction=bounded_correction,
            correction_bound=correction_bound,
        )

    def _validate(
        self,
        pyramid: torch.Tensor,
        query_outputs: Sequence[QueryOperatorOutput],
        route_weights: torch.Tensor,
        query_patch_gates: torch.Tensor,
        official_anchor: torch.Tensor,
    ) -> None:
        if pyramid.ndim != 4:
            raise ValueError("pyramid must have shape [B,C,H,W]")
        batch_size, channels, height, width = pyramid.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"pyramid expected {self.feature_dim} channels but received {channels}"
            )
        if len(query_outputs) != 4:
            raise ValueError("query_outputs must contain exactly four query outputs")
        for index, output in enumerate(query_outputs):
            if output.features.shape != (batch_size, self.feature_dim, height, width):
                raise ValueError(
                    f"query_outputs[{index}].features must have shape [B,C,H,W]"
                )
            if output.heatmap_logits.shape != (batch_size, 1, height, width):
                raise ValueError(
                    f"query_outputs[{index}].heatmap_logits must have shape [B,1,H,W]"
                )
            if output.reliability.shape != (batch_size, 1):
                raise ValueError(f"query_outputs[{index}].reliability must have shape [B,1]")
        if route_weights.shape != (batch_size, 4):
            raise ValueError("route_weights must have shape [B,4]")
        if query_patch_gates.shape != (batch_size, 4, height, width):
            raise ValueError("query_patch_gates must have shape [B,4,H,W]")
        if official_anchor.shape != (batch_size, 1, height, width):
            raise ValueError("official_anchor must have shape [B,1,H,W]")
        if official_anchor.requires_grad:
            raise ValueError("official_anchor must be detached")
        if not torch.isfinite(official_anchor).all():
            raise ValueError("official_anchor must contain finite values")
        if bool((official_anchor <= 0.0).any()) or bool((official_anchor >= 1.0).any()):
            raise ValueError("official_anchor values must be strictly inside (0,1)")
