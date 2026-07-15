from __future__ import annotations

from collections.abc import Sequence

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


class StandaloneDenseDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or feature_dim
        input_channels = feature_dim * 6 + 9
        self.input_projection = nn.Conv2d(input_channels, self.hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            _ResidualConvBlock(self.hidden_dim),
            _ResidualConvBlock(self.hidden_dim),
        )
        self.output_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        pyramid: torch.Tensor,
        query_outputs: Sequence[QueryOperatorOutput],
        route_weights: torch.Tensor,
        query_patch_gates: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(pyramid, query_outputs, route_weights, query_patch_gates)
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
        raw_refinement = self.output_head(self.blocks(self.input_projection(decoder_input)))
        centered_refinement = raw_refinement - raw_refinement.mean(dim=(-2, -1), keepdim=True)
        bounded_refinement = 0.5 * torch.tanh(centered_refinement)
        routed_query_map = weighted_query_maps.sum(dim=1, keepdim=True)
        routed_query_logits = torch.logit(routed_query_map.clamp(1.0e-4, 1.0 - 1.0e-4))
        return torch.sigmoid(routed_query_logits + bounded_refinement)

    def _validate(
        self,
        pyramid: torch.Tensor,
        query_outputs: Sequence[QueryOperatorOutput],
        route_weights: torch.Tensor,
        query_patch_gates: torch.Tensor,
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
