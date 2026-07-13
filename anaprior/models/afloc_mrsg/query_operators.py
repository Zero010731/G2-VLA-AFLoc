from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class QueryOperatorOutput:
    name: str
    features: torch.Tensor
    heatmap_logits: torch.Tensor
    reliability: torch.Tensor
    auxiliary: dict[str, torch.Tensor]


def straight_through_topk_gate(logits: torch.Tensor, fraction: float) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits must have shape [B,1,H,W]")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the interval (0, 1]")

    soft = torch.sigmoid(logits)
    height, width = logits.shape[-2:]
    k = max(1, round(height * width * fraction))
    flat_logits = logits.flatten(2)
    indices = flat_logits.topk(k, dim=-1).indices
    hard = torch.zeros_like(flat_logits).scatter_(2, indices, 1.0).view_as(logits)
    return hard.detach() - soft.detach() + soft


class _BaseQueryOperator(nn.Module):
    name: str

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.output_head = nn.Conv2d(feature_dim, 1, kernel_size=1)
        self.reliability_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 1),
        )

    def _reliability(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.reliability_head(features))


class FocalSparseQueryOperator(_BaseQueryOperator):
    name = "focal"

    def __init__(self, feature_dim: int, focal_slots: int, focal_fraction: float) -> None:
        super().__init__(feature_dim)
        if focal_slots <= 0:
            raise ValueError("focal_slots must be positive")
        self.focal_slots = focal_slots
        self.focal_fraction = focal_fraction
        self.slot_projection = nn.Linear(feature_dim, focal_slots * feature_dim)
        self.feature_projection = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(
        self,
        pyramid: torch.Tensor,
        edge_features: torch.Tensor,
        phrase_vector: torch.Tensor,
    ) -> QueryOperatorOutput:
        del edge_features
        batch_size, channels, height, width = pyramid.shape
        slots = self.slot_projection(phrase_vector).view(batch_size, self.focal_slots, channels)
        norm_pyramid = F.normalize(pyramid, dim=1)
        norm_slots = F.normalize(slots, dim=-1)
        slot_logits = torch.einsum("bchw,bkc->bkhw", norm_pyramid, norm_slots).unsqueeze(2)
        slot_logits = slot_logits * channels**0.5
        slot_gates = torch.cat(
            [
                straight_through_topk_gate(slot_logits[:, slot], self.focal_fraction)
                for slot in range(self.focal_slots)
            ],
            dim=1,
        ).view(batch_size, self.focal_slots, 1, height, width)
        slot_probability = torch.sigmoid(slot_logits) * slot_gates
        support = 1.0 - torch.prod(1.0 - slot_probability.clamp(max=1.0 - 1e-6), dim=1)
        features = self.feature_projection(pyramid) * support
        head_logits = self.output_head(features)
        heatmap_logits = torch.logit(support.clamp(1e-5, 1.0 - 1e-5)) + 0.1 * head_logits
        return QueryOperatorOutput(
            name=self.name,
            features=features,
            heatmap_logits=heatmap_logits,
            reliability=self._reliability(features),
            auxiliary={
                "slot_logits": slot_logits,
                "slot_gates": slot_gates,
                "support": support,
            },
        )


class DiffuseContextQueryOperator(_BaseQueryOperator):
    name = "diffuse"

    def __init__(self, feature_dim: int, num_heads: int, window_size: int = 4) -> None:
        super().__init__(feature_dim)
        self.window_size = window_size
        self.phrase_projection = nn.Linear(feature_dim, feature_dim)
        self.window_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.low_pass = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=5, padding=2, groups=feature_dim),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.GELU(),
        )
        self.fuse = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1)

    def forward(
        self,
        pyramid: torch.Tensor,
        edge_features: torch.Tensor,
        phrase_vector: torch.Tensor,
    ) -> QueryOperatorOutput:
        del edge_features
        _, _, height, width = pyramid.shape
        pooled = F.avg_pool2d(
            pyramid,
            kernel_size=self.window_size,
            stride=self.window_size,
            ceil_mode=True,
        )
        tokens = pooled.flatten(2).transpose(1, 2)
        global_token = pyramid.mean(dim=(-2, -1), keepdim=False).unsqueeze(1)
        phrase_token = self.phrase_projection(phrase_vector).unsqueeze(1)
        context_tokens = torch.cat((global_token, phrase_token), dim=1)
        attended, attention_weights = self.window_attention(
            query=tokens + phrase_token,
            key=context_tokens,
            value=context_tokens,
            need_weights=True,
        )
        context = attended.transpose(1, 2).reshape_as(pooled)
        context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
        low_pass = self.low_pass(pyramid)
        features = self.fuse(torch.cat((low_pass, context), dim=1))
        heatmap_logits = self.output_head(features)
        support = torch.sigmoid(F.avg_pool2d(heatmap_logits, kernel_size=7, stride=1, padding=3))
        return QueryOperatorOutput(
            name=self.name,
            features=features,
            heatmap_logits=heatmap_logits,
            reliability=self._reliability(features),
            auxiliary={
                "window_tokens": tokens,
                "global_token": global_token.squeeze(1),
                "attention_weights": attention_weights,
                "support": support,
            },
        )


class BoundaryPropagationQueryOperator(_BaseQueryOperator):
    name = "boundary"

    def __init__(self, feature_dim: int) -> None:
        super().__init__(feature_dim)
        self.edge_projection = nn.Conv2d(3, feature_dim, kernel_size=1)
        self.semantic_projection = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.message_passing = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.phrase_gate = nn.Linear(feature_dim, feature_dim)
        self.edge_head = nn.Conv2d(feature_dim, 1, kernel_size=1)

    def forward(
        self,
        pyramid: torch.Tensor,
        edge_features: torch.Tensor,
        phrase_vector: torch.Tensor,
    ) -> QueryOperatorOutput:
        semantic = self.semantic_projection(pyramid)
        edge_context = self.edge_projection(edge_features)
        local_messages = self.message_passing(torch.cat((semantic, edge_context), dim=1))
        phrase_gate = torch.sigmoid(self.phrase_gate(phrase_vector)).view(-1, self.feature_dim, 1, 1)
        features = local_messages * phrase_gate + edge_context
        heatmap_logits = self.output_head(features) + self.edge_head(edge_context)
        probabilities = torch.sigmoid(heatmap_logits)
        horizontal_pairs = probabilities[:, :, :, 1:] * probabilities[:, :, :, :-1]
        vertical_pairs = probabilities[:, :, 1:, :] * probabilities[:, :, :-1, :]
        return QueryOperatorOutput(
            name=self.name,
            features=features,
            heatmap_logits=heatmap_logits,
            reliability=self._reliability(features),
            auxiliary={
                "horizontal_pairs": horizontal_pairs,
                "vertical_pairs": vertical_pairs,
                "edge_response": self.edge_head(edge_context),
            },
        )


class StructuralRelationQueryOperator(_BaseQueryOperator):
    name = "structural"

    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__(feature_dim)
        self.relation_projection = nn.Linear(feature_dim * 3, feature_dim)
        self.phrase_projection = nn.Linear(feature_dim, feature_dim)
        self.relation_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.context_projection = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(
        self,
        pyramid: torch.Tensor,
        edge_features: torch.Tensor,
        phrase_vector: torch.Tensor,
    ) -> QueryOperatorOutput:
        del edge_features
        batch_size, channels, height, width = pyramid.shape
        left_width = width // 2
        right_width = width - left_width
        pair_width = min(left_width, right_width)
        if pair_width == 0:
            column_context = pyramid.mean(dim=-1, keepdim=True)
            global_context = pyramid.mean(dim=(-2, -1), keepdim=True).expand(-1, -1, height, 1)
            relation_input = torch.cat(
                (column_context, global_context, (column_context - global_context).abs()),
                dim=1,
            )
            relation_tokens = relation_input.permute(0, 2, 3, 1)
            flat_relations = relation_tokens.reshape(batch_size, height, channels * 3)
        else:
            left = pyramid[:, :, :, :pair_width]
            right = torch.flip(pyramid[:, :, :, width - pair_width :], dims=(-1,))
            relation_input = torch.cat((left, right, (left - right).abs()), dim=1)
            relation_tokens = relation_input.permute(0, 2, 3, 1)
            flat_relations = relation_tokens.reshape(batch_size, height * pair_width, channels * 3)
        relation_embedding = self.relation_projection(flat_relations)
        phrase_token = self.phrase_projection(phrase_vector).unsqueeze(1)
        attended, attention_weights = self.relation_attention(
            query=relation_embedding + phrase_token,
            key=relation_embedding,
            value=relation_embedding,
            need_weights=True,
        )
        if pair_width == 0:
            relation_map = attended.transpose(1, 2).reshape(batch_size, channels, height, 1)
            broadcast = relation_map.expand(-1, -1, -1, width)
        else:
            relation_map = attended.transpose(1, 2).reshape(batch_size, channels, height, pair_width)
            broadcast = torch.zeros_like(pyramid)
            broadcast[:, :, :, :pair_width] = relation_map
            broadcast[:, :, :, width - pair_width :] = torch.flip(relation_map, dims=(-1,))
            if width > pair_width * 2:
                middle_start = pair_width
                middle_end = width - pair_width
                middle = relation_map.mean(dim=-1, keepdim=True).expand(-1, -1, -1, middle_end - middle_start)
                broadcast[:, :, :, middle_start:middle_end] = middle
        features = self.context_projection(pyramid + broadcast)
        heatmap_logits = self.output_head(features)
        return QueryOperatorOutput(
            name=self.name,
            features=features,
            heatmap_logits=heatmap_logits,
            reliability=self._reliability(features),
            auxiliary={
                "relation_tokens": relation_tokens,
                "attention_weights": attention_weights,
            },
        )


class MorphologyQueryBank(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        focal_slots: int = 4,
        focal_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_heads <= 0 or feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        self.feature_dim = feature_dim
        self.operators = nn.ModuleList(
            [
                FocalSparseQueryOperator(feature_dim, focal_slots, focal_fraction),
                DiffuseContextQueryOperator(feature_dim, num_heads),
                BoundaryPropagationQueryOperator(feature_dim),
                StructuralRelationQueryOperator(feature_dim, num_heads),
            ]
        )

    def forward(
        self,
        pyramid: torch.Tensor,
        edge_features: torch.Tensor,
        phrase_vector: torch.Tensor,
    ) -> tuple[
        QueryOperatorOutput,
        QueryOperatorOutput,
        QueryOperatorOutput,
        QueryOperatorOutput,
    ]:
        self._validate_inputs(pyramid, edge_features, phrase_vector)
        return tuple(
            operator(pyramid, edge_features, phrase_vector)
            for operator in self.operators
        )

    def _validate_inputs(
        self,
        pyramid: torch.Tensor,
        edge_features: torch.Tensor,
        phrase_vector: torch.Tensor,
    ) -> None:
        if pyramid.ndim != 4:
            raise ValueError("pyramid must have shape [B,C,H,W]")
        if pyramid.shape[1] != self.feature_dim:
            raise ValueError(
                f"pyramid expected {self.feature_dim} channels but received {pyramid.shape[1]}"
            )
        if edge_features.ndim != 4:
            raise ValueError("edge_features must have shape [B,3,H,W]")
        if edge_features.shape[1] != 3:
            raise ValueError("edge_features must contain three edge channels")
        if phrase_vector.ndim != 2:
            raise ValueError("phrase_vector must have shape [B,C]")
        if phrase_vector.shape[1] != self.feature_dim:
            raise ValueError(
                f"phrase_vector expected {self.feature_dim} channels but received {phrase_vector.shape[1]}"
            )
        if (
            edge_features.shape[0] != pyramid.shape[0]
            or phrase_vector.shape[0] != pyramid.shape[0]
        ):
            raise ValueError("pyramid, edge_features, and phrase_vector must share batch size")
        if edge_features.shape[-2:] != pyramid.shape[-2:]:
            raise ValueError("edge_features must match pyramid spatial size")
