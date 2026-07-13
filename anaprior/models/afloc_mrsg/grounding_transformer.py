from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.query_operators import QueryOperatorOutput


@dataclass(frozen=True)
class SparseGroundingOutput:
    query_phrase_patch_logits: torch.Tensor
    query_gated_features: torch.Tensor
    query_reconstructed_phrase: torch.Tensor
    query_patch_gates: torch.Tensor


class MultiQuerySparsePhrasePatchGrounder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        topk_fraction: float = 0.125,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_heads <= 0 or feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0, 1]")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.topk_fraction = topk_fraction
        self.word_projection = nn.Linear(feature_dim, feature_dim)
        self.query_visual_projections = nn.ModuleList(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1) for _ in range(4)
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        query_features: torch.Tensor | Sequence[QueryOperatorOutput],
        word_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SparseGroundingOutput:
        query_tensor = self._query_tensor(query_features)
        query_tensor, word_features, attention_mask = self._validate(
            query_tensor,
            word_features,
            attention_mask,
        )

        projected_words = self.word_projection(word_features)
        projected_words = projected_words * attention_mask.unsqueeze(-1).to(projected_words.dtype)
        projected_queries = torch.stack(
            [
                projection(query_tensor[:, index])
                for index, projection in enumerate(self.query_visual_projections)
            ],
            dim=1,
        )

        norm_queries = F.normalize(projected_queries, dim=2)
        norm_words = F.normalize(projected_words, dim=-1)
        logits = torch.einsum("bqchw,btc->bqthw", norm_queries, norm_words)
        logits = logits * (self.feature_dim**0.5)
        logits = logits.masked_fill(~attention_mask[:, None, :, None, None], -1.0e4)

        token_weights = torch.softmax(logits, dim=2)
        token_weights = token_weights * attention_mask[:, None, :, None, None].to(logits.dtype)
        patch_scores = (logits * token_weights).sum(dim=2)
        patch_gates = self._straight_through_topk_gate(patch_scores)
        gated_features = projected_queries * patch_gates.unsqueeze(2)

        spatial_logits = logits.flatten(3)
        spatial_weights = torch.softmax(spatial_logits, dim=-1).view_as(logits)
        token_context = torch.einsum("bqthw,bqchw->bqtc", spatial_weights, projected_queries)
        valid = attention_mask[:, None, :, None].to(token_context.dtype)
        phrase_sum = (token_context * valid).sum(dim=2)
        valid_count = valid.sum(dim=2).clamp_min(1.0)
        reconstructed_phrase = self.output_norm(phrase_sum / valid_count)

        return SparseGroundingOutput(
            query_phrase_patch_logits=logits,
            query_gated_features=gated_features,
            query_reconstructed_phrase=reconstructed_phrase,
            query_patch_gates=patch_gates,
        )

    def _straight_through_topk_gate(self, scores: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(scores)
        height, width = scores.shape[-2:]
        k = max(1, round(height * width * self.topk_fraction))
        flat_scores = scores.flatten(2)
        indices = flat_scores.topk(k, dim=-1).indices
        hard = torch.zeros_like(flat_scores).scatter_(2, indices, 1.0).view_as(scores)
        return hard.detach() - soft.detach() + soft

    def _query_tensor(
        self,
        query_features: torch.Tensor | Sequence[QueryOperatorOutput],
    ) -> torch.Tensor:
        if isinstance(query_features, torch.Tensor):
            return query_features
        if len(query_features) != 4:
            raise ValueError("query_features must contain exactly four query outputs")
        return torch.stack([output.features for output in query_features], dim=1)

    def _validate(
        self,
        query_features: torch.Tensor,
        word_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_features.ndim != 5:
            raise ValueError("query_features must have shape [B,4,C,H,W]")
        batch_size, queries, channels, height, width = query_features.shape
        if queries != 4:
            raise ValueError("query_features must contain exactly four query maps")
        if channels != self.feature_dim:
            raise ValueError(
                f"query_features expected {self.feature_dim} channels but received {channels}"
            )
        if height <= 0 or width <= 0:
            raise ValueError("query_features spatial dimensions must be positive")
        if word_features.ndim != 3:
            raise ValueError("word_features must have shape [B,T,C]")
        if word_features.shape[0] != batch_size:
            raise ValueError("word_features must share query_features batch size")
        if word_features.shape[2] != self.feature_dim:
            raise ValueError(
                f"word_features expected {self.feature_dim} channels but received {word_features.shape[2]}"
            )
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B,T]")
        if attention_mask.shape != word_features.shape[:2]:
            raise ValueError("attention_mask must match word_features batch and token dimensions")

        attention_mask = attention_mask.to(dtype=torch.bool)
        if not attention_mask.any(dim=1).all():
            raise ValueError("attention_mask must keep at least one token per sample")
        return query_features, word_features, attention_mask
