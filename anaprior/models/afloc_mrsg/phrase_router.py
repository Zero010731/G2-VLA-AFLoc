from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.contracts import PhraseFeatureBatch


@dataclass(frozen=True)
class RouterOutput:
    projected_words: torch.Tensor
    phrase_vector: torch.Tensor
    route_logits: torch.Tensor
    route_weights: torch.Tensor


class PhraseMorphologyRouter(nn.Module):
    def __init__(
        self,
        text_dim: int,
        feature_dim: int,
        num_heads: int = 8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if text_dim <= 0:
            raise ValueError("text_dim must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.text_dim = text_dim
        self.feature_dim = feature_dim
        self.temperature = temperature

        self.word_projection = nn.Linear(text_dim, feature_dim)
        self.sentence_projection = nn.Linear(text_dim, feature_dim)
        self.description_projection = nn.Linear(text_dim, feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.phrase_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.route_head = nn.Linear(feature_dim * 3, 4)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, phrase_features: PhraseFeatureBatch) -> RouterOutput:
        word_embeddings, sentence_embedding, description_embedding, attention_mask = self._validate(
            phrase_features
        )
        projected_words = self.word_projection(word_embeddings)
        masked_projected_words = projected_words * attention_mask.unsqueeze(-1).to(
            dtype=projected_words.dtype
        )
        sentence_query = self.sentence_projection(sentence_embedding)
        description_query = self.description_projection(description_embedding)
        sentence_context = self._attend(sentence_query, masked_projected_words, attention_mask)
        description_context = self._attend(description_query, masked_projected_words, attention_mask)
        phrase_vector = self.output_norm(
            self.phrase_fusion(torch.cat((sentence_context, description_context), dim=-1))
        )
        route_features = torch.cat((sentence_context, description_context, phrase_vector), dim=-1)
        route_logits = self.route_head(route_features)
        route_weights = F.softmax(route_logits / self.temperature, dim=-1)
        return RouterOutput(
            projected_words=masked_projected_words,
            phrase_vector=phrase_vector,
            route_logits=route_logits,
            route_weights=route_weights,
        )

    def _attend(
        self,
        query: torch.Tensor,
        projected_words: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = self.cross_attention(
            query=query.unsqueeze(1),
            key=projected_words,
            value=projected_words,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        return context[:, 0, :]

    def _validate(
        self,
        phrase_features: PhraseFeatureBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        word_embeddings = phrase_features.word_embeddings.detach()
        sentence_embedding = phrase_features.sentence_embedding.detach()
        description_embedding = phrase_features.disease_description_embedding.detach()
        attention_mask = phrase_features.attention_mask.detach()

        if word_embeddings.ndim != 3:
            raise ValueError("word_embeddings must have shape [B,T,D]")
        if sentence_embedding.ndim != 2:
            raise ValueError("sentence_embedding must have shape [B,D]")
        if description_embedding.ndim != 2:
            raise ValueError("disease_description_embedding must have shape [B,D]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B,T]")

        batch_size, tokens, text_dim = word_embeddings.shape
        if text_dim != self.text_dim:
            raise ValueError(
                f"word_embeddings expected trailing dimension {self.text_dim} but received {text_dim}"
            )
        if sentence_embedding.shape != (batch_size, self.text_dim):
            raise ValueError("sentence_embedding must match word_embeddings batch and text dimension")
        if description_embedding.shape != (batch_size, self.text_dim):
            raise ValueError(
                "disease_description_embedding must match word_embeddings batch and text dimension"
            )
        if attention_mask.shape != (batch_size, tokens):
            raise ValueError("attention_mask must match word_embeddings batch and token dimensions")

        attention_mask = attention_mask.to(dtype=torch.bool)
        if not attention_mask.any(dim=1).all():
            raise ValueError("attention_mask must keep at least one token per sample")

        return (
            word_embeddings,
            sentence_embedding,
            description_embedding,
            attention_mask,
        )


def route_balance_loss(route_weights: torch.Tensor) -> torch.Tensor:
    if route_weights.ndim != 2 or route_weights.shape[1] != 4:
        raise ValueError("route_weights must have shape [B,4]")

    target = torch.full_like(route_weights[0], 0.25)
    route_usage = route_weights.mean(dim=0)
    return (route_usage - target).square().mean()
