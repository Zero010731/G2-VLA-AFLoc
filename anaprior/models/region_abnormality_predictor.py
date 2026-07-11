"""Lightweight region abnormality predictor for learned AnaPrior-Loc."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RegionAbnormalityPredictor(nn.Module):
    """Score whether each anatomical region is abnormal for a finding.

    The module is intentionally small. AFLoc remains frozen; this head consumes
    pooled region features and a finding id, then emits binary logits.
    """

    def __init__(
        self,
        feature_dim: int,
        num_findings: int,
        finding_embedding_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_findings <= 0:
            raise ValueError("num_findings must be positive")

        self.feature_dim = feature_dim
        self.num_findings = num_findings
        self.finding_embedding = nn.Embedding(num_findings, finding_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + finding_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, region_features: torch.Tensor, finding_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for [B,R,C] region features and [B] finding ids."""

        if region_features.ndim != 3:
            raise ValueError("region_features must have shape [B,R,C]")
        if finding_ids.ndim != 1:
            raise ValueError("finding_ids must have shape [B]")
        if region_features.shape[0] != finding_ids.shape[0]:
            raise ValueError("Batch size mismatch between region_features and finding_ids")

        batch, regions, _ = region_features.shape
        finding_emb = self.finding_embedding(finding_ids).unsqueeze(1).expand(batch, regions, -1)
        x = torch.cat([region_features, finding_emb], dim=-1)
        return self.classifier(x).squeeze(-1)

    def score_pairs(self, region_features: torch.Tensor, finding_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for independent [N,C] region-finding pairs."""

        if region_features.ndim != 2:
            raise ValueError("region_features must have shape [N,C]")
        if finding_ids.ndim != 1:
            raise ValueError("finding_ids must have shape [N]")
        if region_features.shape[0] != finding_ids.shape[0]:
            raise ValueError("Row count mismatch between region_features and finding_ids")

        finding_emb = self.finding_embedding(finding_ids)
        x = torch.cat([region_features, finding_emb], dim=-1)
        return self.classifier(x).squeeze(-1)

    @staticmethod
    def loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Binary cross entropy loss with optional validity mask."""

        if logits.shape != labels.shape:
            raise ValueError("logits and labels must have the same shape")
        if valid_mask is not None:
            if valid_mask.shape != logits.shape:
                raise ValueError("valid_mask must have the same shape as logits")
            logits = logits[valid_mask]
            labels = labels[valid_mask]
        if logits.numel() == 0:
            return logits.sum()
        return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)

