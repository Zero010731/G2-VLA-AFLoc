from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MRSGConfig:
    feature_dim: int = 256
    text_dim: int = 768
    num_heads: int = 8
    focal_slots: int = 4
    topk_fraction: float = 0.15
    route_temperature: float = 1.0
    query_names: tuple[str, ...] = ("focal", "diffuse", "boundary", "structural")

    def __post_init__(self) -> None:
        if len(self.query_names) != 4:
            raise ValueError("MRSG requires exactly four query operators")
        if self.query_names != ("focal", "diffuse", "boundary", "structural"):
            raise ValueError(
                "MRSG query operators must be focal, diffuse, boundary, structural"
            )
        if self.feature_dim % self.num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not 0.0 < self.topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0,1]")


@dataclass(frozen=True)
class AFLocFeatureBatch:
    img_emb_l2: torch.Tensor
    img_emb_l: torch.Tensor
    img_emb_lf: torch.Tensor
    image_gray: torch.Tensor


@dataclass(frozen=True)
class PhraseFeatureBatch:
    word_embeddings: torch.Tensor
    sentence_embedding: torch.Tensor
    disease_description_embedding: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class MRSGOutput:
    final_heatmap: torch.Tensor
    query_heatmaps: torch.Tensor
    query_route_weights: torch.Tensor
    query_reliability: torch.Tensor
    phrase_patch_logits: torch.Tensor
    masked_predictions: dict[str, torch.Tensor] | None = None
    source_targets: dict[str, torch.Tensor] | None = None
    patch_mask: torch.Tensor | None = None
    query_reconstructed_phrase: torch.Tensor | None = None
    query_patch_gates: torch.Tensor | None = None

    def validate(self) -> None:
        if self.final_heatmap.ndim != 4:
            raise ValueError("final_heatmap must have shape [B,1,H,W]")

        batch, channels, height, width = self.final_heatmap.shape
        if channels != 1:
            raise ValueError("final_heatmap must have one channel")
        if self.query_heatmaps.shape != (batch, 4, height, width):
            raise ValueError("query_heatmaps must have shape [B,4,H,W]")
        if self.query_route_weights.shape != (batch, 4):
            raise ValueError("query_route_weights must have shape [B,4]")
        if self.query_reliability.shape != (batch, 4):
            raise ValueError("query_reliability must have shape [B,4]")
        if self.phrase_patch_logits.ndim != 4:
            raise ValueError("phrase_patch_logits must have shape [B,T,H,W]")
        if self.phrase_patch_logits.shape[0] != batch:
            raise ValueError("phrase_patch_logits must have the same batch size")
        if self.phrase_patch_logits.shape[-2:] != (height, width):
            raise ValueError("phrase_patch_logits must have the same spatial shape")
        self._validate_training_fields(batch, height, width)

    def _validate_training_fields(self, batch: int, height: int, width: int) -> None:
        if self.masked_predictions is not None or self.source_targets is not None:
            if self.masked_predictions is None or self.source_targets is None:
                raise ValueError("masked_predictions and source_targets must be provided together")
            expected_keys = {"l2", "l", "lf"}
            if set(self.masked_predictions) != expected_keys:
                raise ValueError("masked_predictions must contain l2, l, and lf")
            if set(self.source_targets) != expected_keys:
                raise ValueError("source_targets must contain l2, l, and lf")
            for name in ("l2", "l", "lf"):
                prediction = self.masked_predictions[name]
                target = self.source_targets[name]
                if prediction.ndim != 4:
                    raise ValueError(f"masked_predictions[{name}] must have shape [B,C,H,W]")
                if prediction.shape[0] != batch or prediction.shape[-2:] != (height, width):
                    raise ValueError(
                        f"masked_predictions[{name}] must match final_heatmap batch and spatial shape"
                    )
                if target.shape != prediction.shape:
                    raise ValueError(
                        f"source_targets[{name}] must match masked_predictions[{name}]"
                    )
                if target.requires_grad:
                    raise ValueError("source_targets must be detached")

        if self.patch_mask is not None:
            if self.patch_mask.shape != (batch, 1, height, width):
                raise ValueError("patch_mask must have shape [B,1,H,W]")
            if self.patch_mask.dtype != torch.bool:
                raise ValueError("patch_mask must be a boolean tensor")

        if self.query_reconstructed_phrase is not None:
            if self.query_reconstructed_phrase.ndim != 3:
                raise ValueError("query_reconstructed_phrase must have shape [B,4,C]")
            if self.query_reconstructed_phrase.shape[:2] != (batch, 4):
                raise ValueError("query_reconstructed_phrase must have shape [B,4,C]")

        if self.query_patch_gates is not None:
            if self.query_patch_gates.shape != (batch, 4, height, width):
                raise ValueError("query_patch_gates must have shape [B,4,H,W]")
