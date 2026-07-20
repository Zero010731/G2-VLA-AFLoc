from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch
from torch import nn

from anaprior.models.afloc_mrsg.contracts import (
    AFLocFeatureBatch,
    MRSGConfig,
    MRSGOutput,
    PhraseFeatureBatch,
)
from anaprior.models.afloc_mrsg.dense_decoder import AnchorBoundedResidualDecoder
from anaprior.models.afloc_mrsg.feature_pyramid import LocalityAlignedFeaturePyramid
from anaprior.models.afloc_mrsg.grounding_transformer import (
    MultiQuerySparsePhrasePatchGrounder,
)
from anaprior.models.afloc_mrsg.phrase_router import PhraseMorphologyRouter
from anaprior.models.afloc_mrsg.query_operators import MorphologyQueryBank


class AFLocMRSG(nn.Module):
    """Anchor-preserving AFLoc-MRSG forward graph over frozen AFLoc features."""

    def __init__(
        self,
        config: MRSGConfig | None = None,
        image_channels: Sequence[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.config = config or MRSGConfig()
        self.feature_pyramid = LocalityAlignedFeaturePyramid(
            source_channels=image_channels,
            feature_dim=self.config.feature_dim,
            num_heads=self.config.num_heads,
        )
        self.phrase_router = PhraseMorphologyRouter(
            text_dim=self.config.text_dim,
            feature_dim=self.config.feature_dim,
            num_heads=self.config.num_heads,
            temperature=self.config.route_temperature,
        )
        self.query_bank = MorphologyQueryBank(
            feature_dim=self.config.feature_dim,
            num_heads=self.config.num_heads,
            focal_slots=self.config.focal_slots,
            focal_fraction=self.config.topk_fraction,
        )
        self.grounder = MultiQuerySparsePhrasePatchGrounder(
            feature_dim=self.config.feature_dim,
            num_heads=self.config.num_heads,
            topk_fraction=self.config.topk_fraction,
        )
        self.decoder = AnchorBoundedResidualDecoder(
            feature_dim=self.config.feature_dim,
            residual_logit_bound=self.config.residual_logit_bound,
            residual_logit_cap=self.config.residual_logit_cap,
        )
        self.last_forward_debug: dict[str, torch.Tensor] = {}

    def forward(
        self,
        image_features: AFLocFeatureBatch,
        phrase_features: PhraseFeatureBatch,
        official_anchor: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> MRSGOutput:
        patch_mask = self._validate_and_prepare_inputs(
            image_features,
            phrase_features,
            patch_mask,
        )

        pyramid = self.feature_pyramid(image_features, patch_mask)
        router = self.phrase_router(phrase_features)
        query_outputs = self.query_bank(
            pyramid.fused,
            pyramid.edge_features,
            router.phrase_vector,
        )
        query_tensor = torch.stack([item.features for item in query_outputs], dim=1)
        grounding = self.grounder(
            query_tensor,
            router.projected_words,
            phrase_features.attention_mask.detach(),
        )

        grounded_query_logits = torch.cat(
            [item.heatmap_logits for item in query_outputs],
            dim=1,
        )
        grounded_query_outputs = tuple(
            replace(
                item,
                heatmap_logits=grounded_query_logits[:, index : index + 1],
            )
            for index, item in enumerate(query_outputs)
        )

        decoder_output = self.decoder(
            pyramid.fused,
            grounded_query_outputs,
            router.route_weights,
            grounding.query_patch_gates,
            official_anchor,
        )
        query_heatmaps = torch.sigmoid(grounded_query_logits)
        phrase_patch_logits = (
            grounding.query_phrase_patch_logits
            * router.route_weights[:, :, None, None, None]
        ).sum(dim=1)
        query_reliability = self._query_reliability(
            grounded_query_outputs,
            grounding.query_patch_gates,
        )

        output = MRSGOutput(
            final_heatmap=decoder_output.final_heatmap,
            query_heatmaps=query_heatmaps,
            query_route_weights=router.route_weights,
            query_reliability=query_reliability,
            phrase_patch_logits=phrase_patch_logits,
            masked_predictions=pyramid.masked_prediction,
            source_targets=pyramid.source_targets,
            patch_mask=patch_mask,
            query_reconstructed_phrase=grounding.query_reconstructed_phrase,
            query_patch_gates=grounding.query_patch_gates,
            anchor_heatmap=official_anchor.detach(),
            residual_logits=decoder_output.residual_logits,
            bounded_correction=decoder_output.bounded_correction,
            correction_bound=decoder_output.correction_bound,
            raw_residual_logits=decoder_output.raw_residual_logits,
        )
        output.validate()
        self.last_forward_debug = {
            "grounded_query_logits": grounded_query_logits.detach(),
            "query_phrase_patch_logits": grounding.query_phrase_patch_logits.detach(),
            "query_patch_gates": grounding.query_patch_gates.detach(),
            "residual_logits": decoder_output.residual_logits.detach(),
            "bounded_correction": decoder_output.bounded_correction.detach(),
        }
        return output

    def _validate_and_prepare_inputs(
        self,
        image_features: AFLocFeatureBatch,
        phrase_features: PhraseFeatureBatch,
        patch_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not isinstance(image_features, AFLocFeatureBatch):
            raise ValueError("image_features must be an AFLocFeatureBatch")
        if not isinstance(phrase_features, PhraseFeatureBatch):
            raise ValueError("phrase_features must be a PhraseFeatureBatch")

        batch_size = image_features.img_emb_l2.shape[0]
        if phrase_features.word_embeddings.shape[0] != batch_size:
            raise ValueError("image_features and phrase_features must share the same batch size")

        target_size = image_features.img_emb_l2.shape[-2:]
        if patch_mask is None:
            patch_mask = torch.zeros(
                batch_size,
                1,
                *target_size,
                dtype=torch.bool,
                device=image_features.img_emb_l2.device,
            )
        if patch_mask.device != image_features.img_emb_l2.device:
            patch_mask = patch_mask.to(device=image_features.img_emb_l2.device)
        return patch_mask

    def _query_reliability(
        self,
        query_outputs: Sequence,
        query_patch_gates: torch.Tensor,
    ) -> torch.Tensor:
        operator_reliability = torch.cat(
            [item.reliability for item in query_outputs],
            dim=1,
        )
        support = query_patch_gates.mean(dim=(-2, -1))
        activation_variance = torch.stack(
            [item.features.var(dim=(-2, -1), unbiased=False).mean(dim=1) for item in query_outputs],
            dim=1,
        )
        variance_score = activation_variance / (activation_variance + 1.0)
        return (0.5 * operator_reliability + 0.25 * support + 0.25 * variance_score).clamp(
            0.0,
            1.0,
        )
