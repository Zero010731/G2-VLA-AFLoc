from __future__ import annotations

import torch
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch, PhraseFeatureBatch


def normalize_anchor_map(values: torch.Tensor) -> torch.Tensor:
    low = values.amin(dim=(-2, -1), keepdim=True)
    high = values.amax(dim=(-2, -1), keepdim=True)
    scale = (high - low).clamp_min(1.0e-6)
    return ((values - low) / scale).clamp(0.0, 1.0)


def compute_afloc_phrase_anchor(
    image_features: AFLocFeatureBatch,
    phrase_features: PhraseFeatureBatch,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build a frozen, multi-scale AFLoc phrase-patch anchor.

    The confidence map is based only on agreement among AFLoc scales. It does
    not use annotations or category IDs and is detached from optimization.
    """
    words = phrase_features.word_embeddings.detach()
    mask = phrase_features.attention_mask.detach().to(dtype=torch.bool)
    if words.ndim != 3 or mask.shape != words.shape[:2]:
        raise ValueError("phrase features have incompatible word and mask shapes")
    if not mask.any(dim=1).all():
        raise ValueError("every phrase must contain at least one valid token")

    target_size = tuple(image_features.img_emb_l2.shape[-2:])
    maps: dict[str, torch.Tensor] = {}
    for name in ("l2", "l", "lf"):
        feature_map = getattr(image_features, f"img_emb_{name}").detach()
        if feature_map.shape[1] != words.shape[2]:
            raise ValueError(f"{name} image channels must match phrase feature dimension")
        image_norm = F.normalize(feature_map, dim=1)
        word_norm = F.normalize(words, dim=-1)
        scores = torch.einsum("bchw,btc->bthw", image_norm, word_norm)
        scores = scores.masked_fill(~mask[:, :, None, None], -1.0e4)
        scale_map = normalize_anchor_map(scores.amax(dim=1, keepdim=True))
        if tuple(scale_map.shape[-2:]) != target_size:
            scale_map = F.interpolate(
                scale_map,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        maps[name] = scale_map

    stacked = torch.stack((maps["l2"], maps["l"], maps["lf"]), dim=1)
    anchor = stacked.mean(dim=1)
    confidence = (1.0 - 2.0 * stacked.sub(anchor[:, None]).abs().mean(dim=1)).clamp(0.0, 1.0)
    return anchor.detach(), confidence.detach(), maps
