"""Disease-conditioned region score gating for AnaPrior learned repair."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DiseaseEvidenceConfig:
    allowed_regions: frozenset[str] | None = None
    blocked_regions: frozenset[str] = field(default_factory=frozenset)
    region_weights: dict[str, float] = field(default_factory=dict)
    require_allowed_top: bool = False
    bypass_on_blocked_top: bool = True
    min_top_score: float = 1e-6


@dataclass(frozen=True)
class GatedEvidence:
    scores: np.ndarray
    status: str
    reason: str
    top_region: str
    top_score: float


def _regions(values: list[str] | tuple[str, ...]) -> frozenset[str]:
    return frozenset(str(value).strip().lower() for value in values)


def default_disease_configs() -> dict[str, DiseaseEvidenceConfig]:
    return {
        "atelectasis": DiseaseEvidenceConfig(),
        "consolidation": DiseaseEvidenceConfig(
            allowed_regions=_regions(
                [
                    "bilateral_lungs",
                    "left_upper_lung",
                    "right_upper_lung",
                    "left_mid_lung",
                    "right_mid_lung",
                    "left_lower_lung",
                    "right_lower_lung",
                ]
            ),
            region_weights={
                "cardiac_silhouette": 0.3,
                "hilar_mediastinal": 0.4,
            },
        ),
        "lung opacity": DiseaseEvidenceConfig(
            allowed_regions=_regions(
                [
                    "bilateral_lungs",
                    "left_mid_lung",
                    "right_mid_lung",
                    "left_lower_lung",
                    "right_lower_lung",
                ]
            ),
            region_weights={
                "left_mid_lung": 1.3,
                "right_mid_lung": 1.3,
                "left_lower_lung": 1.4,
                "right_lower_lung": 1.4,
                "cardiac_silhouette": 0.2,
                "hilar_mediastinal": 0.3,
            },
        ),
        "pleural effusion": DiseaseEvidenceConfig(
            region_weights={
                "pleural_space_costophrenic": 2.0,
                "left_lower_lung": 1.5,
                "right_lower_lung": 1.5,
                "bilateral_lungs": 0.8,
                "cardiac_silhouette": 0.3,
                "hilar_mediastinal": 0.2,
            },
        ),
        "pneumothorax": DiseaseEvidenceConfig(
            allowed_regions=_regions(
                [
                    "bilateral_lungs",
                    "left_upper_lung",
                    "right_upper_lung",
                    "pleural_space_costophrenic",
                ]
            ),
            blocked_regions=_regions(["cardiac_silhouette", "hilar_mediastinal"]),
            require_allowed_top=True,
            bypass_on_blocked_top=True,
        ),
    }


def default_disease_pooling_configs() -> dict[str, DiseaseEvidenceConfig]:
    """Disease-specific score transforms for DCEM-v2-B repair maps.

    Unlike `default_disease_configs`, these configs do not bypass just because
    the raw top region is implausible.  They suppress implausible regions and
    keep usable disease-specific evidence for the repair map.
    """

    lung_regions = _regions(
        [
            "bilateral_lungs",
            "left_upper_lung",
            "right_upper_lung",
            "left_mid_lung",
            "right_mid_lung",
            "left_lower_lung",
            "right_lower_lung",
        ]
    )
    mid_lower_lung_regions = _regions(
        [
            "bilateral_lungs",
            "left_mid_lung",
            "right_mid_lung",
            "left_lower_lung",
            "right_lower_lung",
        ]
    )
    return {
        "atelectasis": DiseaseEvidenceConfig(
            allowed_regions=mid_lower_lung_regions,
            region_weights={
                "left_lower_lung": 1.2,
                "right_lower_lung": 1.2,
                "left_mid_lung": 1.1,
                "right_mid_lung": 1.1,
            },
            bypass_on_blocked_top=False,
        ),
        "cardiomegaly": DiseaseEvidenceConfig(
            allowed_regions=_regions(["cardiac_silhouette"]),
            bypass_on_blocked_top=False,
        ),
        "consolidation": DiseaseEvidenceConfig(
            allowed_regions=lung_regions,
            blocked_regions=_regions(["cardiac_silhouette", "hilar_mediastinal"]),
            bypass_on_blocked_top=False,
        ),
        "edema": DiseaseEvidenceConfig(
            region_weights={
                "bilateral_lungs": 1.5,
                "left_upper_lung": 1.0,
                "right_upper_lung": 1.0,
                "left_mid_lung": 1.0,
                "right_mid_lung": 1.0,
                "left_lower_lung": 1.0,
                "right_lower_lung": 1.0,
            },
            bypass_on_blocked_top=False,
        ),
        "lung opacity": DiseaseEvidenceConfig(
            allowed_regions=mid_lower_lung_regions,
            blocked_regions=_regions(["cardiac_silhouette", "hilar_mediastinal"]),
            region_weights={
                "left_mid_lung": 1.3,
                "right_mid_lung": 1.3,
                "left_lower_lung": 1.4,
                "right_lower_lung": 1.4,
            },
            bypass_on_blocked_top=False,
        ),
        "pleural effusion": DiseaseEvidenceConfig(
            region_weights={
                "pleural_space_costophrenic": 2.0,
                "left_lower_lung": 1.5,
                "right_lower_lung": 1.5,
                "bilateral_lungs": 0.8,
                "cardiac_silhouette": 0.3,
                "hilar_mediastinal": 0.0,
            },
            bypass_on_blocked_top=False,
        ),
        "pneumonia": DiseaseEvidenceConfig(
            allowed_regions=lung_regions,
            blocked_regions=_regions(["cardiac_silhouette", "hilar_mediastinal"]),
            bypass_on_blocked_top=False,
        ),
        "pneumothorax": DiseaseEvidenceConfig(
            allowed_regions=_regions(["left_upper_lung", "right_upper_lung", "pleural_space_costophrenic"]),
            blocked_regions=_regions(["cardiac_silhouette", "hilar_mediastinal"]),
            region_weights={
                "left_upper_lung": 1.5,
                "right_upper_lung": 1.5,
                "pleural_space_costophrenic": 1.3,
            },
            bypass_on_blocked_top=False,
        ),
    }


def _zero_scores(scores: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(scores, dtype=np.float32), dtype=np.float32)


def apply_disease_conditioned_gate(
    category: str,
    regions: list[str],
    scores: np.ndarray,
    configs: dict[str, DiseaseEvidenceConfig] | None = None,
) -> GatedEvidence:
    raw_scores = np.asarray(scores, dtype=np.float32)
    if raw_scores.ndim != 1:
        raise ValueError("scores must have shape [R]")
    if len(regions) != raw_scores.shape[0]:
        raise ValueError("regions must match scores length")
    if raw_scores.size == 0:
        return GatedEvidence(raw_scores.copy(), "bypass", "empty_scores", "", 0.0)

    normalized_regions = [str(region).strip().lower() for region in regions]
    finite_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    top_idx = int(np.argmax(finite_scores))
    top_region = normalized_regions[top_idx]
    top_score = float(finite_scores[top_idx])

    config = (configs or default_disease_configs()).get(str(category).strip().lower())
    if config is None:
        return GatedEvidence(finite_scores.copy(), "use", "no_disease_config", top_region, top_score)
    if top_score <= float(config.min_top_score):
        return GatedEvidence(_zero_scores(finite_scores), "bypass", "low_top_score", top_region, top_score)
    if config.bypass_on_blocked_top and top_region in config.blocked_regions:
        return GatedEvidence(_zero_scores(finite_scores), "bypass", "blocked_top_region", top_region, top_score)
    if config.require_allowed_top and config.allowed_regions is not None and top_region not in config.allowed_regions:
        return GatedEvidence(_zero_scores(finite_scores), "bypass", "top_region_not_allowed", top_region, top_score)

    gated = finite_scores.copy()
    for idx, region in enumerate(normalized_regions):
        if region in config.blocked_regions:
            gated[idx] = 0.0
            continue
        if config.allowed_regions is not None and region not in config.allowed_regions:
            gated[idx] = 0.0
            continue
        gated[idx] *= float(config.region_weights.get(region, 1.0))

    if float(np.nanmax(gated)) <= float(config.min_top_score):
        return GatedEvidence(gated, "bypass", "no_usable_disease_evidence", top_region, top_score)
    return GatedEvidence(gated.astype(np.float32), "use", "passed", top_region, top_score)


def apply_disease_specific_pooling(
    category: str,
    regions: list[str],
    scores: np.ndarray,
    configs: dict[str, DiseaseEvidenceConfig] | None = None,
) -> GatedEvidence:
    """Apply DCEM-v2-B disease-specific score pooling without strict bypass."""

    raw_scores = np.asarray(scores, dtype=np.float32)
    if raw_scores.ndim != 1:
        raise ValueError("scores must have shape [R]")
    if len(regions) != raw_scores.shape[0]:
        raise ValueError("regions must match scores length")
    if raw_scores.size == 0:
        return GatedEvidence(raw_scores.copy(), "bypass", "empty_scores", "", 0.0)

    normalized_regions = [str(region).strip().lower() for region in regions]
    finite_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    top_idx = int(np.argmax(finite_scores))
    top_region = normalized_regions[top_idx]
    top_score = float(finite_scores[top_idx])

    config = (configs or default_disease_pooling_configs()).get(str(category).strip().lower())
    if config is None:
        return GatedEvidence(finite_scores.copy(), "use", "no_disease_pooling_config", top_region, top_score)

    pooled = finite_scores.copy()
    for idx, region in enumerate(normalized_regions):
        if region in config.blocked_regions:
            pooled[idx] = 0.0
            continue
        if config.allowed_regions is not None and region not in config.allowed_regions:
            pooled[idx] = 0.0
            continue
        pooled[idx] *= float(config.region_weights.get(region, 1.0))

    if float(np.nanmax(pooled)) <= float(config.min_top_score):
        return GatedEvidence(pooled.astype(np.float32), "bypass", "no_usable_disease_evidence", top_region, top_score)
    return GatedEvidence(pooled.astype(np.float32), "use", "pooled", top_region, top_score)
