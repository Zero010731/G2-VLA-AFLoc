"""Phrase-anatomy routing for DCEM-v3 learned repair.

The router is intentionally deterministic and rule-based. It uses generic
anatomical terms from the phrase text to reweight already-learned region
evidence; it does not read MS-CXR boxes, masks, or oracle profiles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from anaprior.eval.disease_conditioned_gate import (
    GatedEvidence,
    apply_disease_specific_pooling,
    default_disease_pooling_configs,
)


@dataclass(frozen=True)
class PhraseAnatomy:
    laterality: frozenset[str]
    vertical: frozenset[str]
    compartments: frozenset[str]
    tokens: frozenset[str]


LEFT_TERMS = frozenset({"left", "lt"})
RIGHT_TERMS = frozenset({"right", "rt"})
BILATERAL_TERMS = frozenset({"bilateral", "bibasilar", "both", "diffuse"})
UPPER_TERMS = frozenset({"upper", "apical", "apex", "suprahilar"})
MID_TERMS = frozenset({"mid", "middle", "perihilar"})
LOWER_TERMS = frozenset({"lower", "basal", "base", "basilar", "bibasilar", "infrahilar"})
DIFFUSE_TERMS = frozenset({"diffuse", "bilateral", "multifocal", "widespread", "interstitial"})
PLEURAL_TERMS = frozenset({"pleural", "costophrenic", "pneumothorax"})
CARDIAC_TERMS = frozenset({"cardiac", "cardiomegaly", "retrocardiac", "heart", "silhouette"})
HILAR_TERMS = frozenset({"hilar", "hilum", "mediastinal", "mediastinum", "perihilar"})
LUNG_TERMS = frozenset(
    {
        "lung",
        "lungs",
        "lobe",
        "lobar",
        "opacity",
        "opacities",
        "consolidation",
        "pneumonia",
        "atelectasis",
        "edema",
        "infiltrate",
        "airspace",
    }
)

PARENCHYMAL_FINDINGS = frozenset({"atelectasis", "consolidation", "lung opacity", "pneumonia"})


def _normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return set(_normalize_text(text).split())


def _has_phrase(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def _has_any(text: str, terms: frozenset[str]) -> bool:
    return any(_has_phrase(text, term) for term in terms)


def parse_phrase_anatomy(phrase: str) -> PhraseAnatomy:
    """Parse generic anatomical hints from a report phrase."""

    text = _normalize_text(phrase)
    token_set = _tokens(text)

    laterality: set[str] = set()
    if _has_any(text, BILATERAL_TERMS):
        laterality.add("bilateral")
    if _has_any(text, LEFT_TERMS):
        laterality.add("left")
    if _has_any(text, RIGHT_TERMS):
        laterality.add("right")
    if "left" in laterality and "right" in laterality:
        laterality = {"bilateral"}

    vertical: set[str] = set()
    if _has_any(text, UPPER_TERMS):
        vertical.add("upper")
    if _has_any(text, MID_TERMS):
        vertical.add("mid")
    if _has_any(text, LOWER_TERMS):
        vertical.add("lower")
    if _has_any(text, DIFFUSE_TERMS):
        vertical.add("diffuse")

    compartments: set[str] = set()
    if _has_any(text, PLEURAL_TERMS):
        compartments.add("pleural")
    if _has_any(text, CARDIAC_TERMS):
        compartments.add("cardiac")
    if _has_any(text, HILAR_TERMS):
        compartments.add("hilar")
    if _has_any(text, LUNG_TERMS) or vertical or laterality:
        compartments.add("lung")

    return PhraseAnatomy(
        laterality=frozenset(laterality),
        vertical=frozenset(vertical),
        compartments=frozenset(compartments),
        tokens=frozenset(token_set),
    )


def _normal_regions(regions: list[str]) -> list[str]:
    return [str(region).strip().lower() for region in regions]


def _category_key(category: str) -> str:
    return str(category or "").strip().lower()


def _explicitly_mentions_region(anatomy: PhraseAnatomy, region: str) -> bool:
    if region == "cardiac_silhouette":
        return "cardiac" in anatomy.compartments
    if region == "hilar_mediastinal":
        return "hilar" in anatomy.compartments
    if region == "pleural_space_costophrenic":
        return "pleural" in anatomy.compartments
    return False


def _region_matches_laterality(region: str, anatomy: PhraseAnatomy) -> bool:
    if "bilateral" in anatomy.laterality:
        return "lung" in region or region == "pleural_space_costophrenic"
    if "left" in anatomy.laterality:
        return region.startswith("left_")
    if "right" in anatomy.laterality:
        return region.startswith("right_")
    return False


def _region_conflicts_laterality(region: str, anatomy: PhraseAnatomy) -> bool:
    if "left" in anatomy.laterality and region.startswith("right_"):
        return True
    if "right" in anatomy.laterality and region.startswith("left_"):
        return True
    return False


def _region_matches_vertical(region: str, anatomy: PhraseAnatomy) -> bool:
    if "upper" in anatomy.vertical and "upper_lung" in region:
        return True
    if "mid" in anatomy.vertical and "mid_lung" in region:
        return True
    if "lower" in anatomy.vertical and "lower_lung" in region:
        return True
    if "diffuse" in anatomy.vertical and (region == "bilateral_lungs" or "lung" in region):
        return True
    return False


def _region_conflicts_vertical(region: str, anatomy: PhraseAnatomy) -> bool:
    lung_zone = any(marker in region for marker in ("upper_lung", "mid_lung", "lower_lung"))
    if not lung_zone:
        return False
    if "upper" in anatomy.vertical and "upper_lung" not in region:
        return True
    if "lower" in anatomy.vertical and "lower_lung" not in region:
        return True
    if "mid" in anatomy.vertical and "mid_lung" not in region:
        return True
    return False


def _phrase_region_weight(category: str, region: str, anatomy: PhraseAnatomy) -> float:
    weight = 1.0
    if _region_matches_laterality(region, anatomy):
        weight *= 1.8
    elif _region_conflicts_laterality(region, anatomy):
        weight *= 0.35

    if _region_matches_vertical(region, anatomy):
        weight *= 1.8
    elif _region_conflicts_vertical(region, anatomy):
        weight *= 0.55

    if "pleural" in anatomy.compartments and region == "pleural_space_costophrenic":
        weight *= 1.8
    if "cardiac" in anatomy.compartments and region == "cardiac_silhouette":
        weight *= 1.4
    if "hilar" in anatomy.compartments and region == "hilar_mediastinal":
        weight *= 1.4

    if _category_key(category) == "pneumothorax" and region in {"cardiac_silhouette", "hilar_mediastinal"}:
        return 0.0
    return weight


def _coarse_disease_weight(category: str, region: str, anatomy: PhraseAnatomy) -> float:
    config = default_disease_pooling_configs().get(_category_key(category))
    if config is None:
        return 1.0

    explicit_region = _explicitly_mentions_region(anatomy, region)
    can_unblock_central = _category_key(category) in PARENCHYMAL_FINDINGS and explicit_region
    if region in config.blocked_regions and not can_unblock_central:
        return 0.0
    if config.allowed_regions is not None and region not in config.allowed_regions and not explicit_region:
        return 0.0
    return float(config.region_weights.get(region, 1.0))


def _top_region(regions: list[str], scores: np.ndarray) -> tuple[str, float]:
    if scores.size == 0:
        return "", 0.0
    idx = int(np.argmax(scores))
    return regions[idx], float(scores[idx])


def apply_phrase_anatomy_router(
    category: str,
    phrase: str,
    regions: list[str],
    scores: np.ndarray,
) -> GatedEvidence:
    """Apply DCEM-v3 phrase-anatomy region routing to learned region scores."""

    raw_scores = np.asarray(scores, dtype=np.float32)
    if raw_scores.ndim != 1:
        raise ValueError("scores must have shape [R]")
    if len(regions) != raw_scores.shape[0]:
        raise ValueError("regions must match scores length")
    if raw_scores.size == 0:
        return GatedEvidence(raw_scores.copy(), "bypass", "empty_scores", "", 0.0)

    normalized_regions = _normal_regions(regions)
    finite_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    top_region, top_score = _top_region(normalized_regions, finite_scores)

    if not _normalize_text(phrase):
        fallback = apply_disease_specific_pooling(category, normalized_regions, finite_scores)
        return GatedEvidence(
            fallback.scores,
            fallback.status,
            "phrase_missing_disease_pooling_fallback" if fallback.status == "use" else fallback.reason,
            top_region,
            top_score,
        )

    anatomy = parse_phrase_anatomy(phrase)
    routed = finite_scores.copy()
    for idx, region in enumerate(normalized_regions):
        routed[idx] *= _coarse_disease_weight(category, region, anatomy)
        routed[idx] *= _phrase_region_weight(category, region, anatomy)

    if float(np.nanmax(routed)) <= 1e-6:
        return GatedEvidence(
            routed.astype(np.float32),
            "bypass",
            "no_usable_phrase_anatomy_evidence",
            top_region,
            top_score,
        )
    return GatedEvidence(
        routed.astype(np.float32),
        "use",
        "phrase_anatomy_routed",
        top_region,
        top_score,
    )
