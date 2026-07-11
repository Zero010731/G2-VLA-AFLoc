"""Frozen phrase subtype rules for DP-MSA.

The subtype encoder is intentionally deterministic. It uses clinical/anatomical
vocabulary only and must not be tuned from MS-CXR box outcomes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


PHRASE_SUBTYPES: tuple[str, ...] = (
    "basilar",
    "upper_apical",
    "mid_lung",
    "focal",
    "multifocal_patchy",
    "bilateral_diffuse",
    "airspace",
    "ground_glass",
    "retrocardiac_hilar",
    "pneumonia_like",
    "consolidation_like",
    "opacity_like",
    "other",
)
PHRASE_SUBTYPE_TO_ID = {name: idx for idx, name in enumerate(PHRASE_SUBTYPES)}


@dataclass(frozen=True)
class PhraseSubtype:
    name: str
    id: int
    matched_terms: tuple[str, ...]


def _norm_text(value: str) -> str:
    text = str(value).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _matches(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    found = []
    padded = f" {text} "
    for term in terms:
        norm = _norm_text(term)
        if not norm:
            continue
        if f" {norm} " in padded:
            found.append(term)
    return tuple(found)


RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ground_glass", ("ground glass", "ground-glass", "gg opacity", "ggo")),
    ("retrocardiac_hilar", ("retrocardiac", "retro cardiac", "hilar", "perihilar", "infrahilar")),
    ("basilar", ("basilar", "bibasilar", "base", "bases", "lower lobe", "lower lung", "lung base")),
    ("upper_apical", ("apical", "apex", "upper lobe", "upper lung", "right upper", "left upper")),
    ("mid_lung", ("mid lung", "middle lobe", "right middle", "left middle", "lingula", "lingular")),
    ("focal", ("focal", "small area", "small focal", "nodular", "solitary")),
    ("multifocal_patchy", ("multifocal", "patchy", "multilobar")),
    ("bilateral_diffuse", ("bilateral", "diffuse", "widespread", "both lungs")),
    ("airspace", ("airspace", "air space", "air-space")),
    ("pneumonia_like", ("pneumonia", "infection", "infectious")),
    ("consolidation_like", ("consolidation", "consolidative", "consolidations")),
    ("opacity_like", ("opacity", "opacities", "opaque")),
)


def infer_phrase_subtype(phrase: str, category: str = "") -> PhraseSubtype:
    """Infer the first matching frozen DP-MSA phrase subtype."""

    text = _norm_text(f"{phrase} {category}")
    for subtype, terms in RULES:
        matched = _matches(text, terms)
        if matched:
            return PhraseSubtype(
                name=subtype,
                id=PHRASE_SUBTYPE_TO_ID[subtype],
                matched_terms=matched,
            )
    return PhraseSubtype(name="other", id=PHRASE_SUBTYPE_TO_ID["other"], matched_terms=())


def phrase_subtype_id(phrase: str, category: str = "") -> int:
    return infer_phrase_subtype(phrase, category=category).id
