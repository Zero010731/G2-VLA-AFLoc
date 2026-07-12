"""Frozen disease spatial properties for property-conditioned DP-MSA.

The values in this module describe clinically motivated spatial attributes.
They are not repair-branch assignments and must not be tuned from MS-CXR box
or oracle outcomes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch


DISEASE_PROPERTY_NAMES: tuple[str, ...] = (
    "focality_focal",
    "focality_regional",
    "focality_diffuse",
    "compartment_parenchymal",
    "compartment_pleural",
    "compartment_cardiac",
    "compartment_mediastinal",
    "laterality_unilateral",
    "laterality_bilateral",
    "vertical_apical",
    "vertical_lower",
    "boundary_dependence",
    "texture_dependence",
    "expected_extent_large",
    "central_shortcut_risk",
)


_TABLE: dict[str, tuple[float, ...]] = {
    "Atelectasis": (
        0.45,
        0.75,
        0.15,
        0.90,
        0.10,
        0.00,
        0.10,
        0.60,
        0.30,
        0.10,
        0.75,
        0.55,
        0.60,
        0.40,
        0.40,
    ),
    "Cardiomegaly": (
        0.05,
        0.75,
        0.20,
        0.00,
        0.00,
        1.00,
        0.30,
        0.00,
        0.85,
        0.00,
        0.20,
        0.95,
        0.10,
        0.90,
        0.10,
    ),
    "Consolidation": (
        0.65,
        0.70,
        0.25,
        1.00,
        0.10,
        0.00,
        0.10,
        0.70,
        0.35,
        0.30,
        0.60,
        0.20,
        0.95,
        0.50,
        0.65,
    ),
    "Edema": (
        0.05,
        0.40,
        1.00,
        0.80,
        0.20,
        0.20,
        0.40,
        0.10,
        1.00,
        0.20,
        0.80,
        0.10,
        0.80,
        1.00,
        0.40,
    ),
    "Lung Opacity": (
        0.50,
        0.80,
        0.50,
        1.00,
        0.10,
        0.10,
        0.10,
        0.60,
        0.50,
        0.40,
        0.60,
        0.20,
        1.00,
        0.60,
        0.70,
    ),
    "Pleural Effusion": (
        0.20,
        0.80,
        0.40,
        0.20,
        1.00,
        0.10,
        0.10,
        0.70,
        0.50,
        0.10,
        1.00,
        0.85,
        0.30,
        0.70,
        0.70,
    ),
    "Pneumonia": (
        0.50,
        0.80,
        0.50,
        1.00,
        0.10,
        0.00,
        0.10,
        0.60,
        0.50,
        0.30,
        0.70,
        0.20,
        1.00,
        0.60,
        0.65,
    ),
    "Pneumothorax": (
        0.85,
        0.70,
        0.20,
        0.30,
        1.00,
        0.00,
        0.10,
        0.80,
        0.30,
        0.85,
        0.10,
        1.00,
        0.20,
        0.50,
        1.00,
    ),
}

DISEASE_PROPERTY_TABLE: Mapping[str, tuple[float, ...]] = MappingProxyType(_TABLE)

_ALIASES: dict[str, str] = {
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pulmonary edema": "Edema",
    "lung opacity": "Lung Opacity",
    "opacity": "Lung Opacity",
    "pleural effusion": "Pleural Effusion",
    "effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
}


def _normalize_category(value: str) -> str:
    text = str(value).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_disease_name(category: str) -> str:
    """Return the canonical finding name for a supported disease category."""

    key = _normalize_category(category)
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(DISEASE_PROPERTY_TABLE)
        raise ValueError(f"Unknown disease category: {category!r}. Supported: {supported}") from exc


def disease_property_vector(category: str) -> tuple[float, ...]:
    """Return the frozen spatial property vector for a disease category."""

    return DISEASE_PROPERTY_TABLE[canonical_disease_name(category)]


def disease_property_matrix(categories: Sequence[str]) -> torch.Tensor:
    """Return property vectors as a float32 tensor with shape `[N, P]`."""

    rows = [disease_property_vector(category) for category in categories]
    return torch.tensor(rows, dtype=torch.float32)
