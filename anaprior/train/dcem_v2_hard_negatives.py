"""Disease-specific hard-negative priors for DCEM-v2-B training.

These rules are anatomy priors over Chest ImaGenome region names.  They are
used only to construct training pairs inside Chest ImaGenome feature caches;
MS-CXR boxes/masks are not part of these definitions.
"""

from __future__ import annotations


LUNG_REGIONS = frozenset(
    {
        "bilateral_lungs",
        "left_upper_lung",
        "right_upper_lung",
        "left_mid_lung",
        "right_mid_lung",
        "left_lower_lung",
        "right_lower_lung",
    }
)

LOWER_LUNG_REGIONS = frozenset({"left_lower_lung", "right_lower_lung"})
MID_LOWER_LUNG_REGIONS = frozenset({"left_mid_lung", "right_mid_lung", "left_lower_lung", "right_lower_lung"})
PLEURAL_REGIONS = frozenset({"pleural_space_costophrenic"})
CENTRAL_SHORTCUT_REGIONS = frozenset({"cardiac_silhouette", "hilar_mediastinal"})


POSITIVE_PRIOR_REGIONS = {
    "atelectasis": MID_LOWER_LUNG_REGIONS.union({"bilateral_lungs"}),
    "cardiomegaly": frozenset({"cardiac_silhouette"}),
    "consolidation": LUNG_REGIONS,
    "edema": LUNG_REGIONS,
    "lung opacity": MID_LOWER_LUNG_REGIONS.union({"bilateral_lungs"}),
    "pleural effusion": LOWER_LUNG_REGIONS.union(PLEURAL_REGIONS).union({"bilateral_lungs"}),
    "pneumonia": LUNG_REGIONS,
    "pneumothorax": frozenset({"left_upper_lung", "right_upper_lung", "pleural_space_costophrenic"}),
}


HARD_NEGATIVE_REGIONS = {
    "atelectasis": frozenset(),
    "cardiomegaly": LUNG_REGIONS.union(PLEURAL_REGIONS).union(frozenset({"hilar_mediastinal"})),
    "consolidation": CENTRAL_SHORTCUT_REGIONS,
    "edema": frozenset(),
    "lung opacity": CENTRAL_SHORTCUT_REGIONS,
    "pleural effusion": frozenset({"hilar_mediastinal"}),
    "pneumonia": CENTRAL_SHORTCUT_REGIONS,
    "pneumothorax": CENTRAL_SHORTCUT_REGIONS,
}


def normalize_key(value: str) -> str:
    return str(value).strip().lower()


def normalize_region_name(value: str) -> str:
    """Normalize Chest ImaGenome raw bbox names to DCEM region names."""

    raw = " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())
    if not raw:
        return ""
    if raw == "cardiac silhouette" or "cardiac silhouette" in raw:
        return "cardiac_silhouette"
    if "hilar" in raw or "mediastinal" in raw or raw == "mediastinum":
        return "hilar_mediastinal"
    if "costophrenic" in raw or "pleural" in raw:
        return "pleural_space_costophrenic"
    if "upper" in raw and "lung" in raw:
        if raw.startswith("left"):
            return "left_upper_lung"
        if raw.startswith("right"):
            return "right_upper_lung"
    if "mid" in raw and "lung" in raw:
        if raw.startswith("left"):
            return "left_mid_lung"
        if raw.startswith("right"):
            return "right_mid_lung"
    if "lower" in raw and "lung" in raw:
        if raw.startswith("left"):
            return "left_lower_lung"
        if raw.startswith("right"):
            return "right_lower_lung"
    if raw in {"left lung", "right lung", "lung", "lungs", "both lungs", "bilateral lungs"}:
        return "bilateral_lungs"
    return raw.replace(" ", "_")


def positive_prior_regions_for(category: str) -> frozenset[str]:
    return POSITIVE_PRIOR_REGIONS.get(normalize_key(category), frozenset())


def hard_negative_regions_for(category: str) -> frozenset[str]:
    return HARD_NEGATIVE_REGIONS.get(normalize_key(category), frozenset())


def is_hard_negative(category: str, region: str) -> bool:
    return normalize_region_name(region) in hard_negative_regions_for(category)
