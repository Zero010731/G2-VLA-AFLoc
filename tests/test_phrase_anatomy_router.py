import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from anaprior.eval.phrase_anatomy_router import (
    apply_phrase_anatomy_router,
    parse_phrase_anatomy,
)


def test_parse_phrase_anatomy_extracts_laterality_vertical_and_compartment() -> None:
    anatomy = parse_phrase_anatomy("small left apical pneumothorax along the pleural line")

    assert anatomy.laterality == frozenset({"left"})
    assert "upper" in anatomy.vertical
    assert "pleural" in anatomy.compartments
    assert "apical" in anatomy.tokens


def test_parse_phrase_anatomy_marks_bilateral_diffuse_lower_lung_edema() -> None:
    anatomy = parse_phrase_anatomy("bilateral diffuse lower lung pulmonary edema")

    assert anatomy.laterality == frozenset({"bilateral"})
    assert {"diffuse", "lower"} <= anatomy.vertical
    assert "lung" in anatomy.compartments


def test_parse_phrase_anatomy_keeps_cardiac_and_hilar_compartments() -> None:
    anatomy = parse_phrase_anatomy("retrocardiac opacity near the left hilar region")

    assert anatomy.laterality == frozenset({"left"})
    assert "cardiac" in anatomy.compartments
    assert "hilar" in anatomy.compartments


def test_pneumothorax_phrase_routes_to_left_apical_region_and_suppresses_cardiac() -> None:
    regions = ["cardiac_silhouette", "left_upper_lung", "right_upper_lung", "pleural_space_costophrenic"]
    scores = np.array([0.95, 0.35, 0.30, 0.20], dtype=np.float32)

    routed = apply_phrase_anatomy_router(
        "Pneumothorax",
        "small left apical pneumothorax",
        regions,
        scores,
    )

    assert routed.status == "use"
    assert routed.scores[0] == 0.0
    assert routed.scores[1] > routed.scores[2]
    assert routed.scores[1] > routed.scores[3]
    assert routed.reason == "phrase_anatomy_routed"


def test_pneumonia_phrase_routes_right_lower_lung_over_central_shortcut() -> None:
    regions = ["cardiac_silhouette", "right_lower_lung", "left_lower_lung", "right_upper_lung"]
    scores = np.array([0.90, 0.36, 0.34, 0.30], dtype=np.float32)

    routed = apply_phrase_anatomy_router(
        "Pneumonia",
        "right lower lobe pneumonia",
        regions,
        scores,
    )

    assert routed.status == "use"
    assert routed.scores[1] > routed.scores[0]
    assert routed.scores[1] > routed.scores[2]
    assert routed.scores[1] > routed.scores[3]


def test_missing_phrase_falls_back_to_disease_specific_pooling() -> None:
    regions = ["cardiac_silhouette", "left_upper_lung"]
    scores = np.array([0.9, 0.2], dtype=np.float32)

    routed = apply_phrase_anatomy_router("Pneumothorax", "", regions, scores)

    assert routed.status == "use"
    assert routed.reason == "phrase_missing_disease_pooling_fallback"
    assert routed.scores[0] == 0.0
    assert routed.scores[1] > 0.0
