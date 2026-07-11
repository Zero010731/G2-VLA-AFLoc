import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from anaprior.eval.selective_repair import (
    RepairCase,
    apply_repair_cases,
    blend_heatmap,
    repair_map_from_region_scores,
    shuffle_region_scores,
)


def test_blend_heatmap_scales_inputs_and_preserves_nan() -> None:
    hmap = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    repair = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    fused = blend_heatmap(hmap, repair, alpha=0.5)

    assert np.isnan(fused[1, 0])
    assert fused[0, 0] == 0.0
    assert fused[0, 1] > fused[0, 0]
    assert fused[1, 1] == 1.0


def test_repair_map_uses_only_positive_scored_regions() -> None:
    region_maps = np.zeros((2, 3, 3), dtype=np.float32)
    region_maps[0, :2, :] = 1.0
    region_maps[1, 2:, :] = 1.0
    scores = np.array([0.0, 1.0], dtype=np.float32)

    repair = repair_map_from_region_scores(region_maps, scores)

    assert repair is not None
    assert repair[:2, :].max() == 0.0
    assert repair[2:, :].max() == 1.0


def test_apply_repair_cases_repairs_candidates_and_keeps_non_candidates() -> None:
    candidate = RepairCase(
        case_id="case-candidate",
        category="Pneumothorax",
        heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        region_maps=np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        region_scores=np.array([1.0, 0.0], dtype=np.float32),
    )
    non_candidate = RepairCase(
        case_id="case-non-candidate",
        category="Edema",
        heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        region_maps=candidate.region_maps.copy(),
        region_scores=np.array([1.0, 0.0], dtype=np.float32),
    )

    repaired, stats = apply_repair_cases(
        [candidate, non_candidate],
        candidate_categories={"Pneumothorax"},
        alpha=1.0,
        repair_scope="candidate",
    )

    assert repaired["case-candidate"].status == "repaired"
    assert repaired["case-candidate"].heatmap[0, 0] > repaired["case-candidate"].heatmap[1, 0]
    assert repaired["case-non-candidate"].status == "baseline_copy"
    assert np.allclose(repaired["case-non-candidate"].heatmap, non_candidate.heatmap)
    assert stats["repaired_categories"]["Pneumothorax"] == 1
    assert stats["skipped_non_candidate"]["Edema"] == 1


def test_all_class_scope_can_repair_non_candidate_classes() -> None:
    case = RepairCase(
        case_id="case-edema",
        category="Edema",
        heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        region_maps=np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        region_scores=np.array([1.0, 0.0], dtype=np.float32),
    )

    repaired, stats = apply_repair_cases(
        [case],
        candidate_categories={"Pneumothorax"},
        alpha=1.0,
        repair_scope="all",
    )

    assert repaired["case-edema"].status == "repaired"
    assert repaired["case-edema"].heatmap[0, 0] > repaired["case-edema"].heatmap[1, 0]
    assert stats["repaired_categories"]["Edema"] == 1


def test_shuffled_region_scores_preserve_mass_but_move_location() -> None:
    scores = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    shuffled = shuffle_region_scores(scores, seed_key="case:Pneumothorax")

    assert np.isclose(shuffled.sum(), scores.sum())
    assert shuffled.max() == 1.0
    assert not np.allclose(shuffled, scores)


def test_uniform_score_mode_ignores_learned_scores() -> None:
    case = RepairCase(
        case_id="case-uniform",
        category="Pneumothorax",
        heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        region_maps=np.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        region_scores=np.array([0.0, 3.0], dtype=np.float32),
    )

    learned, _ = apply_repair_cases([case], {"Pneumothorax"}, alpha=1.0, score_mode="learned")
    uniform, _ = apply_repair_cases([case], {"Pneumothorax"}, alpha=1.0, score_mode="uniform")

    assert learned["case-uniform"].heatmap[1, 1] > learned["case-uniform"].heatmap[0, 0]
    assert uniform["case-uniform"].heatmap[0, 0] == uniform["case-uniform"].heatmap[1, 1]


def test_disease_gated_mode_bypasses_implausible_pneumothorax_without_changing_raw_learned() -> None:
    case = RepairCase(
        case_id="case-ptx-central",
        category="Pneumothorax",
        heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        region_maps=np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        region_scores=np.array([0.9, 0.2], dtype=np.float32),
        regions=["cardiac_silhouette", "left_upper_lung"],
    )

    raw, raw_stats = apply_repair_cases([case], {"Pneumothorax"}, alpha=1.0, score_mode="learned")
    gated, gated_stats = apply_repair_cases([case], {"Pneumothorax"}, alpha=1.0, score_mode="disease_gated")

    assert raw["case-ptx-central"].status == "repaired"
    assert raw["case-ptx-central"].heatmap[0, 0] > raw["case-ptx-central"].heatmap[1, 0]
    assert raw_stats["repaired_categories"]["Pneumothorax"] == 1
    assert gated["case-ptx-central"].status == "disease_gate_bypass:blocked_top_region"
    assert np.allclose(gated["case-ptx-central"].heatmap, case.heatmap)
    assert gated_stats["gate_bypassed_categories"]["Pneumothorax"] == 1


def test_disease_pooled_mode_suppresses_shortcut_and_repairs_allowed_pneumothorax_region() -> None:
    case = RepairCase(
        case_id="case-ptx-pooled",
        category="Pneumothorax",
        heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        region_maps=np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        region_scores=np.array([0.9, 0.2], dtype=np.float32),
        regions=["cardiac_silhouette", "left_upper_lung"],
    )

    repaired, stats = apply_repair_cases([case], {"Pneumothorax"}, alpha=1.0, score_mode="disease_pooled")

    assert repaired["case-ptx-pooled"].status == "repaired"
    assert repaired["case-ptx-pooled"].heatmap[1, 0] > repaired["case-ptx-pooled"].heatmap[0, 0]
    assert stats["gate_passed_categories"]["Pneumothorax"] == 1


def test_phrase_anatomy_mode_routes_pneumothorax_by_phrase_without_changing_raw_learned() -> None:
    case = RepairCase(
        case_id="case-ptx-phrase",
        category="Pneumothorax",
        phrase="right apical pneumothorax",
        heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        region_maps=np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        region_scores=np.array([0.95, 0.35, 0.30], dtype=np.float32),
        regions=["cardiac_silhouette", "right_upper_lung", "left_upper_lung"],
    )

    raw, raw_stats = apply_repair_cases([case], {"Pneumothorax"}, alpha=1.0, score_mode="learned")
    routed, routed_stats = apply_repair_cases([case], {"Pneumothorax"}, alpha=1.0, score_mode="phrase_anatomy")

    assert raw["case-ptx-phrase"].status == "repaired"
    assert raw["case-ptx-phrase"].heatmap[0, 0] > raw["case-ptx-phrase"].heatmap[1, 1]
    assert raw_stats["repaired_categories"]["Pneumothorax"] == 1
    assert routed["case-ptx-phrase"].status == "repaired"
    assert routed["case-ptx-phrase"].heatmap[1, 1] > routed["case-ptx-phrase"].heatmap[0, 0]
    assert routed["case-ptx-phrase"].heatmap[1, 1] > routed["case-ptx-phrase"].heatmap[1, 0]
    assert routed_stats["gate_passed_categories"]["Pneumothorax"] == 1
