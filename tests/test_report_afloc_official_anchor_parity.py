from __future__ import annotations

import numpy as np

from anaprior.eval.report_afloc_official_anchor_parity import compare_heatmap_sets


def _generated(hmap: np.ndarray) -> dict:
    return {
        "case-1": {
            "hmap_key": "/images/a.jpgopacity",
            "path": "/images/a.jpg",
            "label_text": "opacity",
            "hmap": hmap,
        }
    }


def test_exact_official_baseline_match_passes() -> None:
    hmap = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = compare_heatmap_sets(
        _generated(hmap),
        {"/images/a.jpgopacity": {"hmap": hmap.copy()}},
    )

    assert result["parity_passed"] is True
    assert result["matched_cases"] == 1
    assert result["coverage"] == 1.0
    assert result["mean_pearson_r"] == 1.0
    assert result["mean_mae"] == 0.0


def test_exact_small_subset_does_not_satisfy_formal_case_floor() -> None:
    hmap = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = compare_heatmap_sets(
        _generated(hmap),
        {"/images/a.jpgopacity": {"hmap": hmap.copy()}},
        min_matched_cases=2,
    )

    assert result["parity_passed"] is False
    assert result["matched_cases"] == 1


def test_missing_reference_key_fails_coverage() -> None:
    result = compare_heatmap_sets(_generated(np.eye(3, dtype=np.float32)), {})

    assert result["parity_passed"] is False
    assert result["coverage"] == 0.0
    assert result["missing_reference_keys"] == ["/images/a.jpgopacity"]


def test_shape_or_value_mismatch_fails_parity() -> None:
    generated = _generated(np.arange(16, dtype=np.float32).reshape(4, 4))
    shape_result = compare_heatmap_sets(
        generated,
        {"/images/a.jpgopacity": {"hmap": np.ones((3, 3), dtype=np.float32)}},
    )
    value_result = compare_heatmap_sets(
        generated,
        {"/images/a.jpgopacity": {"hmap": np.flipud(np.arange(16).reshape(4, 4)).copy()}},
    )

    assert shape_result["parity_passed"] is False
    assert shape_result["shape_mismatch_count"] == 1
    assert value_result["parity_passed"] is False
    assert value_result["mean_pearson_r"] < 0.999
