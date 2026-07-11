import sys
import types

import numpy as np
import pandas as pd
import pytest

from anaprior.eval.eval_mscxr_learned_repair import LearnedRepairInput
from anaprior.eval.eval_mscxr_oracle_recoverability import (
    build_oracle_recoverability_hmaps,
    build_recoverability_profile,
    score_oracle_recoverability,
)


def test_build_oracle_recoverability_hmaps_uses_gt_region_overlap_as_scores() -> None:
    region_maps = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    inputs = [
        LearnedRepairInput(
            case_id="case-a",
            dicom_id="dicom-a",
            category="Pneumothorax",
            finding="Pneumothorax",
            heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            region_maps=region_maps,
            regions=["upper", "lower"],
        )
    ]
    gt_masks = {"case-a": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)}

    hmaps, stats, oracle_table = build_oracle_recoverability_hmaps(
        inputs=inputs,
        gt_masks_by_case=gt_masks,
        alpha=1.0,
        seed=0,
    )

    assert set(hmaps) == {"baseline", "oracle_repair", "oracle_shuffled", "oracle_uniform"}
    assert hmaps["oracle_repair"]["case-a"]["hmap"][0, 0] > hmaps["oracle_repair"]["case-a"]["hmap"][1, 0]
    assert stats["oracle_repair"]["repaired_categories"]["Pneumothorax"] == 1
    assert oracle_table.iloc[0]["region"] == "upper"
    assert oracle_table.iloc[0]["oracle_score"] == 1.0


def test_build_recoverability_profile_classifies_oracle_upper_bound() -> None:
    per_class = pd.DataFrame(
        [
            {
                "comparison": "oracle_repair_vs_baseline",
                "split": "test",
                "category": "Pleural Effusion",
                "metric": "cnr",
                "mean_delta": 0.08,
                "ci_low": 0.03,
                "ci_high": 0.12,
                "n": 70,
            },
            {
                "comparison": "oracle_repair_vs_baseline",
                "split": "test",
                "category": "Pleural Effusion",
                "metric": "iou",
                "mean_delta": 0.02,
                "ci_low": 0.01,
                "ci_high": 0.03,
                "n": 70,
            },
            {
                "comparison": "oracle_repair_vs_baseline",
                "split": "test",
                "category": "Pleural Effusion",
                "metric": "dice",
                "mean_delta": 0.03,
                "ci_low": 0.01,
                "ci_high": 0.04,
                "n": 70,
            },
            {
                "comparison": "oracle_repair_vs_baseline",
                "split": "test",
                "category": "Pneumothorax",
                "metric": "cnr",
                "mean_delta": 0.0,
                "ci_low": -0.02,
                "ci_high": 0.02,
                "n": 167,
            },
            {
                "comparison": "oracle_repair_vs_baseline",
                "split": "test",
                "category": "Pneumothorax",
                "metric": "iou",
                "mean_delta": 0.03,
                "ci_low": 0.01,
                "ci_high": 0.04,
                "n": 167,
            },
            {
                "comparison": "oracle_repair_vs_baseline",
                "split": "test",
                "category": "Pneumothorax",
                "metric": "dice",
                "mean_delta": 0.04,
                "ci_low": 0.02,
                "ci_high": 0.05,
                "n": 167,
            },
        ]
    )

    profile = build_recoverability_profile(per_class, effect_floor=0.02)

    effusion = profile[profile["category"] == "Pleural Effusion"].iloc[0]
    pneumothorax = profile[profile["category"] == "Pneumothorax"].iloc[0]
    assert effusion["verdict"] == "region_recoverable"
    assert pneumothorax["verdict"] == "overlap_recoverable_cnr_limited"


def test_score_oracle_recoverability_reports_dataset_key_mismatch(monkeypatch, tmp_path) -> None:
    hmaps_root = tmp_path / "hmaps"
    method_dir = hmaps_root / "baseline"
    method_dir.mkdir(parents=True)
    np.save(method_dir / "hmaps.npy", {"unmatched-case": {"hmap": np.zeros((2, 2), dtype=np.float32)}})

    def fake_load_data(dataset: str):
        return pd.DataFrame(
            {
                "path": ["a.jpg"],
                "label_text": ["Findings suggesting Pneumothorax."],
                "gtmasks": [np.ones((2, 2), dtype=np.float32)],
                "category": ["Pneumothorax"],
            }
        )

    fake_module = types.ModuleType("localization.datasets")
    fake_module.load_data = fake_load_data
    monkeypatch.setitem(sys.modules, "localization.datasets", fake_module)

    with pytest.raises(ValueError, match="No evaluation rows matched hmap keys.*baseline.*MS_CXR_CLS"):
        score_oracle_recoverability(
            hmaps_root=hmaps_root,
            outdir=tmp_path / "metrics",
            methods=["baseline"],
            dataset="MS_CXR_CLS",
        )
