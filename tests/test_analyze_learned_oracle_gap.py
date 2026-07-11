import sys
import types

import numpy as np
import pandas as pd

from anaprior.eval.analyze_learned_oracle_gap import (
    GapInput,
    build_gap_diagnosis_table,
    build_gap_tables,
    build_oracle_learned_confusion,
    compute_region_gt_overlaps,
    dataframe_to_markdown,
    load_mscxr_gt_masks_by_case,
)


def test_compute_region_gt_overlaps_measures_lesion_mass_per_region() -> None:
    region_maps = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    gt_mask = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)

    overlaps = compute_region_gt_overlaps(region_maps, gt_mask)

    assert np.allclose(overlaps, np.array([1.0, 0.0], dtype=np.float32))


def test_load_mscxr_gt_masks_uses_requested_dataset(monkeypatch) -> None:
    seen = []

    def fake_load_data(dataset: str):
        seen.append(dataset)
        return pd.DataFrame(
            {
                "path": ["a.jpg"],
                "label_text": ["raw phrase"],
                "gtmasks": [np.ones((2, 2), dtype=np.float32)],
            }
        )

    fake_module = types.ModuleType("localization.datasets")
    fake_module.load_data = fake_load_data
    monkeypatch.setitem(sys.modules, "localization.datasets", fake_module)

    masks = load_mscxr_gt_masks_by_case(dataset="MS_CXR")

    assert seen == ["MS_CXR"]
    assert "a.jpgraw phrase" in masks


def test_build_gap_tables_identifies_learned_miss_and_metric_gap() -> None:
    region_maps = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    inputs = [
        GapInput(
            case_id="case-hit",
            dicom_id="dicom-hit",
            category="Pleural Effusion",
            finding="Pleural Effusion",
            region_maps=region_maps,
            regions=["upper", "lower"],
        ),
        GapInput(
            case_id="case-miss",
            dicom_id="dicom-miss",
            category="Pneumothorax",
            finding="Pneumothorax",
            region_maps=region_maps,
            regions=["upper", "lower"],
        ),
    ]
    gt_masks = {
        "case-hit": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        "case-miss": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    }
    score_table = {
        ("dicom-hit", "pleural effusion"): {"upper": 0.8, "lower": 0.2},
        ("dicom-miss", "pneumothorax"): {"upper": 0.1, "lower": 0.9},
    }
    metrics = pd.DataFrame(
        [
            {"case_id": "case-hit", "method": "baseline", "cnr": 1.0, "iou": 0.20, "dice": 0.30},
            {"case_id": "case-hit", "method": "learned_selective", "cnr": 1.2, "iou": 0.22, "dice": 0.33},
            {"case_id": "case-miss", "method": "baseline", "cnr": 1.0, "iou": 0.20, "dice": 0.30},
            {"case_id": "case-miss", "method": "learned_selective", "cnr": 0.7, "iou": 0.18, "dice": 0.28},
        ]
    )

    region_table, case_table, summary = build_gap_tables(
        inputs=inputs,
        gt_masks_by_case=gt_masks,
        score_table=score_table,
        per_case_metrics=metrics,
        method="learned_selective",
        baseline_method="baseline",
    )

    assert set(region_table["evidence_source"]) == {"learned_vs_oracle_gt_overlap"}
    miss = case_table[case_table["case_id"] == "case-miss"].iloc[0]
    assert miss["learned_top_region"] == "lower"
    assert miss["oracle_top_region"] == "upper"
    assert bool(miss["top1_hit"]) is False
    assert miss["failure_mode"] == "oracle_signal_learned_miss"
    pneumothorax = summary[summary["category"] == "Pneumothorax"].iloc[0]
    assert pneumothorax["top1_hit_rate"] == 0.0
    assert pneumothorax["mean_delta_cnr"] == -0.3


def test_build_gap_tables_can_use_phrase_anatomy_evidence_mode() -> None:
    region_maps = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    inputs = [
        GapInput(
            case_id="case-ptx",
            dicom_id="dicom-ptx",
            category="Pneumothorax",
            finding="Pneumothorax",
            phrase="right apical pneumothorax",
            region_maps=region_maps,
            regions=["right_upper_lung", "cardiac_silhouette"],
        )
    ]
    metrics = pd.DataFrame(
        [
            {"case_id": "case-ptx", "method": "baseline", "cnr": 1.0, "iou": 0.2, "dice": 0.3},
            {"case_id": "case-ptx", "method": "phrase_anatomy_dcem", "cnr": 1.2, "iou": 0.22, "dice": 0.32},
        ]
    )

    region_table, case_table, _ = build_gap_tables(
        inputs=inputs,
        gt_masks_by_case={"case-ptx": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)},
        score_table={("dicom-ptx", "pneumothorax"): {"cardiac_silhouette": 0.95, "right_upper_lung": 0.35}},
        per_case_metrics=metrics,
        method="phrase_anatomy_dcem",
        baseline_method="baseline",
        evidence_mode="phrase_anatomy",
    )

    assert case_table.iloc[0]["learned_top_region"] == "right_upper_lung"
    assert set(region_table["evidence_source"]) == {"phrase_anatomy_vs_oracle_gt_overlap"}


def test_build_gap_tables_marks_missing_learned_evidence() -> None:
    region_maps = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    inputs = [
        GapInput(
            case_id="case-missing",
            dicom_id="dicom-missing",
            category="Atelectasis",
            finding="Atelectasis",
            region_maps=region_maps,
            regions=["upper", "lower"],
        )
    ]
    metrics = pd.DataFrame(
        [
            {"case_id": "case-missing", "method": "baseline", "cnr": 1.0, "iou": 0.2, "dice": 0.3},
            {"case_id": "case-missing", "method": "learned_selective", "cnr": 1.0, "iou": 0.2, "dice": 0.3},
        ]
    )

    region_table, case_table, summary = build_gap_tables(
        inputs=inputs,
        gt_masks_by_case={"case-missing": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)},
        score_table={},
        per_case_metrics=metrics,
    )

    assert bool(case_table.iloc[0]["has_learned_evidence"]) is False
    assert case_table.iloc[0]["learned_top_region"] == "missing_learned_evidence"
    assert np.isnan(case_table.iloc[0]["learned_top_score"])
    assert set(region_table["has_learned_evidence"]) == {False}
    assert region_table["learned_score"].isna().all()
    assert summary.iloc[0]["learned_evidence_case_rate"] == 0.0


def test_gap_diagnosis_distinguishes_missing_misdirected_coarse_and_calibration() -> None:
    oracle_profile = pd.DataFrame(
        [
            {
                "category": "Atelectasis",
                "n": 41,
                "oracle_delta_cnr": 0.17,
                "oracle_cnr_ci_low": 0.10,
                "oracle_delta_iou": 0.03,
                "oracle_delta_dice": 0.03,
                "verdict": "region_recoverable",
            },
            {
                "category": "Pneumothorax",
                "n": 167,
                "oracle_delta_cnr": 0.65,
                "oracle_cnr_ci_low": 0.59,
                "oracle_delta_iou": 0.04,
                "oracle_delta_dice": 0.06,
                "verdict": "region_recoverable",
            },
            {
                "category": "Pleural Effusion",
                "n": 70,
                "oracle_delta_cnr": 0.22,
                "oracle_cnr_ci_low": 0.16,
                "oracle_delta_iou": 0.06,
                "oracle_delta_dice": 0.07,
                "verdict": "region_recoverable",
            },
            {
                "category": "Consolidation",
                "n": 76,
                "oracle_delta_cnr": 0.16,
                "oracle_cnr_ci_low": 0.12,
                "oracle_delta_iou": 0.04,
                "oracle_delta_dice": 0.05,
                "verdict": "region_recoverable",
            },
        ]
    )
    gap_summary = pd.DataFrame(
        [
            {
                "split": "test",
                "category": "Pneumothorax",
                "n_cases": 167,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.03,
                "top3_hit_rate": 0.30,
                "learned_oracle_spearman": -0.1,
                "mean_oracle_top_overlap": 0.6,
                "mean_oracle_top_learned_score": 0.4,
                "mean_learned_top_score": 0.7,
                "mean_delta_cnr": -0.14,
            },
            {
                "split": "test",
                "category": "Pleural Effusion",
                "n_cases": 70,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.20,
                "top3_hit_rate": 0.87,
                "learned_oracle_spearman": 0.1,
                "mean_oracle_top_overlap": 0.45,
                "mean_oracle_top_learned_score": 0.3,
                "mean_learned_top_score": 0.5,
                "mean_delta_cnr": -0.01,
            },
            {
                "split": "test",
                "category": "Consolidation",
                "n_cases": 76,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.70,
                "top3_hit_rate": 0.92,
                "learned_oracle_spearman": 0.4,
                "mean_oracle_top_overlap": 0.6,
                "mean_oracle_top_learned_score": 0.05,
                "mean_learned_top_score": 0.08,
                "mean_delta_cnr": 0.0,
            },
        ]
    )

    diagnosis = build_gap_diagnosis_table(gap_summary, oracle_profile)

    by_category = {row["category"]: row["diagnosis"] for _, row in diagnosis.iterrows()}
    assert by_category["Atelectasis"] == "missing_learned_evidence"
    assert by_category["Pneumothorax"] == "evidence_misdirected"
    assert by_category["Pleural Effusion"] == "region_too_coarse_or_near_miss"
    assert by_category["Consolidation"] == "predictor_calibration_or_score_scale"


def test_dataframe_to_markdown_does_not_require_optional_tabulate() -> None:
    table = pd.DataFrame([{"category": "Pneumothorax", "mean_delta_cnr": -0.145526}])

    markdown = dataframe_to_markdown(table)

    assert "category" in markdown
    assert "mean_delta_cnr" in markdown
    assert "Pneumothorax" in markdown


def test_build_oracle_learned_confusion_counts_misroutes_by_category() -> None:
    case_table = pd.DataFrame(
        [
            {
                "category": "Pneumothorax",
                "oracle_top_region": "bilateral_lungs",
                "learned_top_region": "cardiac_silhouette",
                "delta_cnr": -0.4,
                "top1_hit": False,
            },
            {
                "category": "Pneumothorax",
                "oracle_top_region": "bilateral_lungs",
                "learned_top_region": "cardiac_silhouette",
                "delta_cnr": -0.2,
                "top1_hit": False,
            },
            {
                "category": "Pneumothorax",
                "oracle_top_region": "left_upper_lung",
                "learned_top_region": "left_upper_lung",
                "delta_cnr": 0.1,
                "top1_hit": True,
            },
            {
                "category": "Pleural Effusion",
                "oracle_top_region": "pleural_space_costophrenic",
                "learned_top_region": "hilar_mediastinal",
                "delta_cnr": -0.3,
                "top1_hit": False,
            },
        ]
    )

    confusion, top_misroutes = build_oracle_learned_confusion(case_table)

    row = confusion[
        (confusion["category"] == "Pneumothorax")
        & (confusion["oracle_top_region"] == "bilateral_lungs")
        & (confusion["learned_top_region"] == "cardiac_silhouette")
    ].iloc[0]
    assert row["count"] == 2
    assert row["mean_delta_cnr"] == -0.3
    assert bool(row["is_hit"]) is False
    top = top_misroutes[top_misroutes["category"] == "Pneumothorax"].iloc[0]
    assert top["oracle_top_region"] == "bilateral_lungs"
    assert top["learned_top_region"] == "cardiac_silhouette"
