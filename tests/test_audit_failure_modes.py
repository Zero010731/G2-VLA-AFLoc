import json
from pathlib import Path

import pandas as pd

from anaprior.eval.audit_failure_modes import (
    TARGET_FAILURE_CATEGORIES,
    build_failure_audit,
    main,
    phrase_subtype,
)


def _case_gap() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case-pna",
                "category": "Pneumonia",
                "phrase": "right lower lobe pneumonia with airspace opacity",
                "learned_top_region": "hilar_mediastinal",
                "oracle_top_region": "right_lower_lung",
                "top1_hit": False,
                "top3_hit": True,
                "learned_top_score": 0.91,
                "oracle_top_overlap": 0.55,
                "learned_top_oracle_overlap": 0.05,
                "delta_cnr": -0.31,
                "delta_iou": -0.04,
                "delta_dice": -0.05,
                "failure_mode": "oracle_signal_learned_miss",
            },
            {
                "case_id": "case-cons",
                "category": "Consolidation",
                "phrase": "dense retrocardiac consolidation",
                "learned_top_region": "cardiac_silhouette",
                "oracle_top_region": "left_lower_lung",
                "top1_hit": False,
                "top3_hit": False,
                "learned_top_score": 0.88,
                "oracle_top_overlap": 0.49,
                "learned_top_oracle_overlap": 0.08,
                "delta_cnr": -0.12,
                "delta_iou": -0.02,
                "delta_dice": -0.03,
                "failure_mode": "oracle_signal_learned_miss",
            },
            {
                "case_id": "case-opacity",
                "category": "Lung Opacity",
                "phrase": "left basilar opacity",
                "learned_top_region": "left_lower_lung",
                "oracle_top_region": "left_lower_lung",
                "top1_hit": True,
                "top3_hit": True,
                "learned_top_score": 0.93,
                "oracle_top_overlap": 0.60,
                "learned_top_oracle_overlap": 0.60,
                "delta_cnr": 0.06,
                "delta_iou": 0.01,
                "delta_dice": 0.01,
                "failure_mode": "learned_matches_and_repair_helps",
            },
            {
                "case_id": "case-eff",
                "category": "Pleural Effusion",
                "phrase": "small pleural effusion",
                "learned_top_region": "pleural_space_costophrenic",
                "oracle_top_region": "pleural_space_costophrenic",
                "top1_hit": True,
                "top3_hit": True,
                "learned_top_score": 0.95,
                "oracle_top_overlap": 0.70,
                "learned_top_oracle_overlap": 0.70,
                "delta_cnr": 0.20,
                "delta_iou": 0.03,
                "delta_dice": 0.04,
                "failure_mode": "learned_matches_and_repair_helps",
            },
        ]
    )


def _region_gap() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case-pna",
                "category": "Pneumonia",
                "region": "hilar_mediastinal",
                "learned_score": 0.91,
                "oracle_gt_overlap": 0.05,
                "is_learned_top_region": True,
                "is_oracle_top_region": False,
            },
            {
                "case_id": "case-pna",
                "category": "Pneumonia",
                "region": "right_lower_lung",
                "learned_score": 0.42,
                "oracle_gt_overlap": 0.55,
                "is_learned_top_region": False,
                "is_oracle_top_region": True,
            },
            {
                "case_id": "case-cons",
                "category": "Consolidation",
                "region": "cardiac_silhouette",
                "learned_score": 0.88,
                "oracle_gt_overlap": 0.08,
                "is_learned_top_region": True,
                "is_oracle_top_region": False,
            },
            {
                "case_id": "case-cons",
                "category": "Consolidation",
                "region": "left_lower_lung",
                "learned_score": 0.45,
                "oracle_gt_overlap": 0.49,
                "is_learned_top_region": False,
                "is_oracle_top_region": True,
            },
            {
                "case_id": "case-opacity",
                "category": "Lung Opacity",
                "region": "left_lower_lung",
                "learned_score": 0.93,
                "oracle_gt_overlap": 0.60,
                "is_learned_top_region": True,
                "is_oracle_top_region": True,
            },
        ]
    )


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"case_id": "case-pna", "method": "baseline", "cnr": 1.0, "iou": 0.20, "dice": 0.30},
            {"case_id": "case-pna", "method": "phrase_anatomy_dcem", "cnr": 0.69, "iou": 0.16, "dice": 0.25},
            {"case_id": "case-cons", "method": "baseline", "cnr": 1.0, "iou": 0.21, "dice": 0.31},
            {"case_id": "case-cons", "method": "phrase_anatomy_dcem", "cnr": 0.88, "iou": 0.19, "dice": 0.28},
            {"case_id": "case-opacity", "method": "baseline", "cnr": 1.0, "iou": 0.22, "dice": 0.33},
            {"case_id": "case-opacity", "method": "phrase_anatomy_dcem", "cnr": 1.06, "iou": 0.23, "dice": 0.34},
        ]
    )


def test_phrase_subtype_groups_pneumonia_consolidation_and_opacity_terms() -> None:
    assert phrase_subtype("right lower lobe pneumonia with airspace opacity") == "pneumonia_airspace"
    assert phrase_subtype("dense retrocardiac consolidation") == "consolidation_retrocardiac"
    assert phrase_subtype("left basilar opacity") == "opacity_basilar"


def test_build_failure_audit_summarizes_target_classes_and_shortcuts() -> None:
    result = build_failure_audit(
        case_gap=_case_gap(),
        region_gap=_region_gap(),
        per_case_metrics=_metrics(),
        categories=TARGET_FAILURE_CATEGORIES,
        method="phrase_anatomy_dcem",
        baseline_method="baseline",
    )

    assert set(result.phrase_summary["category"]) == {"Pneumonia", "Consolidation", "Lung Opacity"}
    assert "Pleural Effusion" not in set(result.phrase_summary["category"])

    pna = result.shortcut_summary[result.shortcut_summary["category"] == "Pneumonia"].iloc[0]
    assert pna["cardiac_or_hilar_shortcut_rate"] == 1.0
    assert pna["mean_delta_cnr"] == -0.31

    confusion = result.region_confusion
    route = confusion[confusion["category"] == "Consolidation"].iloc[0]
    assert route["oracle_top_region"] == "left_lower_lung"
    assert route["learned_top_region"] == "cardiac_silhouette"
    assert route["is_hit"] is False

    assert "recommended_v4_action" in result.recommendations.columns
    assert "phrase subtype encoder" in result.report_markdown


def test_main_writes_failure_audit_outputs(tmp_path: Path) -> None:
    case_gap = tmp_path / "case_gap_table.csv"
    region_gap = tmp_path / "region_gap_table.csv"
    metrics = tmp_path / "all_per_case_metrics.csv"
    _case_gap().to_csv(case_gap, index=False)
    _region_gap().to_csv(region_gap, index=False)
    _metrics().to_csv(metrics, index=False)

    status = main(
        [
            "--case-gap-csv",
            str(case_gap),
            "--region-gap-csv",
            str(region_gap),
            "--metrics-csv",
            str(metrics),
            "--outdir",
            str(tmp_path / "out"),
            "--method",
            "phrase_anatomy_dcem",
            "--baseline-method",
            "baseline",
        ]
    )

    assert status == 0
    manifest = json.loads((tmp_path / "out" / "failure_audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["num_audited_cases"] == 3
    assert (tmp_path / "out" / "failure_phrase_summary.csv").exists()
    assert (tmp_path / "out" / "failure_region_confusion.csv").exists()
    assert (tmp_path / "out" / "failure_shortcut_summary.csv").exists()
    assert (tmp_path / "out" / "failure_case_examples.csv").exists()
    assert (tmp_path / "out" / "failure_v4_recommendations.csv").exists()
    assert "Stage F Failure Audit" in (tmp_path / "out" / "failure_audit_report.md").read_text(encoding="utf-8")
