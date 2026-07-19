from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from anaprior.eval.report_afloc_anchor_phase0 import build_anchor_phase0_report


def write_phase0_inputs(
    tmp_path: Path,
    *,
    ci_low: float | None,
    zero_variance_count: int = 0,
    absolute_cnr: float = 0.6,
) -> tuple[Path, Path]:
    anchor_summary = tmp_path / "anchor_eval_summary.json"
    anchor_summary.write_text(
        json.dumps(
            {
                "status": "ok",
                "num_cases": 2,
                "num_hmaps": 2,
                "zero_variance_count": zero_variance_count,
                "mean_anchor_confidence": 0.75,
            }
        ),
        encoding="utf-8",
    )
    metrics_dir = tmp_path / "metrics"
    anchor_dir = metrics_dir / "afloc_anchor"
    anchor_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"case_id": "a", "split": "test", "category": "Edema", "cnr": absolute_cnr},
            {"case_id": "b", "split": "test", "category": "Pneumonia", "cnr": absolute_cnr},
        ]
    ).to_csv(anchor_dir / "per_case_metric.csv", index=False)
    rows = []
    if ci_low is not None:
        rows.append(
            {
                "comparison": "afloc_anchor_vs_baseline",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.01,
                "ci_low": ci_low,
                "ci_high": 0.03,
                "n_cases": 2,
            }
        )
    pd.DataFrame(rows).to_csv(metrics_dir / "bootstrap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison": "afloc_anchor_vs_baseline",
                "split": "test",
                "category": "Edema",
                "metric": "cnr",
                "mean_delta": 0.02,
                "ci_low": -0.01,
                "ci_high": 0.05,
                "n": 1,
            }
        ]
    ).to_csv(metrics_dir / "per_class_bootstrap_summary.csv", index=False)
    metrics_summary = metrics_dir / "learned_repair_metrics_summary.json"
    metrics_summary.write_text(
        json.dumps(
            {
                "status": "ok",
                "num_eval_rows_after_category_filter": 2,
                "num_eval_rows_after_hmap_filter": 2,
                "methods": ["baseline", "phrase_anatomy_dcem", "afloc_anchor"],
                "outputs": {
                    "bootstrap_summary": str(metrics_dir / "bootstrap_summary.csv"),
                    "per_class_bootstrap_summary": str(metrics_dir / "per_class_bootstrap_summary.csv"),
                },
                "method_outputs": {
                    "afloc_anchor": {
                        "per_case_metric_csv": str(anchor_dir / "per_case_metric.csv")
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return anchor_summary, metrics_summary


def test_phase0_report_marks_anchor_ready(tmp_path: Path) -> None:
    anchor_summary, metrics_summary = write_phase0_inputs(tmp_path, ci_low=-0.004)

    decision = build_anchor_phase0_report(
        anchor_summary_path=anchor_summary,
        metrics_summary_path=metrics_summary,
        outdir=tmp_path / "report",
    )

    assert decision["status"] == "anchor_ready_for_bounded_refinement"
    assert decision["coverage_complete"] is True
    assert decision["macro_cnr_delta_vs_baseline"] == 0.01
    assert decision["macro_cnr_ci_low_vs_baseline"] == -0.004
    assert decision["anchor_absolute_pooled_cnr"] == 0.6
    assert (tmp_path / "report" / "anchor_phase0_decision.json").exists()
    assert (tmp_path / "report" / "anchor_phase0_report.md").exists()


def test_phase0_report_rejects_material_macro_harm(tmp_path: Path) -> None:
    anchor_summary, metrics_summary = write_phase0_inputs(tmp_path, ci_low=-0.05)

    decision = build_anchor_phase0_report(
        anchor_summary_path=anchor_summary,
        metrics_summary_path=metrics_summary,
        outdir=tmp_path / "report",
    )

    assert decision["status"] == "anchor_not_ready_improve_construction"
    assert "macro_cnr_ci_low_below_harm_floor" in decision["reasons"]


def test_phase0_report_marks_missing_bootstrap_as_incomplete(tmp_path: Path) -> None:
    anchor_summary, metrics_summary = write_phase0_inputs(tmp_path, ci_low=None)

    decision = build_anchor_phase0_report(
        anchor_summary_path=anchor_summary,
        metrics_summary_path=metrics_summary,
        outdir=tmp_path / "report",
    )

    assert decision["status"] == "incomplete_evidence"
    assert "missing_anchor_vs_baseline_macro_cnr" in decision["reasons"]
