import json
from pathlib import Path

import pandas as pd

from anaprior.eval.report_stage_c_results import build_stage_c_markdown_report


def write_stage_c_outputs(metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "learned_repair_decision.json").write_text(
        json.dumps(
            {
                "verdict": "keep_learned_selective_repair",
                "reason": "candidate_gain_specificity_and_macro_safety_met",
                "candidate_vs_baseline_cnr_delta": 0.123456,
                "candidate_vs_baseline_cnr_ci_low": 0.08,
                "specificity_vs_shuffled_cnr_delta": 0.2,
                "specificity_vs_shuffled_cnr_ci_low": 0.1,
                "macro_all_cnr_delta": 0.03,
                "macro_all_cnr_ci_low": 0.0,
                "per_class_decisions": {
                    "Pleural Effusion": {
                        "expected_role": "primary_evidence",
                        "verdict": "strong_pass",
                        "reason": "effect_floor_ci_and_specificity_met",
                    },
                    "Pneumothorax": {
                        "expected_role": "sparse_class_partial_success_allowed",
                        "verdict": "directional_partial_success",
                        "reason": "positive_direction_and_specificity_met_but_strong_floor_not_met",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "macro_scope": "macro_candidate",
                "metric": "cnr",
                "mean_delta": 0.123456,
                "ci_low": 0.08,
                "ci_high": 0.18,
            },
            {
                "comparison": "learned_selective_vs_candidate_shuffled",
                "split": "test",
                "macro_scope": "macro_candidate",
                "metric": "cnr",
                "mean_delta": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
            },
        ]
    ).to_csv(metrics_dir / "bootstrap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "category": "Pneumothorax",
                "metric": "cnr",
                "mean_delta": 0.5,
                "ci_low": 0.4,
                "ci_high": 0.6,
                "n": 100,
            }
        ]
    ).to_csv(metrics_dir / "per_class_bootstrap_summary.csv", index=False)
    (metrics_dir / "validation_gate_decision.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "method": "validation_gated_dcem",
                "selection_split": "val",
                "effect_floor": 0.02,
                "ci_low_floor": 0.0,
                "decisions": {
                    "Pleural Effusion": {
                        "decision": "pass",
                        "reason": "validation_effect_and_ci_met",
                        "mean_delta": 0.05,
                        "ci_low": 0.01,
                    },
                    "Consolidation": {
                        "decision": "bypass",
                        "reason": "validation_effect_or_ci_not_met",
                        "mean_delta": -0.02,
                        "ci_low": -0.08,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_stage_c_markdown_report_summarizes_decision_and_tables(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    output_md = tmp_path / "stage_c_report.md"
    write_stage_c_outputs(metrics_dir)

    report = build_stage_c_markdown_report(metrics_dir=metrics_dir, output_md=output_md)

    assert output_md.exists()
    assert "keep_learned_selective_repair" in report
    assert "learned_selective_vs_baseline" in report
    assert "learned_selective_vs_candidate_shuffled" in report
    assert "Pneumothorax" in report
    assert "建议" in report
    assert "Per-Class 判定" in report
    assert "directional_partial_success" in report
    assert "validation_gated_dcem" in report
    assert "bypass" in report
