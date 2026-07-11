import json
from pathlib import Path

import pandas as pd

from anaprior.eval.report_dcem_v1_paper_results import build_dcem_v1_paper_results


def write_stage_c_metrics(metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "macro_scope": "pooled_all",
                "metric": "cnr",
                "mean_delta": -0.12,
                "ci_low": -0.14,
                "ci_high": -0.10,
                "n": 433,
            },
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": -0.10,
                "ci_low": -0.18,
                "ci_high": -0.02,
                "n_categories": 8,
                "n_cases": 900,
            },
            {
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "macro_scope": "pooled_all",
                "metric": "cnr",
                "mean_delta": 0.012,
                "ci_low": 0.008,
                "ci_high": 0.018,
                "n": 433,
            },
            {
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.018,
                "ci_low": 0.0,
                "ci_high": 0.054,
                "n_categories": 8,
                "n_cases": 900,
            },
            {
                "comparison": "validation_gated_dcem_vs_learned_selective",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.136,
                "ci_low": 0.048,
                "ci_high": 0.205,
                "n_categories": 8,
                "n_cases": 900,
            },
            {
                "comparison": "validation_gated_dcem_vs_candidate_shuffled",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.184,
                "ci_low": 0.126,
                "ci_high": 0.241,
                "n_categories": 8,
                "n_cases": 900,
            },
        ]
    ).to_csv(metrics_dir / "bootstrap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "category": "Pleural Effusion",
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": -0.07,
                "ci_low": -0.12,
                "ci_high": -0.02,
                "n": 72,
            },
            {
                "category": "Pleural Effusion",
                "comparison": "disease_gated_learned_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.144,
                "ci_low": 0.10,
                "ci_high": 0.19,
                "n": 72,
            },
            {
                "category": "Pleural Effusion",
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.144,
                "ci_low": 0.10,
                "ci_high": 0.19,
                "n": 72,
            },
            {
                "category": "Pneumothorax",
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": -0.275,
                "ci_low": -0.31,
                "ci_high": -0.24,
                "n": 175,
            },
            {
                "category": "Pneumothorax",
                "comparison": "disease_gated_learned_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.026,
                "ci_low": 0.005,
                "ci_high": 0.048,
                "n": 175,
            },
            {
                "category": "Pneumothorax",
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "n": 175,
            },
        ]
    ).to_csv(metrics_dir / "per_class_bootstrap_summary.csv", index=False)
    (metrics_dir / "validation_gate_decision.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "method": "validation_gated_dcem",
                "effect_floor": 0.02,
                "ci_low_floor": 0.0,
                "decisions": {
                    "Pleural Effusion": {
                        "decision": "pass",
                        "reason": "validation_effect_and_ci_met",
                        "mean_delta": 0.15,
                        "ci_low": 0.05,
                        "n": 24,
                    },
                    "Pneumothorax": {
                        "decision": "bypass",
                        "reason": "validation_effect_or_ci_not_met",
                        "mean_delta": 0.04,
                        "ci_low": -0.001,
                        "n": 70,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def write_stage_e_gap_outputs(stage_e_dir: Path) -> None:
    stage_e_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "split": "test",
                "category": "Pleural Effusion",
                "n_cases": 72,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.236,
                "top3_hit_rate": 0.361,
                "learned_oracle_spearman": 0.254,
                "mean_oracle_top_overlap": 0.320,
                "mean_learned_top_oracle_overlap": 0.132,
                "failure_modes": json.dumps(
                    {"partial_region_mismatch": 31, "weak_oracle_region_signal": 16}
                ),
            },
            {
                "split": "test",
                "category": "Pneumothorax",
                "n_cases": 175,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.023,
                "top3_hit_rate": 0.166,
                "learned_oracle_spearman": -0.152,
                "mean_oracle_top_overlap": 0.427,
                "mean_learned_top_oracle_overlap": 0.043,
                "failure_modes": json.dumps(
                    {"partial_region_mismatch": 117, "oracle_signal_learned_miss": 49}
                ),
            },
        ]
    ).to_csv(stage_e_dir / "per_class_gap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "category": "Pleural Effusion",
                "diagnosis": "learned_gap_small",
                "recommended_next_step": "keep_as_supported_learned_repair_candidate",
            },
            {
                "category": "Pneumothorax",
                "diagnosis": "evidence_misdirected",
                "recommended_next_step": "inspect_label_bias_feature_bias",
            },
        ]
    ).to_csv(stage_e_dir / "gap_diagnosis.csv", index=False)


def test_build_dcem_v1_paper_results_writes_main_tables_and_report(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "stage_c" / "learned_repair_metrics"
    stage_e_dir = tmp_path / "stage_e"
    outdir = tmp_path / "paper_results"
    write_stage_c_metrics(metrics_dir)
    write_stage_e_gap_outputs(stage_e_dir)

    result = build_dcem_v1_paper_results(
        stage_c_metrics_dir=metrics_dir,
        stage_e_dir=stage_e_dir,
        outdir=outdir,
    )

    assert result["status"] == "ok"
    macro = pd.read_csv(outdir / "main_macro_cnr_table.csv")
    assert "pooled_mean_delta" in macro.columns
    assert (
        macro.loc[
            macro["comparison"] == "validation_gated_dcem_vs_baseline",
            "pooled_mean_delta",
        ].iloc[0]
        == 0.012
    )

    per_class = pd.read_csv(outdir / "main_per_class_cnr_table.csv")
    effusion = per_class[per_class["category"] == "Pleural Effusion"].iloc[0]
    assert effusion["gate_decision"] == "pass"
    assert effusion["validation_gated_cnr_delta"] == 0.144
    pneumothorax = per_class[per_class["category"] == "Pneumothorax"].iloc[0]
    assert pneumothorax["gate_decision"] == "bypass"
    assert pneumothorax["disease_gated_cnr_delta"] == 0.026

    gate_gap = pd.read_csv(outdir / "gate_and_gap_table.csv")
    assert "dominant_failure_mode" in gate_gap.columns
    assert (
        gate_gap.loc[
            gate_gap["category"] == "Pneumothorax",
            "dominant_failure_mode",
        ].iloc[0]
        == "partial_region_mismatch"
    )
    assert (
        gate_gap.loc[
            gate_gap["category"] == "Pleural Effusion",
            "diagnosis",
        ].iloc[0]
        == "learned_gap_small"
    )

    markdown = (outdir / "dcem_v1_results.md").read_text(encoding="utf-8")
    assert "DCEM-v1 Paper Results" in markdown
    assert "validation_gated_dcem_vs_baseline" in markdown
    assert "coverage fixed" in markdown
    assert "Pleural Effusion" in markdown


def write_stage_c_metrics_v2b(metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "comparison": "disease_pooled_learned_vs_baseline",
                "split": "test",
                "macro_scope": "pooled_all",
                "metric": "cnr",
                "mean_delta": -0.005205,
                "ci_low": -0.024367,
                "ci_high": 0.014056,
                "n": 833,
            },
            {
                "comparison": "disease_pooled_learned_vs_baseline",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": -0.028737,
                "ci_low": -0.110687,
                "ci_high": 0.058099,
                "n_categories": 8,
                "n_cases": 833,
            },
            {
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "macro_scope": "pooled_all",
                "metric": "cnr",
                "mean_delta": 0.050093,
                "ci_low": 0.040792,
                "ci_high": 0.060120,
                "n": 833,
            },
            {
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.040038,
                "ci_low": 0.0,
                "ci_high": 0.082685,
                "n_categories": 8,
                "n_cases": 833,
            },
            {
                "comparison": "validation_gated_dcem_vs_disease_pooled_learned",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.068775,
                "ci_low": 0.012,
                "ci_high": 0.121,
                "n_categories": 8,
                "n_cases": 833,
            },
            {
                "comparison": "validation_gated_dcem_vs_learned_selective",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.157909,
                "ci_low": 0.070116,
                "ci_high": 0.221100,
                "n_categories": 8,
                "n_cases": 833,
            },
            {
                "comparison": "validation_gated_dcem_vs_candidate_shuffled",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.207707,
                "ci_low": 0.152435,
                "ci_high": 0.259657,
                "n_categories": 8,
                "n_cases": 833,
            },
        ]
    ).to_csv(metrics_dir / "bootstrap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "category": "Cardiomegaly",
                "comparison": "disease_pooled_learned_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.122693,
                "ci_low": 0.097048,
                "ci_high": 0.151451,
                "n": 237,
            },
            {
                "category": "Cardiomegaly",
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.122693,
                "ci_low": 0.097279,
                "ci_high": 0.148314,
                "n": 237,
            },
            {
                "category": "Edema",
                "comparison": "disease_pooled_learned_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.050910,
                "ci_low": 0.002089,
                "ci_high": 0.096377,
                "n": 41,
            },
            {
                "category": "Edema",
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.050910,
                "ci_low": 0.003278,
                "ci_high": 0.096672,
                "n": 41,
            },
            {
                "category": "Pleural Effusion",
                "comparison": "disease_pooled_learned_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.146699,
                "ci_low": 0.103019,
                "ci_high": 0.191839,
                "n": 72,
            },
            {
                "category": "Pleural Effusion",
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.146699,
                "ci_low": 0.103986,
                "ci_high": 0.192144,
                "n": 72,
            },
            {
                "category": "Pneumonia",
                "comparison": "disease_pooled_learned_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": -0.243282,
                "ci_low": -0.298487,
                "ci_high": -0.183012,
                "n": 124,
            },
            {
                "category": "Pneumonia",
                "comparison": "validation_gated_dcem_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "n": 124,
            },
        ]
    ).to_csv(metrics_dir / "per_class_bootstrap_summary.csv", index=False)
    (metrics_dir / "validation_gate_decision.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "method": "validation_gated_dcem",
                "source_method": "disease_pooled_learned",
                "decisions": {
                    "Cardiomegaly": {"decision": "pass", "reason": "validation_effect_and_ci_met"},
                    "Edema": {"decision": "pass", "reason": "validation_effect_and_ci_met"},
                    "Pleural Effusion": {
                        "decision": "pass",
                        "reason": "validation_effect_and_ci_met",
                    },
                    "Pneumonia": {
                        "decision": "bypass",
                        "reason": "validation_effect_or_ci_not_met",
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def write_stage_e_gap_outputs_v2b(stage_e_dir: Path) -> None:
    stage_e_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "split": "test",
                "category": "Cardiomegaly",
                "n_cases": 237,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.063291,
                "top3_hit_rate": 0.261603,
                "learned_oracle_spearman": 0.237997,
                "mean_oracle_top_overlap": 0.247736,
                "failure_modes": json.dumps({"useful_oracle_signal": 237}),
            },
            {
                "split": "test",
                "category": "Edema",
                "n_cases": 41,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.024390,
                "top3_hit_rate": 0.292683,
                "learned_oracle_spearman": -0.075094,
                "mean_oracle_top_overlap": 0.363736,
                "failure_modes": json.dumps({"diffuse_signal_partial": 41}),
            },
            {
                "split": "test",
                "category": "Pleural Effusion",
                "n_cases": 72,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.166667,
                "top3_hit_rate": 0.388889,
                "learned_oracle_spearman": 0.257154,
                "mean_oracle_top_overlap": 0.319747,
                "failure_modes": json.dumps({"useful_oracle_signal": 72}),
            },
            {
                "split": "test",
                "category": "Pneumonia",
                "n_cases": 124,
                "learned_evidence_case_rate": 1.0,
                "top1_hit_rate": 0.032258,
                "top3_hit_rate": 0.282258,
                "learned_oracle_spearman": 0.039663,
                "mean_oracle_top_overlap": 0.352467,
                "failure_modes": json.dumps({"learned_top_region_is_far_from_oracle": 124}),
            },
        ]
    ).to_csv(stage_e_dir / "per_class_gap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "category": "Cardiomegaly",
                "diagnosis": "learned_gap_small",
                "recommended_next_step": "keep_as_supported_learned_repair_candidate",
            },
            {
                "category": "Edema",
                "diagnosis": "learned_gap_small",
                "recommended_next_step": "keep_as_supported_learned_repair_candidate",
            },
            {
                "category": "Pleural Effusion",
                "diagnosis": "learned_gap_small",
                "recommended_next_step": "keep_as_supported_learned_repair_candidate",
            },
            {
                "category": "Pneumonia",
                "diagnosis": "evidence_misdirected",
                "recommended_next_step": "inspect_label_bias_feature_bias",
            },
        ]
    ).to_csv(stage_e_dir / "gap_diagnosis.csv", index=False)


def test_build_dcem_v2b_result_pack_includes_disease_pooled_and_uses_v2b_names(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "stage_c" / "learned_repair_metrics"
    stage_e_dir = tmp_path / "stage_e"
    outdir = tmp_path / "dcem_v2b_results"
    write_stage_c_metrics_v2b(metrics_dir)
    write_stage_e_gap_outputs_v2b(stage_e_dir)

    result = build_dcem_v1_paper_results(
        stage_c_metrics_dir=metrics_dir,
        stage_e_dir=stage_e_dir,
        outdir=outdir,
        method_name="DCEM-v2B",
        output_prefix="dcem_v2b",
    )

    assert result["status"] == "ok"
    assert result["method_name"] == "DCEM-v2B"
    assert (outdir / "dcem_v2b_results.md").exists()
    assert (outdir / "dcem_v2b_result_pack.json").exists()

    claims = json.loads((outdir / "claims.json").read_text(encoding="utf-8"))
    assert claims["status"] == "dcem_v2b_frozen"
    assert claims["disease_pooled_vs_baseline"]["pooled_mean_delta"] == -0.005205
    assert set(claims["passed_categories"]) == {
        "Cardiomegaly",
        "Edema",
        "Pleural Effusion",
    }
    assert claims["coverage_fixed"] is True

    per_class = pd.read_csv(outdir / "main_per_class_cnr_table.csv")
    cardiomegaly = per_class[per_class["category"] == "Cardiomegaly"].iloc[0]
    assert cardiomegaly["disease_pooled_cnr_delta"] == 0.122693
    pneumonia = per_class[per_class["category"] == "Pneumonia"].iloc[0]
    assert pneumonia["gate_decision"] == "bypass"

    markdown = (outdir / "dcem_v2b_results.md").read_text(encoding="utf-8")
    assert "DCEM-v2B Paper Results" in markdown
    assert "disease_pooled_learned_vs_baseline" in markdown
    assert "Cardiomegaly" in markdown


def test_build_dcem_v3_result_pack_includes_phrase_anatomy_and_named_gate(
    tmp_path: Path,
) -> None:
    metrics_dir = tmp_path / "stage_c" / "learned_repair_metrics"
    stage_e_dir = tmp_path / "stage_e"
    outdir = tmp_path / "dcem_v3_results"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stage_e_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "comparison": "phrase_anatomy_dcem_vs_baseline",
                "split": "test",
                "macro_scope": "pooled_all",
                "metric": "cnr",
                "mean_delta": 0.03,
                "ci_low": 0.01,
                "ci_high": 0.05,
                "n": 100,
            },
            {
                "comparison": "validation_gated_dcem_v3_vs_baseline",
                "split": "test",
                "macro_scope": "pooled_all",
                "metric": "cnr",
                "mean_delta": 0.04,
                "ci_low": 0.02,
                "ci_high": 0.06,
                "n": 100,
            },
            {
                "comparison": "validation_gated_dcem_v3_vs_phrase_anatomy_dcem",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.01,
                "ci_low": 0.0,
                "ci_high": 0.02,
                "n_categories": 8,
                "n_cases": 100,
            },
        ]
    ).to_csv(metrics_dir / "bootstrap_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "category": "Pneumothorax",
                "comparison": "phrase_anatomy_dcem_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.07,
                "ci_low": 0.02,
                "ci_high": 0.11,
                "n": 40,
            },
            {
                "category": "Pneumothorax",
                "comparison": "validation_gated_dcem_v3_vs_baseline",
                "split": "test",
                "metric": "cnr",
                "mean_delta": 0.07,
                "ci_low": 0.02,
                "ci_high": 0.11,
                "n": 40,
            },
        ]
    ).to_csv(metrics_dir / "per_class_bootstrap_summary.csv", index=False)
    (metrics_dir / "validation_gate_decision.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "method": "validation_gated_dcem_v3",
                "source_method": "phrase_anatomy_dcem",
                "decisions": {
                    "Pneumothorax": {
                        "decision": "pass",
                        "reason": "validation_effect_and_ci_met",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "category": "Pneumothorax",
                "n_cases": 40,
                "learned_evidence_case_rate": 1.0,
            }
        ]
    ).to_csv(stage_e_dir / "per_class_gap_summary.csv", index=False)

    build_dcem_v1_paper_results(
        stage_c_metrics_dir=metrics_dir,
        stage_e_dir=stage_e_dir,
        outdir=outdir,
        method_name="DCEM-v3",
        output_prefix="dcem_v3",
    )

    macro = pd.read_csv(outdir / "main_macro_cnr_table.csv")
    assert "phrase_anatomy_dcem_vs_baseline" in set(macro["comparison"])
    assert "validation_gated_dcem_v3_vs_baseline" in set(macro["comparison"])

    per_class = pd.read_csv(outdir / "main_per_class_cnr_table.csv")
    pneumothorax = per_class[per_class["category"] == "Pneumothorax"].iloc[0]
    assert pneumothorax["phrase_anatomy_cnr_delta"] == 0.07
    assert pneumothorax["validation_gated_v3_cnr_delta"] == 0.07
    assert pneumothorax["gate_decision"] == "pass"

    markdown = (outdir / "dcem_v3_results.md").read_text(encoding="utf-8")
    assert "DCEM-v3 Paper Results" in markdown
    assert "validation_gated_dcem_v3_vs_baseline" in markdown
