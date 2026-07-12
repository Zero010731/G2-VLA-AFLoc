import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import pytest

from anaprior.eval.score_mscxr_learned_repair_metrics import (
    DEFAULT_COMPARISONS,
    DEFAULT_METHODS,
    build_validation_gate_decisions,
    build_validation_gated_per_case,
    build_paired_delta_table,
    comparison_specs_for_methods,
    coerce_eval_dataframe,
    evaluate_hmaps,
    filter_eval_dataframe_by_hmap_keys,
    filter_eval_dataframe_by_categories,
    metric_dataframe_from_category_values,
    make_learned_repair_decision,
    make_per_class_learned_repair_decisions,
    paired_bootstrap_summary,
    summarize_comparison,
    write_bootstrap_ci,
)


def metric_rows(method: str, cnr_a: float, cnr_b: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_id": "case-a",
                "category": "Pneumothorax",
                "split": "test",
                "method": method,
                "cnr": cnr_a,
                "iou": 0.2 + cnr_a / 100.0,
                "dice": 0.3 + cnr_a / 100.0,
            },
            {
                "case_id": "case-b",
                "category": "Pleural Effusion",
                "split": "test",
                "method": method,
                "cnr": cnr_b,
                "iou": 0.2 + cnr_b / 100.0,
                "dice": 0.3 + cnr_b / 100.0,
            },
        ]
    )


def test_build_validation_gate_decisions_uses_val_cnr_floor_and_ci() -> None:
    per_class = pd.DataFrame(
        [
            {
                "comparison": "disease_gated_learned_vs_baseline",
                "split": "val",
                "category": "Pleural Effusion",
                "metric": "cnr",
                "mean_delta": 0.05,
                "ci_low": 0.01,
                "ci_high": 0.09,
                "n": 12,
            },
            {
                "comparison": "disease_gated_learned_vs_baseline",
                "split": "val",
                "category": "Consolidation",
                "metric": "cnr",
                "mean_delta": 0.04,
                "ci_low": -0.02,
                "ci_high": 0.10,
                "n": 10,
            },
        ]
    )

    decisions = build_validation_gate_decisions(
        per_class,
        candidate_categories=["Pleural Effusion", "Consolidation", "Pneumothorax"],
        effect_floor=0.02,
        ci_low_floor=0.0,
    )

    assert decisions["Pleural Effusion"]["decision"] == "pass"
    assert decisions["Pleural Effusion"]["reason"] == "validation_effect_and_ci_met"
    assert decisions["Consolidation"]["decision"] == "bypass"
    assert decisions["Consolidation"]["reason"] == "validation_effect_or_ci_not_met"
    assert decisions["Pneumothorax"]["decision"] == "bypass"
    assert decisions["Pneumothorax"]["reason"] == "missing_validation_summary"


def test_build_validation_gated_per_case_uses_gated_for_pass_and_baseline_for_bypass() -> None:
    baseline = pd.DataFrame(
        [
            {
                "case_id": "case-eff",
                "category": "Pleural Effusion",
                "split": "test",
                "method": "baseline",
                "cnr": 1.0,
                "iou": 0.1,
                "dice": 0.2,
            },
            {
                "case_id": "case-con",
                "category": "Consolidation",
                "split": "test",
                "method": "baseline",
                "cnr": 2.0,
                "iou": 0.2,
                "dice": 0.3,
            },
        ]
    )
    gated = pd.DataFrame(
        [
            {
                "case_id": "case-eff",
                "category": "Pleural Effusion",
                "split": "test",
                "method": "disease_gated_learned",
                "cnr": 1.5,
                "iou": 0.15,
                "dice": 0.25,
            },
            {
                "case_id": "case-con",
                "category": "Consolidation",
                "split": "test",
                "method": "disease_gated_learned",
                "cnr": 1.2,
                "iou": 0.12,
                "dice": 0.22,
            },
        ]
    )
    decisions = {
        "Pleural Effusion": {"decision": "pass"},
        "Consolidation": {"decision": "bypass"},
    }

    validation_gated = build_validation_gated_per_case(
        baseline,
        gated,
        decisions,
        method_name="validation_gated_dcem",
    )

    by_case = validation_gated.set_index("case_id")
    assert by_case.loc["case-eff", "cnr"] == 1.5
    assert by_case.loc["case-eff", "method"] == "validation_gated_dcem"
    assert by_case.loc["case-eff", "validation_gate_decision"] == "pass"
    assert by_case.loc["case-con", "cnr"] == 2.0
    assert by_case.loc["case-con", "validation_gate_decision"] == "bypass"


def test_paired_delta_pairs_cases_and_bootstraps_three_metrics() -> None:
    baseline = metric_rows("baseline", 1.0, 1.0)
    learned = metric_rows("learned_selective", 1.4, 1.2)

    deltas = build_paired_delta_table(learned, baseline, split="test")
    summary = paired_bootstrap_summary(deltas, n_boot=20, seed=0)

    assert deltas["delta_cnr"].tolist() == pytest.approx([0.4, 0.2])
    assert set(summary["metric"]) == {"cnr", "iou", "dice"}
    assert summary.loc[summary["metric"] == "cnr", "mean_delta"].iloc[0] == 0.3


def test_coerce_eval_dataframe_accepts_afloc_dict_output_and_applies_max_cases() -> None:
    data = {
        "path": ["a.jpg", "b.jpg", "c.jpg"],
        "label_text": ["pa", "pb", "pc"],
        "category": ["Pneumothorax", "Pleural Effusion", "Edema"],
    }

    df = coerce_eval_dataframe(data, max_cases=2)

    assert df.shape[0] == 2
    assert df["path"].tolist() == ["a.jpg", "b.jpg"]


def test_filter_eval_dataframe_by_categories_drops_unsupported_categories_before_scoring() -> None:
    data = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg"],
            "label_text": ["pa", "pb", "pc"],
            "category": ["Pneumothorax", "Pleural Effusion", "Pneumonia"],
        }
    )

    filtered = filter_eval_dataframe_by_categories(
        data,
        ["Pneumothorax", "Pleural Effusion"],
    )

    assert filtered["category"].tolist() == ["Pneumothorax", "Pleural Effusion"]
    assert filtered.index.tolist() == [0, 1]


def test_filter_eval_dataframe_by_hmap_keys_keeps_only_scoreable_rows() -> None:
    data = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg"],
            "label_text": ["finding A", "finding B", "finding C"],
            "category": ["Pneumonia", "Pneumonia", "Pneumonia"],
        }
    )

    filtered = filter_eval_dataframe_by_hmap_keys(data, {"a.jpgfinding A", "c.jpgfinding C"})

    assert filtered["path"].tolist() == ["a.jpg", "c.jpg"]
    assert filtered.index.tolist() == [0, 1]


def test_filter_eval_dataframe_by_hmap_keys_accepts_unique_path_prefix_alias() -> None:
    data = pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg", "c.jpg"],
            "label_text": [
                "Findings suggesting Pneumonia.",
                "Findings suggesting Pneumonia.",
                "Findings suggesting Pneumonia.",
            ],
            "category": ["Pneumonia", "Pneumonia", "Pneumonia"],
        }
    )

    filtered = filter_eval_dataframe_by_hmap_keys(
        data,
        {
            "a.jpgPneumonia",
            "b.jpgPneumonia",
            "b.jpgPneumonia alternative",
        },
    )

    assert filtered["path"].tolist() == ["a.jpg"]


def test_metric_dataframe_from_category_values_keeps_numeric_columns() -> None:
    df = metric_dataframe_from_category_values(
        {
            "Pneumonia": [0.1, 0.2],
            "Pneumothorax": [0.3],
        }
    )

    assert df.shape == (3, 2)
    assert df["Pneumonia"].dropna().tolist() == [0.1, 0.2]
    assert df["Pneumothorax"].dropna().tolist() == [0.3]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes)


def test_write_bootstrap_ci_handles_string_task_names_on_pandas2(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "Pneumonia": [0.1, 0.2, 0.3],
            "Pneumothorax": [0.4, 0.5, 0.6],
        }
    )

    ci = write_bootstrap_ci(df, tmp_path, "iou", num_replicates=10, seed=0)

    assert (tmp_path / "iou_bootstrap_results.csv").exists()
    assert (tmp_path / "test_iou_summary_results.csv").exists()
    assert ci.iloc[-1]["name"] == "mean"
    assert ci[["lower", "mean", "upper"]].iloc[-1].notna().all()


def test_evaluate_hmaps_rejects_empty_eval_data_before_bootstrap(tmp_path) -> None:
    empty = pd.DataFrame(columns=["path", "label_text", "gtmasks", "category"])

    with pytest.raises(ValueError, match="No evaluation rows"):
        evaluate_hmaps(empty, {}, dataset="MS_CXR", save_dir=tmp_path, margin=True)


def test_summarize_comparison_reports_pooled_macro_and_per_class() -> None:
    baseline = metric_rows("baseline", 1.0, 1.0)
    learned = metric_rows("learned_selective", 1.4, 1.2)

    deltas, summary, per_class = summarize_comparison(
        method_per_case=learned,
        baseline_per_case=baseline,
        comparison="learned_selective_vs_baseline",
        candidate_categories={"Pneumothorax", "Pleural Effusion"},
        split="test",
        n_boot=20,
        seed=1,
    )

    assert deltas["comparison"].unique().tolist() == ["learned_selective_vs_baseline"]
    assert {"pooled_all", "macro_all", "macro_candidate", "macro_non_candidate"} <= set(summary["macro_scope"])
    assert set(per_class["category"]) == {"Pneumothorax", "Pleural Effusion"}


def test_default_methods_include_gated_dcem_without_removing_raw_learned_comparisons() -> None:
    comparison_names = {name for name, _, _ in DEFAULT_COMPARISONS}

    assert "learned_selective" in DEFAULT_METHODS
    assert "disease_gated_learned" in DEFAULT_METHODS
    assert "disease_pooled_learned" in DEFAULT_METHODS
    assert "phrase_anatomy_dcem" in DEFAULT_METHODS
    assert "dp_msa" in DEFAULT_METHODS
    assert "learned_selective_vs_baseline" in comparison_names
    assert "disease_gated_learned_vs_baseline" in comparison_names
    assert "disease_gated_learned_vs_learned_selective" in comparison_names
    assert "disease_gated_learned_vs_candidate_shuffled" in comparison_names
    assert "disease_pooled_learned_vs_baseline" in comparison_names
    assert "disease_pooled_learned_vs_disease_gated_learned" in comparison_names
    assert "phrase_anatomy_dcem_vs_baseline" in comparison_names
    assert "phrase_anatomy_dcem_vs_learned_selective" in comparison_names
    assert "phrase_anatomy_dcem_vs_candidate_shuffled" in comparison_names
    assert "dp_msa_vs_baseline" in comparison_names
    assert "dp_msa_vs_phrase_anatomy_dcem" in comparison_names
    assert "dp_msa_vs_candidate_shuffled" in comparison_names
    assert "validation_gated_dcem_vs_disease_pooled_learned" in comparison_names
    assert "validation_gated_dcem_vs_baseline" in comparison_names
    assert "validation_gated_dcem_vs_disease_gated_learned" in comparison_names
    assert "validation_gated_dcem_vs_learned_selective" in comparison_names


def test_comparison_specs_for_methods_adds_named_v3_validation_gate() -> None:
    specs = comparison_specs_for_methods(
        {
            "baseline",
            "learned_selective",
            "disease_pooled_learned",
            "phrase_anatomy_dcem",
            "candidate_shuffled",
            "validation_gated_dcem_v3",
        },
        validation_gate_method_name="validation_gated_dcem_v3",
        validation_gate_source_method="phrase_anatomy_dcem",
    )
    names = {name for name, _, _ in specs}

    assert "validation_gated_dcem_v3_vs_baseline" in names
    assert "validation_gated_dcem_v3_vs_phrase_anatomy_dcem" in names
    assert "validation_gated_dcem_v3_vs_disease_pooled_learned" in names
    assert "validation_gated_dcem_v3_vs_candidate_shuffled" in names


def test_comparison_specs_for_methods_adds_validation_gate_fallback_comparison() -> None:
    specs = comparison_specs_for_methods(
        {
            "baseline",
            "phrase_anatomy_dcem",
            "dp_msa_v2_property_over_v3",
            "validation_gated_dp_msa_v2_property_over_v3",
        },
        validation_gate_method_name="validation_gated_dp_msa_v2_property_over_v3",
        validation_gate_source_method="dp_msa_v2_property_over_v3",
        validation_gate_fallback_method="phrase_anatomy_dcem",
    )
    names = {name for name, _, _ in specs}

    assert "validation_gated_dp_msa_v2_property_over_v3_vs_phrase_anatomy_dcem" in names


def test_comparison_specs_for_methods_adds_dp_msa_lambda_sweep_comparisons() -> None:
    specs = comparison_specs_for_methods(
        {
            "baseline",
            "phrase_anatomy_dcem",
            "candidate_shuffled",
            "dp_msa_lambda0p02",
        }
    )
    names = {name for name, _, _ in specs}

    assert "dp_msa_lambda0p02_vs_baseline" in names
    assert "dp_msa_lambda0p02_vs_phrase_anatomy_dcem" in names
    assert "dp_msa_lambda0p02_vs_candidate_shuffled" in names


def test_decision_requires_baseline_gain_specificity_and_no_macro_harm() -> None:
    summary = pd.DataFrame(
        [
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "macro_scope": "macro_candidate",
                "metric": "cnr",
                "mean_delta": 0.12,
                "ci_low": 0.08,
                "ci_high": 0.18,
            },
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.03,
                "ci_low": 0.0,
                "ci_high": 0.05,
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
    )

    decision = make_learned_repair_decision(summary, candidate_effect_floor=0.02)

    assert decision["verdict"] == "keep_learned_selective_repair"
    assert decision["candidate_vs_baseline_cnr_delta"] == 0.12
    assert decision["specificity_vs_shuffled_cnr_delta"] == 0.2


def test_decision_rejects_when_specificity_against_shuffled_is_missing() -> None:
    summary = pd.DataFrame(
        [
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "macro_scope": "macro_candidate",
                "metric": "cnr",
                "mean_delta": 0.12,
                "ci_low": 0.08,
                "ci_high": 0.18,
            },
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "macro_scope": "macro_all",
                "metric": "cnr",
                "mean_delta": 0.03,
                "ci_low": 0.0,
                "ci_high": 0.05,
            },
            {
                "comparison": "learned_selective_vs_candidate_shuffled",
                "split": "test",
                "macro_scope": "macro_candidate",
                "metric": "cnr",
                "mean_delta": 0.01,
                "ci_low": -0.01,
                "ci_high": 0.03,
            },
        ]
    )

    decision = make_learned_repair_decision(summary, candidate_effect_floor=0.02)

    assert decision["verdict"] == "reject_learned_selective_repair"
    assert decision["reason"] == "specificity_against_shuffled_not_met"


def test_per_class_decision_separates_effusion_strong_and_pneumothorax_partial() -> None:
    per_class = pd.DataFrame(
        [
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "category": "Pleural Effusion",
                "metric": "cnr",
                "mean_delta": 0.08,
                "ci_low": 0.03,
                "ci_high": 0.12,
                "n": 70,
            },
            {
                "comparison": "learned_selective_vs_candidate_shuffled",
                "split": "test",
                "category": "Pleural Effusion",
                "metric": "cnr",
                "mean_delta": 0.3,
                "ci_low": 0.1,
                "ci_high": 0.4,
                "n": 70,
            },
            {
                "comparison": "learned_selective_vs_baseline",
                "split": "test",
                "category": "Pneumothorax",
                "metric": "cnr",
                "mean_delta": 0.01,
                "ci_low": -0.02,
                "ci_high": 0.05,
                "n": 160,
            },
            {
                "comparison": "learned_selective_vs_candidate_shuffled",
                "split": "test",
                "category": "Pneumothorax",
                "metric": "cnr",
                "mean_delta": 0.2,
                "ci_low": 0.05,
                "ci_high": 0.3,
                "n": 160,
            },
        ]
    )

    decisions = make_per_class_learned_repair_decisions(
        per_class,
        candidate_categories=["Pneumothorax", "Pleural Effusion"],
        strong_effect_floor=0.02,
    )

    assert decisions["Pleural Effusion"]["verdict"] == "strong_pass"
    assert decisions["Pleural Effusion"]["expected_role"] == "primary_evidence"
    assert decisions["Pneumothorax"]["verdict"] == "directional_partial_success"
    assert decisions["Pneumothorax"]["expected_role"] == "sparse_class_partial_success_allowed"
