"""Score MS-CXR learned repair heatmaps and summarize paired deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = ("cnr", "iou", "dice")
DEFAULT_METHODS = (
    "baseline",
    "learned_selective",
    "disease_gated_learned",
    "disease_pooled_learned",
    "phrase_anatomy_dcem",
    "dp_msa",
    "all_class_learned",
    "candidate_shuffled",
    "candidate_uniform",
)
VALIDATION_GATED_METHOD = "validation_gated_dcem"
DEFAULT_COMPARISONS = (
    ("learned_selective_vs_baseline", "learned_selective", "baseline"),
    ("disease_gated_learned_vs_baseline", "disease_gated_learned", "baseline"),
    ("disease_gated_learned_vs_learned_selective", "disease_gated_learned", "learned_selective"),
    ("disease_gated_learned_vs_candidate_shuffled", "disease_gated_learned", "candidate_shuffled"),
    ("disease_pooled_learned_vs_baseline", "disease_pooled_learned", "baseline"),
    ("disease_pooled_learned_vs_disease_gated_learned", "disease_pooled_learned", "disease_gated_learned"),
    ("disease_pooled_learned_vs_learned_selective", "disease_pooled_learned", "learned_selective"),
    ("disease_pooled_learned_vs_candidate_shuffled", "disease_pooled_learned", "candidate_shuffled"),
    ("phrase_anatomy_dcem_vs_baseline", "phrase_anatomy_dcem", "baseline"),
    ("phrase_anatomy_dcem_vs_learned_selective", "phrase_anatomy_dcem", "learned_selective"),
    ("phrase_anatomy_dcem_vs_disease_pooled_learned", "phrase_anatomy_dcem", "disease_pooled_learned"),
    ("phrase_anatomy_dcem_vs_candidate_shuffled", "phrase_anatomy_dcem", "candidate_shuffled"),
    ("dp_msa_vs_baseline", "dp_msa", "baseline"),
    ("dp_msa_vs_phrase_anatomy_dcem", "dp_msa", "phrase_anatomy_dcem"),
    ("dp_msa_vs_disease_pooled_learned", "dp_msa", "disease_pooled_learned"),
    ("dp_msa_vs_learned_selective", "dp_msa", "learned_selective"),
    ("dp_msa_vs_candidate_shuffled", "dp_msa", "candidate_shuffled"),
    ("validation_gated_dcem_vs_baseline", VALIDATION_GATED_METHOD, "baseline"),
    ("validation_gated_dcem_vs_disease_gated_learned", VALIDATION_GATED_METHOD, "disease_gated_learned"),
    ("validation_gated_dcem_vs_disease_pooled_learned", VALIDATION_GATED_METHOD, "disease_pooled_learned"),
    ("validation_gated_dcem_vs_learned_selective", VALIDATION_GATED_METHOD, "learned_selective"),
    ("validation_gated_dcem_vs_candidate_shuffled", VALIDATION_GATED_METHOD, "candidate_shuffled"),
    ("all_class_learned_vs_baseline", "all_class_learned", "baseline"),
    ("candidate_shuffled_vs_baseline", "candidate_shuffled", "baseline"),
    ("candidate_uniform_vs_baseline", "candidate_uniform", "baseline"),
    ("learned_selective_vs_all_class_learned", "learned_selective", "all_class_learned"),
    ("learned_selective_vs_candidate_shuffled", "learned_selective", "candidate_shuffled"),
    ("learned_selective_vs_candidate_uniform", "learned_selective", "candidate_uniform"),
)


def comparison_specs_for_methods(
    methods: set[str] | list[str] | tuple[str, ...],
    validation_gate_method_name: str = VALIDATION_GATED_METHOD,
    validation_gate_source_method: str = "disease_gated_learned",
) -> list[tuple[str, str, str]]:
    """Return comparison specs supported by the available method outputs."""

    method_set = {str(method) for method in methods}
    specs = [spec for spec in DEFAULT_COMPARISONS if spec[1] in method_set and spec[2] in method_set]
    existing_names = {name for name, _, _ in specs}

    def append_dynamic(name: str, method_a: str, method_b: str) -> None:
        if name in existing_names:
            return
        if method_a in method_set and method_b in method_set:
            specs.append((name, method_a, method_b))
            existing_names.add(name)

    gate_method = str(validation_gate_method_name)
    source_method = str(validation_gate_source_method)
    if gate_method in method_set:
        dynamic_specs = [
            (f"{gate_method}_vs_baseline", gate_method, "baseline"),
            (f"{gate_method}_vs_{source_method}", gate_method, source_method),
            (f"{gate_method}_vs_disease_pooled_learned", gate_method, "disease_pooled_learned"),
            (f"{gate_method}_vs_candidate_shuffled", gate_method, "candidate_shuffled"),
        ]
        for name, method_a, method_b in dynamic_specs:
            append_dynamic(name, method_a, method_b)

    for method in sorted(method_set):
        if not method.startswith("dp_msa"):
            continue
        for baseline_method in ["baseline", "phrase_anatomy_dcem", "candidate_shuffled"]:
            append_dynamic(f"{method}_vs_{baseline_method}", method, baseline_method)
    return specs


def stable_int(value: str) -> int:
    return int(hashlib.md5(str(value).encode("utf-8")).hexdigest()[:8], 16)


def assign_split(case_id: str, val_fraction: float = 0.3, seed: int = 0) -> str:
    if val_fraction <= 0:
        return "test"
    if val_fraction >= 1:
        return "val"
    value = stable_int(f"{seed}:{case_id}") / float(0xFFFFFFFF)
    return "val" if value < float(val_fraction) else "test"


def parse_csv_list(raw: str | None, default: tuple[str, ...]) -> list[str]:
    if raw is None:
        return list(default)
    values = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def add_method_metadata(per_case: pd.DataFrame, method: str, val_fraction: float, seed: int) -> pd.DataFrame:
    out = per_case.copy()
    out["method"] = method
    if "split" not in out.columns:
        out["split"] = [assign_split(str(case_id), val_fraction=val_fraction, seed=seed) for case_id in out["case_id"]]
    return out


def coerce_eval_dataframe(data: Any, max_cases: int | None = None) -> pd.DataFrame:
    """Normalize AFLoc localization data into a DataFrame."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        frame = pd.DataFrame(data)
    else:
        frame = pd.DataFrame(list(data))
    if max_cases is not None:
        frame = frame.iloc[:max_cases].copy()
    return frame


def filter_eval_dataframe_by_categories(
    data: pd.DataFrame,
    categories: set[str] | list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Keep only evaluation rows for categories with generated heatmaps."""

    if "category" not in data.columns:
        raise ValueError("evaluation data is missing required 'category' column")
    category_set = {str(category) for category in categories}
    if not category_set:
        return data.copy()
    return data[data["category"].astype(str).isin(category_set)].copy()


def metric_dataframe_from_category_values(category_values: dict[str, list[float]]) -> pd.DataFrame:
    """Convert per-category metric lists into a numeric, pandas-2-safe DataFrame."""

    columns = list(category_values.keys())
    row_count = sum(len(values) for values in category_values.values())
    df = pd.DataFrame(np.nan, index=range(row_count), columns=columns, dtype=float)
    row_idx = 0
    for category, values in category_values.items():
        for value in values:
            df.loc[row_idx, category] = float(value)
            row_idx += 1
    return df


def write_bootstrap_ci(
    df: pd.DataFrame,
    save_dir: Path,
    metric: str = "iou",
    num_replicates: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Write AFLoc-style bootstrap CI files without mixing string task names into means."""

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(num_replicates)):
        sample_ids = rng.choice(len(numeric_df), size=len(numeric_df), replace=True)
        rows.append(numeric_df.iloc[sample_ids].mean(axis=0, skipna=True).to_dict())
    bs_df = pd.DataFrame.from_records(rows)
    save_dir.mkdir(parents=True, exist_ok=True)
    bs_df.to_csv(save_dir / f"{metric}_bootstrap_results.csv", index=False)

    records = []
    for task in bs_df.columns:
        values = bs_df[task].dropna().sort_values()
        if values.empty:
            lower = mean = upper = np.nan
        else:
            lower = round(float(values.quantile(0.025)), 3)
            mean = round(float(values.mean()), 3)
            upper = round(float(values.quantile(0.975)), 3)
        records.append({"name": task, "lower": lower, "mean": mean, "upper": upper})
    ci_df = pd.DataFrame.from_records(records).sort_values(by="name").reset_index(drop=True)
    numeric_mean = ci_df[["lower", "mean", "upper"]].mean(axis=0, numeric_only=True)
    mean_row = {"name": "mean", **{key: round(float(value), 3) for key, value in numeric_mean.items()}}
    ci_df = pd.concat([ci_df, pd.DataFrame([mean_row])], axis=0, ignore_index=True)
    ci_df.to_csv(save_dir / f"test_{metric}_summary_results.csv", index=False)
    return ci_df


def build_paired_delta_table(
    method_per_case: pd.DataFrame,
    baseline_per_case: pd.DataFrame,
    split: str | None = None,
    categories: set[str] | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    left = method_per_case.copy()
    right = baseline_per_case.copy()
    if split is not None:
        left = left[left["split"] == split]
        right = right[right["split"] == split]
    if categories is not None:
        category_set = {str(category) for category in categories}
        left = left[left["category"].isin(category_set)]
        right = right[right["category"].isin(category_set)]

    index_cols = ["case_id", "category", "split"]
    merged = left[index_cols + list(METRICS)].merge(
        right[index_cols + list(METRICS)],
        on=index_cols,
        suffixes=("_method", "_baseline"),
    )
    rows = []
    for _, row in merged.iterrows():
        record = {col: row[col] for col in index_cols}
        for metric in METRICS:
            record[f"delta_{metric}"] = float(row[f"{metric}_method"] - row[f"{metric}_baseline"])
        rows.append(record)
    return pd.DataFrame(rows)


def paired_bootstrap_summary(deltas: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        col = f"delta_{metric}"
        if col not in deltas.columns:
            rows.append({"metric": metric, "mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0})
            continue
        values = deltas[col].dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            rows.append({"metric": metric, "mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0})
            continue
        boot = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(int(n_boot))]
        rows.append(
            {
                "metric": metric,
                "mean_delta": round(float(np.mean(values)), 6),
                "ci_low": round(float(np.quantile(boot, 0.025)), 6),
                "ci_high": round(float(np.quantile(boot, 0.975)), 6),
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def per_class_paired_bootstrap_summary(deltas: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    if deltas.empty or "category" not in deltas.columns:
        return pd.DataFrame(columns=["category", "metric", "mean_delta", "ci_low", "ci_high", "n"])
    frames = []
    for idx, (category, group) in enumerate(sorted(deltas.groupby("category"), key=lambda item: str(item[0]))):
        summary = paired_bootstrap_summary(group, n_boot=n_boot, seed=seed + idx)
        summary.insert(0, "category", category)
        frames.append(summary)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def macro_paired_bootstrap_summary(
    deltas: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    scope: str = "macro_all",
    categories: set[str] | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if deltas.empty or "category" not in deltas.columns:
        return pd.DataFrame(columns=["macro_scope", "metric", "mean_delta", "ci_low", "ci_high", "n_categories", "n_cases"])
    df = deltas.copy()
    if categories is not None:
        category_set = {str(category) for category in categories}
        df = df[df["category"].isin(category_set)]

    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        col = f"delta_{metric}"
        if col not in df.columns:
            rows.append({"macro_scope": scope, "metric": metric, "mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_categories": 0, "n_cases": int(len(df))})
            continue
        per_cat = df.groupby("category")[col].mean().dropna()
        if per_cat.empty:
            rows.append({"macro_scope": scope, "metric": metric, "mean_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_categories": 0, "n_cases": int(len(df))})
            continue
        cat_values = per_cat.to_numpy(dtype=np.float64)
        boot = [float(np.mean(rng.choice(cat_values, size=len(cat_values), replace=True))) for _ in range(int(n_boot))]
        rows.append(
            {
                "macro_scope": scope,
                "metric": metric,
                "mean_delta": round(float(np.mean(cat_values)), 6),
                "ci_low": round(float(np.quantile(boot, 0.025)), 6),
                "ci_high": round(float(np.quantile(boot, 0.975)), 6),
                "n_categories": int(len(cat_values)),
                "n_cases": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def summarize_selected_deltas(
    deltas: pd.DataFrame,
    candidate_categories: set[str] | list[str] | tuple[str, ...],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    frames = []
    pooled = paired_bootstrap_summary(deltas, n_boot=n_boot, seed=seed)
    pooled["macro_scope"] = "pooled_all"
    frames.append(pooled)
    non_candidate_categories = sorted(set(deltas["category"].unique()) - {str(cat) for cat in candidate_categories}) if not deltas.empty else []
    for scope, cats in [
        ("macro_all", None),
        ("macro_candidate", candidate_categories),
        ("macro_non_candidate", non_candidate_categories),
    ]:
        frames.append(
            macro_paired_bootstrap_summary(
                deltas,
                n_boot=n_boot,
                seed=seed + stable_int(scope) % 100000,
                scope=scope,
                categories=cats,
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True)


def summarize_comparison(
    method_per_case: pd.DataFrame,
    baseline_per_case: pd.DataFrame,
    comparison: str,
    candidate_categories: set[str] | list[str] | tuple[str, ...],
    split: str,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    deltas = build_paired_delta_table(method_per_case, baseline_per_case, split=split)
    deltas["split"] = split
    deltas["comparison"] = comparison
    summary = summarize_selected_deltas(deltas, candidate_categories=candidate_categories, n_boot=n_boot, seed=seed)
    summary["split"] = split
    summary["comparison"] = comparison
    per_class = per_class_paired_bootstrap_summary(
        deltas,
        n_boot=n_boot,
        seed=seed + stable_int(f"per-class:{comparison}:{split}") % 100000,
    )
    per_class["split"] = split
    per_class["comparison"] = comparison
    return deltas, summary, per_class


def build_validation_gate_decisions(
    per_class_summary: pd.DataFrame,
    candidate_categories: set[str] | list[str] | tuple[str, ...],
    comparison: str = "disease_gated_learned_vs_baseline",
    split: str = "val",
    metric: str = "cnr",
    effect_floor: float = 0.02,
    ci_low_floor: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Decide disease-level pass/bypass from validation per-class deltas."""

    decisions: dict[str, dict[str, Any]] = {}
    if per_class_summary.empty:
        rows = pd.DataFrame()
    else:
        rows = per_class_summary[
            (per_class_summary["comparison"] == comparison)
            & (per_class_summary["split"] == split)
            & (per_class_summary["metric"] == metric)
        ].copy()
    by_category = {str(row["category"]): row for _, row in rows.iterrows()} if not rows.empty else {}
    for category in [str(item) for item in candidate_categories]:
        row = by_category.get(category)
        if row is None:
            decisions[category] = {
                "decision": "bypass",
                "reason": "missing_validation_summary",
                "comparison": comparison,
                "split": split,
                "metric": metric,
                "effect_floor": float(effect_floor),
                "ci_low_floor": float(ci_low_floor),
            }
            continue
        mean_delta = float(row["mean_delta"])
        ci_low = float(row["ci_low"])
        passed = mean_delta >= float(effect_floor) and ci_low >= float(ci_low_floor)
        decisions[category] = {
            "decision": "pass" if passed else "bypass",
            "reason": "validation_effect_and_ci_met" if passed else "validation_effect_or_ci_not_met",
            "comparison": comparison,
            "split": split,
            "metric": metric,
            "effect_floor": float(effect_floor),
            "ci_low_floor": float(ci_low_floor),
            "mean_delta": round(mean_delta, 6),
            "ci_low": round(ci_low, 6),
            "ci_high": round(float(row["ci_high"]), 6),
            "n": int(row.get("n", 0)),
        }
    return decisions


def build_validation_gated_per_case(
    baseline_per_case: pd.DataFrame,
    gated_per_case: pd.DataFrame,
    validation_gate_decisions: dict[str, dict[str, Any]],
    method_name: str = VALIDATION_GATED_METHOD,
) -> pd.DataFrame:
    """Combine gated and baseline per-case metrics using validation decisions."""

    index_cols = ["case_id", "category", "split"]
    base = baseline_per_case.copy()
    gated = gated_per_case.copy()
    merged = gated.merge(base, on=index_cols, suffixes=("_gated", "_baseline"))
    rows = []
    passthrough_cols = ["path", "dicom_id", "label_text"]
    for _, row in merged.iterrows():
        category = str(row["category"])
        decision = str(validation_gate_decisions.get(category, {}).get("decision", "bypass"))
        source = "gated" if decision == "pass" else "baseline"
        record: dict[str, Any] = {col: row[col] for col in index_cols}
        for col in passthrough_cols:
            gated_col = f"{col}_gated"
            baseline_col = f"{col}_baseline"
            if gated_col in row:
                record[col] = row[gated_col]
            elif baseline_col in row:
                record[col] = row[baseline_col]
        for metric in METRICS:
            record[metric] = float(row[f"{metric}_{source}"])
        record["method"] = method_name
        record["validation_gate_decision"] = decision
        record["validation_gate_source"] = source
        rows.append(record)
    return pd.DataFrame(rows)


def _summary_row(summary: pd.DataFrame, comparison: str, macro_scope: str, metric: str = "cnr") -> pd.Series | None:
    rows = summary[
        (summary["comparison"] == comparison)
        & (summary["split"] == "test")
        & (summary["macro_scope"] == macro_scope)
        & (summary["metric"] == metric)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _per_class_row(per_class_summary: pd.DataFrame, comparison: str, category: str, metric: str = "cnr") -> pd.Series | None:
    if per_class_summary.empty:
        return None
    rows = per_class_summary[
        (per_class_summary["comparison"] == comparison)
        & (per_class_summary["split"] == "test")
        & (per_class_summary["category"] == category)
        & (per_class_summary["metric"] == metric)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _candidate_role(category: str) -> str:
    if category == "Pleural Effusion":
        return "primary_evidence"
    if category == "Pneumothorax":
        return "sparse_class_partial_success_allowed"
    return "candidate_evidence"


def make_per_class_learned_repair_decisions(
    per_class_summary: pd.DataFrame,
    candidate_categories: set[str] | list[str] | tuple[str, ...],
    strong_effect_floor: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """Create per-class Stage C decisions so candidate macro cannot hide class asymmetry."""

    decisions: dict[str, dict[str, Any]] = {}
    for category in [str(item) for item in candidate_categories]:
        baseline = _per_class_row(per_class_summary, "learned_selective_vs_baseline", category)
        specificity = _per_class_row(per_class_summary, "learned_selective_vs_candidate_shuffled", category)
        role = _candidate_role(category)
        if baseline is None or specificity is None:
            decisions[category] = {
                "expected_role": role,
                "verdict": "insufficient_evidence",
                "reason": "missing_per_class_test_cnr_summary",
                "strong_effect_floor": strong_effect_floor,
            }
            continue

        baseline_delta = float(baseline["mean_delta"])
        baseline_ci_low = float(baseline["ci_low"])
        specificity_delta = float(specificity["mean_delta"])
        specificity_ci_low = float(specificity["ci_low"])
        strong_pass = baseline_delta >= float(strong_effect_floor) and baseline_ci_low > 0
        specificity_pass = specificity_delta > 0 and specificity_ci_low > 0
        directional_positive = baseline_delta > 0

        if strong_pass and specificity_pass:
            verdict = "strong_pass"
            reason = "effect_floor_ci_and_specificity_met"
        elif role == "sparse_class_partial_success_allowed" and directional_positive and specificity_pass:
            verdict = "directional_partial_success"
            reason = "positive_direction_and_specificity_met_but_strong_floor_not_met"
        elif directional_positive and specificity_pass:
            verdict = "weak_positive"
            reason = "positive_direction_and_specificity_met_but_floor_not_met"
        elif not specificity_pass:
            verdict = "reject_class"
            reason = "specificity_against_shuffled_not_met"
        else:
            verdict = "reject_class"
            reason = "no_positive_effect_vs_baseline"

        decisions[category] = {
            "expected_role": role,
            "verdict": verdict,
            "reason": reason,
            "strong_effect_floor": strong_effect_floor,
            "vs_baseline_cnr_delta": round(baseline_delta, 6),
            "vs_baseline_cnr_ci_low": round(baseline_ci_low, 6),
            "vs_shuffled_cnr_delta": round(specificity_delta, 6),
            "vs_shuffled_cnr_ci_low": round(specificity_ci_low, 6),
            "n": int(baseline.get("n", 0)),
        }
    return decisions


def make_learned_repair_decision(
    summary: pd.DataFrame,
    per_class_summary: pd.DataFrame | None = None,
    candidate_categories: set[str] | list[str] | tuple[str, ...] | None = None,
    candidate_effect_floor: float = 0.02,
    macro_all_harm_floor: float = -0.005,
) -> dict[str, Any]:
    candidate = _summary_row(summary, "learned_selective_vs_baseline", "macro_candidate")
    macro_all = _summary_row(summary, "learned_selective_vs_baseline", "macro_all")
    specificity = _summary_row(summary, "learned_selective_vs_candidate_shuffled", "macro_candidate")
    if candidate is None or macro_all is None or specificity is None:
        decision = {
            "verdict": "insufficient_evidence",
            "reason": "missing_test_cnr_summary",
            "candidate_effect_floor": candidate_effect_floor,
            "macro_all_harm_floor": macro_all_harm_floor,
        }
        if per_class_summary is not None and candidate_categories is not None:
            decision["per_class_decisions"] = make_per_class_learned_repair_decisions(
                per_class_summary,
                candidate_categories=candidate_categories,
                strong_effect_floor=candidate_effect_floor,
            )
        return decision

    candidate_pass = float(candidate["mean_delta"]) >= float(candidate_effect_floor) and float(candidate["ci_low"]) > 0
    specificity_pass = float(specificity["mean_delta"]) > 0 and float(specificity["ci_low"]) > 0
    macro_safe = float(macro_all["ci_low"]) >= float(macro_all_harm_floor)
    if not candidate_pass:
        verdict = "reject_learned_selective_repair"
        reason = "candidate_effect_floor_or_ci_not_met"
    elif not specificity_pass:
        verdict = "reject_learned_selective_repair"
        reason = "specificity_against_shuffled_not_met"
    elif not macro_safe:
        verdict = "candidate_only_but_macro_harm"
        reason = "candidate_improves_but_macro_all_harm"
    else:
        verdict = "keep_learned_selective_repair"
        reason = "candidate_gain_specificity_and_macro_safety_met"

    decision = {
        "verdict": verdict,
        "reason": reason,
        "candidate_effect_floor": candidate_effect_floor,
        "macro_all_harm_floor": macro_all_harm_floor,
        "candidate_vs_baseline_cnr_delta": round(float(candidate["mean_delta"]), 6),
        "candidate_vs_baseline_cnr_ci_low": round(float(candidate["ci_low"]), 6),
        "specificity_vs_shuffled_cnr_delta": round(float(specificity["mean_delta"]), 6),
        "specificity_vs_shuffled_cnr_ci_low": round(float(specificity["ci_low"]), 6),
        "macro_all_cnr_delta": round(float(macro_all["mean_delta"]), 6),
        "macro_all_cnr_ci_low": round(float(macro_all["ci_low"]), 6),
    }
    if per_class_summary is not None and candidate_categories is not None:
        decision["per_class_decisions"] = make_per_class_learned_repair_decisions(
            per_class_summary,
            candidate_categories=candidate_categories,
            strong_effect_floor=candidate_effect_floor,
        )
    return decision


def evaluate_hmaps(data: pd.DataFrame, hmaps: dict[str, dict[str, Any]], dataset: str, save_dir: Path, margin: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        raise ValueError(
            f"No evaluation rows available for dataset={dataset}; "
            "check dataset protocol and heatmap keys before scoring."
        )

    from collections import defaultdict

    from localization.common import dict_mean, prefix
    from localization.constants import threshold_list_dct
    from localization.datasets import norm_heatmap
    from localization.metrics import compute_cnr, compute_dice, compute_iou

    save_dir.mkdir(parents=True, exist_ok=True)
    threshold_list = threshold_list_dct[dataset]
    iou_thre_cat = []
    cnr_thre_cat = []
    dice_thre_cat = []
    sep_cat: dict[str, pd.DataFrame] = {}
    metric_df_thre_iou = []
    metric_df_thre_cnr = []
    metric_df_thre_dice = []
    case_metrics = defaultdict(lambda: {"iou": [], "cnr": [], "dice": []})

    if margin:
        from localization.common import Pipeline

    for threshold in threshold_list:
        cat_ious = defaultdict(list)
        cat_cnrs = defaultdict(list)
        cat_dices = defaultdict(list)
        for path, label_text, gtmask, cat in zip(data["path"], data["label_text"], data["gtmasks"], data["category"]):
            key = str(path) + label_text
            hmap = np.asarray(hmaps[key]["hmap"], dtype=np.float32)
            if margin:
                hmap = Pipeline.set_margin(hmap)
            nan = np.isnan(hmap)
            heatmap = norm_heatmap(hmap, nan, mode=0)
            mask = np.where(heatmap > threshold, 1, 0)
            iou = compute_iou(gtmask, mask, nan)
            cnr = compute_cnr(gtmask, heatmap, nan)
            dice = compute_dice(gtmask, mask, nan)
            cat_ious[cat].append(iou)
            cat_cnrs[cat].append(cnr)
            cat_dices[cat].append(dice)

            case_id = key
            case_metrics[case_id]["path"] = str(path)
            case_metrics[case_id]["dicom_id"] = Path(str(path)).stem
            case_metrics[case_id]["label_text"] = label_text
            case_metrics[case_id]["category"] = cat
            case_metrics[case_id]["iou"].append(iou)
            case_metrics[case_id]["cnr"].append(cnr)
            case_metrics[case_id]["dice"].append(dice)

        metric_df_iou = metric_dataframe_from_category_values(cat_ious)
        metric_df_cnr = metric_dataframe_from_category_values(cat_cnrs)
        metric_df_dice = metric_dataframe_from_category_values(cat_dices)
        metric_df_thre_iou.append(metric_df_iou)
        metric_df_thre_cnr.append(metric_df_cnr)
        metric_df_thre_dice.append(metric_df_dice)
        iou_thre_cat.append(dict_mean(cat_ious))
        cnr_thre_cat.append(dict_mean(cat_cnrs))
        dice_thre_cat.append(dict_mean(cat_dices))

        if "iou" not in sep_cat:
            sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
            sep_cat["cnr"] = pd.DataFrame(dict_mean(cat_cnrs, sep=True))
            sep_cat["dice"] = pd.DataFrame(dict_mean(cat_dices, sep=True))
        else:
            sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)
            sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(dict_mean(cat_cnrs, sep=True))], axis=0, ignore_index=True)
            sep_cat["dice"] = pd.concat([sep_cat["dice"], pd.DataFrame(dict_mean(cat_dices, sep=True))], axis=0, ignore_index=True)

    total_df_iou = sum(metric_df_thre_iou[1:], metric_df_thre_iou[0])
    total_df_cnr = sum(metric_df_thre_cnr[1:], metric_df_thre_cnr[0])
    total_df_dice = sum(metric_df_thre_dice[1:], metric_df_thre_dice[0])
    write_bootstrap_ci(total_df_iou / len(metric_df_thre_iou), save_dir, "iou")
    write_bootstrap_ci(total_df_cnr / len(metric_df_thre_cnr), save_dir, "cnr")
    write_bootstrap_ci(total_df_dice / len(metric_df_thre_dice), save_dir, "dice")

    res = pd.DataFrame()
    res["threshold"] = threshold_list + ["mean"]
    res["iou_cat"] = iou_thre_cat + [np.mean(iou_thre_cat)]
    res["cnr_cat"] = cnr_thre_cat + [np.mean(cnr_thre_cat)]
    res["dice_cat"] = dice_thre_cat + [np.mean(dice_thre_cat)]
    for key in METRICS:
        sep_cat[key].rename(columns=prefix(f"{key}_", sep_cat[key].columns), inplace=True)
        sep_cat[key] = pd.concat([sep_cat[key], pd.DataFrame(sep_cat[key].mean(axis=0)).T], axis=0, ignore_index=True)
        res = pd.concat([res, sep_cat[key]], axis=1)
    res = res.round(3)
    res.to_csv(save_dir / "metric.csv", index=False)

    per_case = []
    for case_id, values in case_metrics.items():
        per_case.append(
            {
                "case_id": case_id,
                "path": values["path"],
                "dicom_id": values["dicom_id"],
                "label_text": values["label_text"],
                "category": values["category"],
                "iou": float(np.nanmean(values["iou"])),
                "cnr": float(np.nanmean(values["cnr"])),
                "dice": float(np.nanmean(values["dice"])),
            }
        )
    per_case_df = pd.DataFrame(per_case)
    per_case_df.to_csv(save_dir / "per_case_metric.csv", index=False)
    return res, per_case_df


def score_method_hmaps(
    hmaps_root: Path,
    outdir: Path,
    methods: list[str],
    candidate_categories: list[str],
    dataset: str = "MS_CXR_CLS",
    val_fraction: float = 0.3,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
    margin: bool = False,
    max_cases: int | None = None,
    candidate_effect_floor: float = 0.02,
    macro_all_harm_floor: float = -0.005,
    validation_gate: bool = False,
    validation_gate_source_method: str = "disease_gated_learned",
    validation_gate_method_name: str = VALIDATION_GATED_METHOD,
    validation_gate_effect_floor: float = 0.02,
    validation_gate_ci_low_floor: float = 0.0,
) -> dict[str, Any]:
    from localization.datasets import load_data

    outdir.mkdir(parents=True, exist_ok=True)
    raw_data = coerce_eval_dataframe(load_data(dataset=dataset), max_cases=max_cases)
    data = filter_eval_dataframe_by_categories(raw_data, candidate_categories)
    if data.empty:
        raise ValueError(f"No evaluation rows remain after filtering to candidate_categories={candidate_categories}")

    per_case_by_method: dict[str, pd.DataFrame] = {}
    metric_outputs: dict[str, dict[str, str]] = {}
    for method in methods:
        hmap_path = hmaps_root / method / "hmaps.npy"
        if not hmap_path.exists():
            raise FileNotFoundError(hmap_path)
        hmaps = np.load(hmap_path, allow_pickle=True).item()
        method_dir = outdir / method
        metric_df, per_case = evaluate_hmaps(data, hmaps, dataset=dataset, save_dir=method_dir, margin=margin)
        per_case = add_method_metadata(per_case, method=method, val_fraction=val_fraction, seed=seed)
        per_case.to_csv(method_dir / "per_case_metric.csv", index=False)
        metric_df.to_csv(method_dir / "metric.csv", index=False)
        per_case_by_method[method] = per_case
        metric_outputs[method] = {
            "metric_csv": str(method_dir / "metric.csv"),
            "per_case_metric_csv": str(method_dir / "per_case_metric.csv"),
        }

    validation_gate_payload: dict[str, Any] | None = None
    if validation_gate:
        required_methods = {"baseline", str(validation_gate_source_method)}
        missing_required = sorted(required_methods - set(per_case_by_method))
        if missing_required:
            raise ValueError(f"validation_gate requires scored methods: {missing_required}")
        gate_comparison = f"{validation_gate_source_method}_vs_baseline"
        val_deltas = build_paired_delta_table(
            method_per_case=per_case_by_method[str(validation_gate_source_method)],
            baseline_per_case=per_case_by_method["baseline"],
            split="val",
        )
        val_per_class = per_class_paired_bootstrap_summary(
            val_deltas,
            n_boot=bootstrap_replicates,
            seed=seed + stable_int("validation-gate:per-class") % 100000,
        )
        if not val_per_class.empty:
            val_per_class["split"] = "val"
            val_per_class["comparison"] = gate_comparison
        validation_gate_decisions = build_validation_gate_decisions(
            val_per_class,
            candidate_categories=candidate_categories,
            comparison=gate_comparison,
            effect_floor=validation_gate_effect_floor,
            ci_low_floor=validation_gate_ci_low_floor,
        )
        validation_gated = build_validation_gated_per_case(
            baseline_per_case=per_case_by_method["baseline"],
            gated_per_case=per_case_by_method[str(validation_gate_source_method)],
            validation_gate_decisions=validation_gate_decisions,
            method_name=str(validation_gate_method_name),
        )
        method_dir = outdir / str(validation_gate_method_name)
        method_dir.mkdir(parents=True, exist_ok=True)
        validation_gated.to_csv(method_dir / "per_case_metric.csv", index=False)
        per_case_by_method[str(validation_gate_method_name)] = validation_gated
        metric_outputs[str(validation_gate_method_name)] = {
            "per_case_metric_csv": str(method_dir / "per_case_metric.csv"),
        }
        validation_gate_payload = {
            "enabled": True,
            "method": str(validation_gate_method_name),
            "source_method": str(validation_gate_source_method),
            "fallback_method": "baseline",
            "selection_split": "val",
            "comparison": gate_comparison,
            "metric": "cnr",
            "effect_floor": float(validation_gate_effect_floor),
            "ci_low_floor": float(validation_gate_ci_low_floor),
            "decisions": validation_gate_decisions,
        }
        gate_path = outdir / "validation_gate_decision.json"
        gate_path.write_text(json.dumps(validation_gate_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        metric_outputs[str(validation_gate_method_name)]["validation_gate_decision_json"] = str(gate_path)

    delta_frames = []
    summary_frames = []
    per_class_frames = []
    comparison_specs = comparison_specs_for_methods(
        set(per_case_by_method),
        validation_gate_method_name=str(validation_gate_method_name),
        validation_gate_source_method=str(validation_gate_source_method),
    )
    for split in ["val", "test"]:
        for comparison, method_a, method_b in comparison_specs:
            deltas, summary, per_class = summarize_comparison(
                method_per_case=per_case_by_method[method_a],
                baseline_per_case=per_case_by_method[method_b],
                comparison=comparison,
                candidate_categories=set(candidate_categories),
                split=split,
                n_boot=bootstrap_replicates,
                seed=seed + stable_int(f"{comparison}:{split}") % 100000,
            )
            delta_frames.append(deltas)
            summary_frames.append(summary)
            per_class_frames.append(per_class)

    delta_table = pd.concat(delta_frames, axis=0, ignore_index=True) if delta_frames else pd.DataFrame()
    bootstrap_summary = pd.concat(summary_frames, axis=0, ignore_index=True) if summary_frames else pd.DataFrame()
    per_class_summary = pd.concat(per_class_frames, axis=0, ignore_index=True) if per_class_frames else pd.DataFrame()
    decision = make_learned_repair_decision(
        bootstrap_summary,
        per_class_summary=per_class_summary,
        candidate_categories=candidate_categories,
        candidate_effect_floor=candidate_effect_floor,
        macro_all_harm_floor=macro_all_harm_floor,
    )

    all_per_case = pd.concat(list(per_case_by_method.values()), axis=0, ignore_index=True)
    paths = {
        "per_case_metrics": outdir / "all_per_case_metrics.csv",
        "delta_table": outdir / "delta_table.csv",
        "bootstrap_summary": outdir / "bootstrap_summary.csv",
        "per_class_bootstrap_summary": outdir / "per_class_bootstrap_summary.csv",
        "decision": outdir / "learned_repair_decision.json",
    }
    all_per_case.to_csv(paths["per_case_metrics"], index=False)
    delta_table.to_csv(paths["delta_table"], index=False)
    bootstrap_summary.to_csv(paths["bootstrap_summary"], index=False)
    per_class_summary.to_csv(paths["per_class_bootstrap_summary"], index=False)
    paths["decision"].write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "status": "ok",
        "hmaps_root": str(hmaps_root),
        "outdir": str(outdir),
        "dataset": dataset,
        "methods": methods,
        "candidate_categories": candidate_categories,
        "num_eval_rows_before_category_filter": int(raw_data.shape[0]),
        "num_eval_rows": int(data.shape[0]),
        "decision": decision,
        "validation_gate": validation_gate_payload or {"enabled": False},
        "method_outputs": metric_outputs,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    summary_path = outdir / "learned_repair_metrics_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["outputs"]["summary"] = str(summary_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score pre-built MS-CXR learned repair heatmaps.")
    parser.add_argument("--hmaps-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--candidate-categories", default="Pneumothorax,Pleural Effusion")
    parser.add_argument("--dataset", default="MS_CXR_CLS")
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--candidate-effect-floor", type=float, default=0.02)
    parser.add_argument("--macro-all-harm-floor", type=float, default=-0.005)
    parser.add_argument(
        "--validation-gate",
        action="store_true",
        help="Add validation_gated_dcem by selecting pass/bypass on val split before test summaries.",
    )
    parser.add_argument(
        "--validation-gate-source-method",
        default="disease_gated_learned",
        help="Method used on validation split for disease-level pass/bypass decisions.",
    )
    parser.add_argument(
        "--validation-gate-method-name",
        default=VALIDATION_GATED_METHOD,
        help="Output method name for validation-gated per-case metrics.",
    )
    parser.add_argument("--validation-gate-effect-floor", type=float, default=0.02)
    parser.add_argument("--validation-gate-ci-low-floor", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = score_method_hmaps(
        hmaps_root=args.hmaps_root,
        outdir=args.outdir,
        methods=parse_csv_list(args.methods, DEFAULT_METHODS),
        candidate_categories=parse_csv_list(args.candidate_categories, ("Pneumothorax", "Pleural Effusion")),
        dataset=args.dataset,
        val_fraction=args.val_fraction,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        margin=args.margin,
        max_cases=args.max_cases,
        candidate_effect_floor=args.candidate_effect_floor,
        macro_all_harm_floor=args.macro_all_harm_floor,
        validation_gate=args.validation_gate,
        validation_gate_source_method=args.validation_gate_source_method,
        validation_gate_method_name=args.validation_gate_method_name,
        validation_gate_effect_floor=args.validation_gate_effect_floor,
        validation_gate_ci_low_floor=args.validation_gate_ci_low_floor,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
