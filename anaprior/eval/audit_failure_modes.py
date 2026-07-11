"""Post-hoc failure-mode audit for DCEM-v4 design.

Stage F does not train, tune, or change heatmaps. It summarizes already-written
Stage C/E artifacts so the next module revision is driven by observed failure
routes instead of hand-picked rules.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from anaprior.eval.analyze_learned_oracle_gap import dataframe_to_markdown


TARGET_FAILURE_CATEGORIES = ("Pneumonia", "Consolidation", "Lung Opacity")
CENTRAL_SHORTCUT_REGIONS = {"cardiac_silhouette", "hilar_mediastinal"}
BROAD_SHORTCUT_REGIONS = {"bilateral_lungs"}


@dataclass(frozen=True)
class FailureAuditResult:
    phrase_summary: pd.DataFrame
    region_confusion: pd.DataFrame
    shortcut_summary: pd.DataFrame
    case_examples: pd.DataFrame
    recommendations: pd.DataFrame
    report_markdown: str


def _norm_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_subtype(phrase: str) -> str:
    """Return a compact failure-audit subtype for parenchymal phrases."""

    text = _norm_text(phrase)
    has_pneumonia = "pneumonia" in text or "infect" in text
    has_consolidation = "consolidation" in text or "consolidative" in text
    has_opacity = "opacity" in text or "opacities" in text or "opaque" in text
    has_airspace = "airspace" in text or "air space" in text
    has_retrocardiac = "retrocardiac" in text or "retro cardiac" in text
    has_basilar = any(token in text for token in ("basilar", "bibasilar", "basal", "base", "lower lobe"))
    has_multifocal = any(token in text for token in ("multifocal", "bilateral", "diffuse", "patchy"))

    if has_pneumonia and has_airspace:
        return "pneumonia_airspace"
    if has_pneumonia and has_multifocal:
        return "pneumonia_multifocal"
    if has_pneumonia:
        return "pneumonia"
    if has_consolidation and has_retrocardiac:
        return "consolidation_retrocardiac"
    if has_consolidation and has_basilar:
        return "consolidation_basilar"
    if has_consolidation:
        return "consolidation"
    if has_opacity and has_airspace:
        return "opacity_airspace"
    if has_opacity and has_basilar:
        return "opacity_basilar"
    if has_opacity and has_multifocal:
        return "opacity_multifocal"
    if has_opacity:
        return "opacity"
    return "other"


def _filter_categories(df: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "category" not in df.columns:
        raise ValueError("input dataframe must contain a category column")
    wanted = {str(category) for category in categories}
    return df[df["category"].isin(wanted)].copy()


def _ensure_phrase_column(case_gap: pd.DataFrame, phrase_by_case: dict[str, str] | None = None) -> pd.DataFrame:
    out = case_gap.copy()
    if "phrase" not in out.columns:
        out["phrase"] = ""
    if phrase_by_case:
        out["phrase"] = [
            str(value) if str(value).strip() else phrase_by_case.get(str(case_id), "")
            for case_id, value in zip(out["case_id"], out["phrase"])
        ]
    if "finding" in out.columns:
        out["phrase"] = [
            str(phrase) if str(phrase).strip() else str(finding)
            for phrase, finding in zip(out["phrase"], out["finding"])
        ]
    return out


def _metric_deltas(per_case_metrics: pd.DataFrame, method: str, baseline_method: str) -> pd.DataFrame:
    if per_case_metrics.empty:
        return pd.DataFrame(columns=["case_id", "delta_cnr", "delta_iou", "delta_dice"])
    required = {"case_id", "method", "cnr", "iou", "dice"}
    missing = required - set(per_case_metrics.columns)
    if missing:
        raise ValueError(f"per_case_metrics missing required columns: {sorted(missing)}")
    left = per_case_metrics[per_case_metrics["method"] == method]
    right = per_case_metrics[per_case_metrics["method"] == baseline_method]
    merged = left[["case_id", "cnr", "iou", "dice"]].merge(
        right[["case_id", "cnr", "iou", "dice"]],
        on="case_id",
        suffixes=("_method", "_baseline"),
        how="inner",
    )
    for metric in ("cnr", "iou", "dice"):
        merged[f"delta_{metric}"] = merged[f"{metric}_method"].astype(float) - merged[f"{metric}_baseline"].astype(float)
    return merged[["case_id", "delta_cnr", "delta_iou", "delta_dice"]].copy()


def _prepare_cases(
    case_gap: pd.DataFrame,
    per_case_metrics: pd.DataFrame,
    categories: Sequence[str],
    method: str,
    baseline_method: str,
    phrase_by_case: dict[str, str] | None = None,
) -> pd.DataFrame:
    cases = _ensure_phrase_column(_filter_categories(case_gap, categories), phrase_by_case=phrase_by_case)
    if cases.empty:
        return cases
    for column in ("delta_cnr", "delta_iou", "delta_dice"):
        if column not in cases.columns:
            cases[column] = pd.NA
    missing_delta = cases["delta_cnr"].isna() if "delta_cnr" in cases.columns else pd.Series(True, index=cases.index)
    if missing_delta.any():
        deltas = _metric_deltas(per_case_metrics, method=method, baseline_method=baseline_method)
        cases = cases.drop(columns=[col for col in ("delta_cnr", "delta_iou", "delta_dice") if col in cases.columns])
        cases = cases.merge(deltas, on="case_id", how="left")
    cases["phrase_subtype"] = cases["phrase"].map(phrase_subtype)
    cases["learned_top_region"] = cases["learned_top_region"].astype(str)
    cases["oracle_top_region"] = cases["oracle_top_region"].astype(str)
    cases["is_top1_hit"] = cases["learned_top_region"] == cases["oracle_top_region"]
    cases["is_central_shortcut"] = cases["learned_top_region"].isin(CENTRAL_SHORTCUT_REGIONS)
    cases["is_broad_shortcut"] = cases["learned_top_region"].isin(BROAD_SHORTCUT_REGIONS)
    cases["is_negative_delta_cnr"] = cases["delta_cnr"].astype(float) < 0
    return cases


def _build_phrase_summary(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "phrase_subtype",
                "n_cases",
                "fraction_within_category",
                "mean_delta_cnr",
                "central_shortcut_rate",
                "top1_hit_rate",
            ]
        )
    grouped = cases.groupby(["category", "phrase_subtype"], dropna=False)
    rows: list[dict[str, Any]] = []
    category_counts = cases.groupby("category")["case_id"].count().to_dict()
    for (category, subtype), group in grouped:
        rows.append(
            {
                "category": category,
                "phrase_subtype": subtype,
                "n_cases": int(len(group)),
                "fraction_within_category": round(float(len(group) / category_counts[category]), 6),
                "mean_delta_cnr": round(float(group["delta_cnr"].astype(float).mean()), 6),
                "central_shortcut_rate": round(float(group["is_central_shortcut"].mean()), 6),
                "top1_hit_rate": round(float(group["is_top1_hit"].mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["category", "n_cases", "mean_delta_cnr"], ascending=[True, False, True])


def _build_region_confusion(cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "category",
        "oracle_top_region",
        "learned_top_region",
        "count",
        "route_fraction_within_category",
        "mean_delta_cnr",
        "mean_delta_iou",
        "mean_delta_dice",
        "is_hit",
        "is_central_shortcut",
        "is_broad_shortcut",
    ]
    if cases.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    category_counts = cases.groupby("category")["case_id"].count().to_dict()
    for (category, oracle_region, learned_region), group in cases.groupby(
        ["category", "oracle_top_region", "learned_top_region"],
        dropna=False,
    ):
        is_hit = bool(oracle_region == learned_region)
        rows.append(
            {
                "category": category,
                "oracle_top_region": oracle_region,
                "learned_top_region": learned_region,
                "count": int(len(group)),
                "route_fraction_within_category": round(float(len(group) / category_counts[category]), 6),
                "mean_delta_cnr": round(float(group["delta_cnr"].astype(float).mean()), 6),
                "mean_delta_iou": round(float(group["delta_iou"].astype(float).mean()), 6),
                "mean_delta_dice": round(float(group["delta_dice"].astype(float).mean()), 6),
                "is_hit": is_hit,
                "is_central_shortcut": bool(learned_region in CENTRAL_SHORTCUT_REGIONS),
                "is_broad_shortcut": bool(learned_region in BROAD_SHORTCUT_REGIONS),
            }
        )
    out = pd.DataFrame(rows, columns=columns)
    out["is_hit"] = out["is_hit"].astype(object)
    out["is_central_shortcut"] = out["is_central_shortcut"].astype(object)
    out["is_broad_shortcut"] = out["is_broad_shortcut"].astype(object)
    return out.sort_values(["category", "count", "mean_delta_cnr"], ascending=[True, False, True]).reset_index(drop=True)


def _build_shortcut_summary(cases: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "category",
        "n_cases",
        "mean_delta_cnr",
        "mean_delta_iou",
        "mean_delta_dice",
        "top1_hit_rate",
        "top3_hit_rate",
        "cardiac_or_hilar_shortcut_rate",
        "bilateral_shortcut_rate",
        "shortcut_harm_rate",
        "negative_delta_cnr_rate",
    ]
    if cases.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for category, group in cases.groupby("category", dropna=False):
        top3_col = "top3_hit" if "top3_hit" in group.columns else "is_top1_hit"
        shortcut_mask = group[["is_central_shortcut", "is_broad_shortcut"]].any(axis=1)
        rows.append(
            {
                "category": category,
                "n_cases": int(len(group)),
                "mean_delta_cnr": round(float(group["delta_cnr"].astype(float).mean()), 6),
                "mean_delta_iou": round(float(group["delta_iou"].astype(float).mean()), 6),
                "mean_delta_dice": round(float(group["delta_dice"].astype(float).mean()), 6),
                "top1_hit_rate": round(float(group["is_top1_hit"].mean()), 6),
                "top3_hit_rate": round(float(group[top3_col].astype(bool).mean()), 6),
                "cardiac_or_hilar_shortcut_rate": round(float(group["is_central_shortcut"].mean()), 6),
                "bilateral_shortcut_rate": round(float(group["is_broad_shortcut"].mean()), 6),
                "shortcut_harm_rate": round(float((shortcut_mask & group["is_negative_delta_cnr"]).mean()), 6),
                "negative_delta_cnr_rate": round(float(group["is_negative_delta_cnr"].mean()), 6),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("mean_delta_cnr").reset_index(drop=True)


def _build_case_examples(cases: pd.DataFrame, max_examples_per_category: int) -> pd.DataFrame:
    columns = [
        "category",
        "case_id",
        "phrase",
        "phrase_subtype",
        "learned_top_region",
        "oracle_top_region",
        "delta_cnr",
        "delta_iou",
        "delta_dice",
        "failure_mode",
        "example_reason",
    ]
    if cases.empty:
        return pd.DataFrame(columns=columns)
    rows: list[pd.DataFrame] = []
    for _, group in cases.sort_values(["category", "delta_cnr"]).groupby("category", dropna=False):
        examples = group.head(max_examples_per_category).copy()
        examples["example_reason"] = [
            "central_shortcut" if central else "worst_delta_cnr"
            for central in examples["is_central_shortcut"]
        ]
        rows.append(examples)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    return out[columns].reset_index(drop=True)


def _recommend_action(row: pd.Series) -> str:
    actions: list[str] = []
    if float(row.get("cardiac_or_hilar_shortcut_rate", 0.0)) >= 0.25:
        actions.append("hard-negative ranking against cardiac/hilar shortcuts")
    if float(row.get("top1_hit_rate", 0.0)) < 0.25:
        actions.append("phrase subtype encoder")
    if float(row.get("mean_delta_cnr", 0.0)) < 0.0:
        actions.append("dense residual spatial adapter")
    if float(row.get("bilateral_shortcut_rate", 0.0)) >= 0.25:
        actions.append("local top-k pooling instead of broad bilateral pooling")
    if not actions:
        actions.append("keep v3 routing and validate shape-aware adapter as ablation")
    return " + ".join(actions)


def _build_recommendations(shortcut_summary: pd.DataFrame, phrase_summary: pd.DataFrame) -> pd.DataFrame:
    if shortcut_summary.empty:
        return pd.DataFrame(columns=["category", "dominant_phrase_subtype", "recommended_v4_action", "rationale"])
    rows: list[dict[str, Any]] = []
    for _, row in shortcut_summary.iterrows():
        category = row["category"]
        phrase_rows = phrase_summary[phrase_summary["category"] == category]
        dominant = ""
        if not phrase_rows.empty:
            dominant = str(phrase_rows.sort_values("n_cases", ascending=False).iloc[0]["phrase_subtype"])
        action = _recommend_action(row)
        rows.append(
            {
                "category": category,
                "dominant_phrase_subtype": dominant,
                "recommended_v4_action": action,
                "rationale": (
                    f"mean_delta_cnr={row['mean_delta_cnr']}; "
                    f"top1_hit_rate={row['top1_hit_rate']}; "
                    f"cardiac_or_hilar_shortcut_rate={row['cardiac_or_hilar_shortcut_rate']}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_report(
    phrase_summary: pd.DataFrame,
    region_confusion: pd.DataFrame,
    shortcut_summary: pd.DataFrame,
    case_examples: pd.DataFrame,
    recommendations: pd.DataFrame,
    method: str,
    baseline_method: str,
) -> str:
    lines = [
        "# Stage F Failure Audit",
        "",
        "This post-hoc audit targets Pneumonia, Consolidation, and Lung Opacity to guide DCEM-v4 design.",
        "",
        f"- method: `{method}`",
        f"- baseline_method: `{baseline_method}`",
        "",
        "## Shortcut Summary",
        "",
        dataframe_to_markdown(shortcut_summary),
        "",
        "## Phrase Subtypes",
        "",
        dataframe_to_markdown(phrase_summary),
        "",
        "## Oracle-to-Learned Region Routes",
        "",
        dataframe_to_markdown(region_confusion.head(30)),
        "",
        "## V4 Recommendations",
        "",
        dataframe_to_markdown(recommendations),
        "",
        "## Representative Failure Cases",
        "",
        dataframe_to_markdown(case_examples),
        "",
        "Design implication: prioritize a phrase subtype encoder, hard-negative ranking, and a dense residual spatial adapter before changing the validation gate.",
    ]
    return "\n".join(lines) + "\n"


def build_failure_audit(
    case_gap: pd.DataFrame,
    region_gap: pd.DataFrame,
    per_case_metrics: pd.DataFrame,
    categories: Sequence[str] = TARGET_FAILURE_CATEGORIES,
    method: str = "phrase_anatomy_dcem",
    baseline_method: str = "baseline",
    max_examples_per_category: int = 5,
    phrase_by_case: dict[str, str] | None = None,
) -> FailureAuditResult:
    del region_gap  # Reserved for future region-score distribution summaries.
    cases = _prepare_cases(
        case_gap=case_gap,
        per_case_metrics=per_case_metrics,
        categories=categories,
        method=method,
        baseline_method=baseline_method,
        phrase_by_case=phrase_by_case,
    )
    phrase_summary = _build_phrase_summary(cases)
    region_confusion = _build_region_confusion(cases)
    shortcut_summary = _build_shortcut_summary(cases)
    case_examples = _build_case_examples(cases, max_examples_per_category=max_examples_per_category)
    recommendations = _build_recommendations(shortcut_summary, phrase_summary)
    report = _build_report(
        phrase_summary=phrase_summary,
        region_confusion=region_confusion,
        shortcut_summary=shortcut_summary,
        case_examples=case_examples,
        recommendations=recommendations,
        method=method,
        baseline_method=baseline_method,
    )
    return FailureAuditResult(
        phrase_summary=phrase_summary,
        region_confusion=region_confusion,
        shortcut_summary=shortcut_summary,
        case_examples=case_examples,
        recommendations=recommendations,
        report_markdown=report,
    )


def write_failure_audit(result: FailureAuditResult, outdir: Path) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "phrase_summary": outdir / "failure_phrase_summary.csv",
        "region_confusion": outdir / "failure_region_confusion.csv",
        "shortcut_summary": outdir / "failure_shortcut_summary.csv",
        "case_examples": outdir / "failure_case_examples.csv",
        "recommendations": outdir / "failure_v4_recommendations.csv",
        "markdown": outdir / "failure_audit_report.md",
    }
    result.phrase_summary.to_csv(paths["phrase_summary"], index=False)
    result.region_confusion.to_csv(paths["region_confusion"], index=False)
    result.shortcut_summary.to_csv(paths["shortcut_summary"], index=False)
    result.case_examples.to_csv(paths["case_examples"], index=False)
    result.recommendations.to_csv(paths["recommendations"], index=False)
    paths["markdown"].write_text(result.report_markdown, encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def _parse_csv_list(raw: str, default: Sequence[str]) -> list[str]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or list(default)


def _load_phrase_by_case(prepared_inputs_npz: Path | None) -> dict[str, str]:
    if prepared_inputs_npz is None:
        return {}
    import numpy as np

    payload = np.load(prepared_inputs_npz, allow_pickle=True)
    items = payload["items"].tolist()
    phrase_by_case: dict[str, str] = {}
    for item in items:
        case_id = str(item.get("case_id", ""))
        if not case_id:
            continue
        phrase_by_case[case_id] = str(item.get("phrase", item.get("finding", "")))
    return phrase_by_case


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit parenchymal DCEM failure modes for DCEM-v4 design.")
    parser.add_argument("--case-gap-csv", required=True, type=Path)
    parser.add_argument("--region-gap-csv", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--prepared-inputs-npz", type=Path, default=None)
    parser.add_argument("--categories", default=",".join(TARGET_FAILURE_CATEGORIES))
    parser.add_argument("--method", default="phrase_anatomy_dcem")
    parser.add_argument("--baseline-method", default="baseline")
    parser.add_argument("--max-examples-per-category", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    categories = _parse_csv_list(args.categories, TARGET_FAILURE_CATEGORIES)
    case_gap = pd.read_csv(args.case_gap_csv)
    region_gap = pd.read_csv(args.region_gap_csv)
    metrics = pd.read_csv(args.metrics_csv)
    phrase_by_case = _load_phrase_by_case(args.prepared_inputs_npz)
    result = build_failure_audit(
        case_gap=case_gap,
        region_gap=region_gap,
        per_case_metrics=metrics,
        categories=categories,
        method=args.method,
        baseline_method=args.baseline_method,
        max_examples_per_category=args.max_examples_per_category,
        phrase_by_case=phrase_by_case,
    )
    outputs = write_failure_audit(result, args.outdir)
    num_cases = int(result.shortcut_summary["n_cases"].sum()) if not result.shortcut_summary.empty else 0
    manifest: dict[str, Any] = {
        "status": "ok",
        "case_gap_csv": str(args.case_gap_csv),
        "region_gap_csv": str(args.region_gap_csv),
        "metrics_csv": str(args.metrics_csv),
        "prepared_inputs_npz": str(args.prepared_inputs_npz) if args.prepared_inputs_npz else "",
        "categories": categories,
        "method": args.method,
        "baseline_method": args.baseline_method,
        "num_audited_cases": num_cases,
        "outputs": outputs,
    }
    manifest_path = args.outdir / "failure_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
