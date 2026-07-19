"""Build the fixed AFLoc anchor Phase 0 readiness decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


HARM_FLOOR = -0.005
ANCHOR_COMPARISON = "afloc_anchor_vs_baseline"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _path_from_summary(
    summary: dict[str, Any],
    group: str,
    name: str,
    fallback: Path,
) -> Path:
    value = (summary.get(group) or {}).get(name)
    return Path(value) if value else fallback


def _select_macro_cnr(bootstrap: pd.DataFrame) -> dict[str, float] | None:
    required = {"comparison", "split", "macro_scope", "metric", "mean_delta", "ci_low"}
    if bootstrap.empty or not required.issubset(bootstrap.columns):
        return None
    rows = bootstrap[
        (bootstrap["comparison"] == ANCHOR_COMPARISON)
        & (bootstrap["split"] == "test")
        & (bootstrap["macro_scope"] == "macro_all")
        & (bootstrap["metric"] == "cnr")
    ]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if pd.isna(row["mean_delta"]) or pd.isna(row["ci_low"]):
        return None
    return {
        "mean_delta": float(row["mean_delta"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row.get("ci_high", float("nan"))),
    }


def _absolute_anchor_cnr(per_case: pd.DataFrame) -> float | None:
    if per_case.empty or "cnr" not in per_case.columns:
        return None
    rows = per_case
    if "split" in rows.columns:
        rows = rows[rows["split"] == "test"]
    values = pd.to_numeric(rows["cnr"], errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No available rows._"
    columns = [name for name in columns if name in frame.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, record in frame.loc[:, columns].iterrows():
        values = []
        for name in columns:
            value = record[name]
            values.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_anchor_phase0_report(
    *,
    anchor_summary_path: Path,
    metrics_summary_path: Path,
    outdir: Path,
) -> dict[str, Any]:
    anchor_summary = _read_json(anchor_summary_path)
    metrics_summary = _read_json(metrics_summary_path)
    metrics_dir = metrics_summary_path.parent
    bootstrap_path = _path_from_summary(
        metrics_summary,
        "outputs",
        "bootstrap_summary",
        metrics_dir / "bootstrap_summary.csv",
    )
    per_class_path = _path_from_summary(
        metrics_summary,
        "outputs",
        "per_class_bootstrap_summary",
        metrics_dir / "per_class_bootstrap_summary.csv",
    )
    anchor_method = (metrics_summary.get("method_outputs") or {}).get("afloc_anchor") or {}
    per_case_path = Path(
        anchor_method.get("per_case_metric_csv", metrics_dir / "afloc_anchor" / "per_case_metric.csv")
    )
    bootstrap = _read_csv(bootstrap_path)
    per_class = _read_csv(per_class_path)
    per_case = _read_csv(per_case_path)
    macro = _select_macro_cnr(bootstrap)
    absolute_cnr = _absolute_anchor_cnr(per_case)

    anchor_cases = int(anchor_summary.get("num_cases", 0))
    anchor_hmaps = int(anchor_summary.get("num_hmaps", 0))
    category_rows = int(metrics_summary.get("num_eval_rows_after_category_filter", 0))
    common_rows = int(metrics_summary.get("num_eval_rows_after_hmap_filter", 0))
    coverage_complete = (
        anchor_cases > 0
        and anchor_cases == anchor_hmaps
        and category_rows > 0
        and common_rows == category_rows
    )
    zero_variance_count = int(anchor_summary.get("zero_variance_count", 0))
    reasons: list[str] = []
    incomplete = False
    if not coverage_complete:
        reasons.append("incomplete_heatmap_coverage")
        incomplete = True
    if macro is None:
        reasons.append("missing_anchor_vs_baseline_macro_cnr")
        incomplete = True
    if absolute_cnr is None:
        reasons.append("missing_anchor_absolute_cnr")
        incomplete = True

    if incomplete:
        status = "incomplete_evidence"
    else:
        assert macro is not None and absolute_cnr is not None
        if zero_variance_count > 0:
            reasons.append("zero_variance_anchor_heatmaps")
        if macro["ci_low"] < HARM_FLOOR:
            reasons.append("macro_cnr_ci_low_below_harm_floor")
        if absolute_cnr <= 0.0:
            reasons.append("nonpositive_anchor_absolute_pooled_cnr")
        status = (
            "anchor_ready_for_bounded_refinement"
            if not reasons
            else "anchor_not_ready_improve_construction"
        )

    decision = {
        "status": status,
        "reasons": reasons,
        "coverage_complete": bool(coverage_complete),
        "anchor_num_cases": anchor_cases,
        "anchor_num_hmaps": anchor_hmaps,
        "scoring_num_category_rows": category_rows,
        "scoring_num_common_rows": common_rows,
        "zero_variance_count": zero_variance_count,
        "mean_anchor_confidence": anchor_summary.get("mean_anchor_confidence"),
        "macro_cnr_delta_vs_baseline": None if macro is None else macro["mean_delta"],
        "macro_cnr_ci_low_vs_baseline": None if macro is None else macro["ci_low"],
        "macro_cnr_ci_high_vs_baseline": None if macro is None else macro["ci_high"],
        "macro_cnr_harm_floor": HARM_FLOOR,
        "anchor_absolute_pooled_cnr": absolute_cnr,
        "anchor_summary": str(anchor_summary_path),
        "metrics_summary": str(metrics_summary_path),
    }

    anchor_comparisons = bootstrap[
        bootstrap.get("comparison", pd.Series(dtype=str)).astype(str).str.startswith("afloc_anchor_vs_")
    ] if not bootstrap.empty and "comparison" in bootstrap.columns else pd.DataFrame()
    anchor_per_class = per_class[
        per_class.get("comparison", pd.Series(dtype=str)) == ANCHOR_COMPARISON
    ] if not per_class.empty and "comparison" in per_class.columns else pd.DataFrame()
    report = "\n".join(
        [
            "# AFLoc Anchor Phase 0 Report",
            "",
            f"- status: `{status}`",
            f"- reasons: `{', '.join(reasons)}`",
            f"- coverage_complete: `{coverage_complete}`",
            f"- anchor_absolute_pooled_cnr: `{absolute_cnr}`",
            f"- macro_cnr_delta_vs_baseline: `{decision['macro_cnr_delta_vs_baseline']}`",
            f"- macro_cnr_ci_low_vs_baseline: `{decision['macro_cnr_ci_low_vs_baseline']}`",
            "",
            "## Bootstrap Comparisons",
            "",
            _markdown_table(
                anchor_comparisons,
                ["comparison", "macro_scope", "metric", "mean_delta", "ci_low", "ci_high", "n", "n_cases"],
            ),
            "",
            "## Per-Class Anchor vs AFLoc",
            "",
            _markdown_table(
                anchor_per_class,
                ["category", "metric", "mean_delta", "ci_low", "ci_high", "n"],
            ),
            "",
        ]
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "anchor_phase0_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (outdir / "anchor_phase0_report.md").write_text(report, encoding="utf-8")
    return decision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report AFLoc anchor Phase 0 readiness.")
    parser.add_argument("--anchor-summary", required=True, type=Path)
    parser.add_argument("--metrics-summary", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    decision = build_anchor_phase0_report(
        anchor_summary_path=args.anchor_summary,
        metrics_summary_path=args.metrics_summary,
        outdir=args.outdir,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
