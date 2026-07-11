"""Diagnose region-score bias in learned region abnormality predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anaprior.eval.analyze_learned_oracle_gap import dataframe_to_markdown


REQUIRED_COLUMNS = {"finding", "region", "score_probability"}


def _parse_score_csv_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        dataset = name.strip()
        score_path = Path(path.strip())
    else:
        score_path = Path(spec.strip())
        dataset = score_path.stem
    if not dataset:
        raise ValueError(f"Empty dataset name in score csv spec: {spec}")
    return dataset, score_path


def load_score_csvs(specs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in specs:
        dataset, path = _parse_score_csv_spec(spec)
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        frame = frame.copy()
        frame["dataset"] = dataset
        frame["score_probability"] = pd.to_numeric(frame["score_probability"], errors="coerce")
        frame["finding"] = frame["finding"].astype(str)
        frame["region"] = frame["region"].astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one --score-csv spec is required")
    return pd.concat(frames, axis=0, ignore_index=True)


def _mechanism_hint(row: pd.Series, bias_threshold: float) -> str:
    finding_rank = int(row["rank_within_finding"])
    agnostic_rank = int(row["rank_agnostic"])
    bias_index = float(row["bias_index"])
    if finding_rank <= 3 and agnostic_rank <= 3:
        return "finding_agnostic_feature_or_pooling_bias"
    if finding_rank <= 3 and bias_index >= bias_threshold:
        return "finding_specific_label_or_cooccurrence_bias"
    if agnostic_rank <= 3 and finding_rank > 3:
        return "global_region_bias_not_selected_by_finding"
    return "no_strong_region_bias"


def summarize_region_score_bias(
    scores: pd.DataFrame,
    top_k: int = 5,
    bias_threshold: float = 0.05,
) -> dict[str, pd.DataFrame]:
    required = {"dataset", *REQUIRED_COLUMNS}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {sorted(missing)}")

    valid = scores.dropna(subset=["score_probability"]).copy()
    finding_region_summary = (
        valid.groupby(["dataset", "finding", "region"], dropna=False)
        .agg(
            n_rows=("score_probability", "size"),
            mean_score=("score_probability", "mean"),
            median_score=("score_probability", "median"),
            std_score=("score_probability", "std"),
        )
        .reset_index()
    )
    finding_region_summary["std_score"] = finding_region_summary["std_score"].fillna(0.0)
    finding_region_summary["rank_within_finding"] = (
        finding_region_summary.groupby(["dataset", "finding"])["mean_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    agnostic_region_summary = (
        valid.groupby(["dataset", "region"], dropna=False)
        .agg(
            n_rows=("score_probability", "size"),
            mean_score_agnostic=("score_probability", "mean"),
            median_score_agnostic=("score_probability", "median"),
        )
        .reset_index()
    )
    agnostic_region_summary["rank_agnostic"] = (
        agnostic_region_summary.groupby(["dataset"])["mean_score_agnostic"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    finding_specific_bias = finding_region_summary.merge(
        agnostic_region_summary[["dataset", "region", "mean_score_agnostic", "rank_agnostic"]],
        on=["dataset", "region"],
        how="left",
    )
    finding_specific_bias["bias_index"] = (
        finding_specific_bias["mean_score"] - finding_specific_bias["mean_score_agnostic"]
    )
    finding_specific_bias["mechanism_hint"] = finding_specific_bias.apply(
        lambda row: _mechanism_hint(row, bias_threshold=bias_threshold),
        axis=1,
    )

    route_mask = np.logical_or.reduce(
        [
            finding_specific_bias["rank_within_finding"].le(int(top_k)).to_numpy(),
            finding_specific_bias["rank_agnostic"].le(int(top_k)).to_numpy(),
            finding_specific_bias["bias_index"].abs().ge(float(bias_threshold)).to_numpy(),
        ]
    )
    top_region_bias_routes = finding_specific_bias[route_mask].copy()
    sort_cols = ["dataset", "finding", "rank_within_finding", "rank_agnostic", "bias_index"]
    top_region_bias_routes = top_region_bias_routes.sort_values(
        sort_cols,
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    return {
        "finding_region_summary": finding_region_summary.sort_values(
            ["dataset", "finding", "rank_within_finding", "region"]
        ).reset_index(drop=True),
        "agnostic_region_summary": agnostic_region_summary.sort_values(
            ["dataset", "rank_agnostic", "region"]
        ).reset_index(drop=True),
        "finding_specific_bias": finding_specific_bias.sort_values(
            ["dataset", "finding", "rank_within_finding", "region"]
        ).reset_index(drop=True),
        "top_region_bias_routes": top_region_bias_routes,
    }


def build_markdown_report(tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> str:
    lines = [
        "# Region Score Bias Diagnosis",
        "",
        "目的：判断 learned region score 的中心区域偏置更像 finding-specific label/co-occurrence，还是 finding-agnostic feature/pooling bias。",
        "",
        f"- score_csv: `{', '.join(args.score_csv)}`",
        f"- top_k: `{args.top_k}`",
        f"- bias_threshold: `{args.bias_threshold}`",
        "",
        "## Top Region Bias Routes",
        "",
        dataframe_to_markdown(tables["top_region_bias_routes"].head(50)),
        "",
        "## Finding-Agnostic Region Ranking",
        "",
        dataframe_to_markdown(tables["agnostic_region_summary"].head(50)),
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize learned region score bias by finding and dataset.")
    parser.add_argument(
        "--score-csv",
        action="append",
        required=True,
        help="Named score CSV spec, e.g. mscxr=outputs/.../mscxr_region_scores.csv. Can be repeated.",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bias-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = load_score_csvs(args.score_csv)
    tables = summarize_region_score_bias(scores, top_k=args.top_k, bias_threshold=args.bias_threshold)

    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, table in tables.items():
        path = args.outdir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = path
    report_path = args.outdir / "region_score_bias_report.md"
    report_path.write_text(build_markdown_report(tables, args), encoding="utf-8")
    outputs["markdown"] = report_path

    result: dict[str, Any] = {
        "status": "ok",
        "score_csv": args.score_csv,
        "outdir": str(args.outdir),
        "rows": int(len(scores)),
        "datasets": sorted(scores["dataset"].dropna().astype(str).unique().tolist()),
        "findings": sorted(scores["finding"].dropna().astype(str).unique().tolist()),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    summary_path = args.outdir / "region_score_bias_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["outputs"]["summary"] = str(summary_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
