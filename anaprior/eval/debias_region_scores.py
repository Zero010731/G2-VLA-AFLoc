"""Create debiased region score CSVs for Stage C repair diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {"dicom_id", "finding", "region", "score_probability"}


def load_reference_region_stats(path: Path, dataset: str | None = None) -> dict[str, float]:
    table = pd.read_csv(path)
    required = {"region", "mean_score_agnostic"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if dataset is not None:
        if "dataset" not in table.columns:
            raise ValueError(f"{path} missing required column for dataset filtering: dataset")
        table = table[table["dataset"].astype(str) == str(dataset)].copy()
        if table.empty:
            raise ValueError(f"{path} has no rows for reference dataset: {dataset}")
    duplicated_regions = table["region"].astype(str).duplicated(keep=False)
    if duplicated_regions.any():
        duplicates = sorted(table.loc[duplicated_regions, "region"].astype(str).unique().tolist())
        raise ValueError(
            f"Duplicate region rows in {path}: {duplicates}. "
            "Pass --reference-dataset when using a multi-dataset agnostic summary."
        )
    stats: dict[str, float] = {}
    for _, row in table.iterrows():
        region = str(row["region"])
        if not region:
            continue
        stats[region] = float(row["mean_score_agnostic"])
    return stats


def _rescale_group(values: pd.Series, method: str) -> pd.Series:
    if method == "none":
        return values
    if method != "finding_minmax":
        raise ValueError(f"Unsupported rescale method: {method}")
    min_value = float(values.min())
    max_value = float(values.max())
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    if max_value - min_value <= 1e-12:
        return pd.Series(np.full(len(values), 0.5, dtype=float), index=values.index)
    return (values - min_value) / (max_value - min_value)


def debias_scores(
    scores: pd.DataFrame,
    reference_stats: dict[str, float],
    mode: str = "subtract",
    rescale: str = "finding_minmax",
    missing_reference: str = "keep",
) -> pd.DataFrame:
    missing = REQUIRED_SCORE_COLUMNS - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {sorted(missing)}")
    if mode != "subtract":
        raise ValueError(f"Unsupported debias mode: {mode}")
    if missing_reference not in {"keep", "zero"}:
        raise ValueError(f"Unsupported missing_reference policy: {missing_reference}")

    out = scores.copy()
    out["score_probability_original"] = pd.to_numeric(out["score_probability"], errors="coerce")
    out["region_reference_mean"] = out["region"].map(reference_stats)
    if missing_reference == "keep":
        reference = out["region_reference_mean"].fillna(0.0)
    else:
        reference = out["region_reference_mean"].fillna(0.0)
    out["score_debiased_raw"] = out["score_probability_original"] - reference
    out["score_probability"] = (
        out.groupby(["dicom_id", "finding"], group_keys=False)["score_debiased_raw"]
        .apply(lambda values: _rescale_group(values, rescale))
        .astype(float)
    )
    out["score_debias_mode"] = mode
    out["score_rescale"] = rescale
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debias region score probabilities by region-agnostic reference means.")
    parser.add_argument("--score-csv", required=True, type=Path)
    parser.add_argument("--reference-agnostic-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--mode", default="subtract", choices=["subtract"])
    parser.add_argument("--rescale", default="finding_minmax", choices=["finding_minmax", "none"])
    parser.add_argument("--missing-reference", default="keep", choices=["keep", "zero"])
    parser.add_argument(
        "--reference-dataset",
        default=None,
        help="Optional dataset name to select from a multi-dataset agnostic summary, e.g. imagenome_valid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = pd.read_csv(args.score_csv)
    reference_stats = load_reference_region_stats(args.reference_agnostic_csv, dataset=args.reference_dataset)
    out = debias_scores(
        scores=scores,
        reference_stats=reference_stats,
        mode=args.mode,
        rescale=args.rescale,
        missing_reference=args.missing_reference,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    result: dict[str, Any] = {
        "status": "ok",
        "score_csv": str(args.score_csv),
        "reference_agnostic_csv": str(args.reference_agnostic_csv),
        "output_csv": str(args.output_csv),
        "mode": args.mode,
        "rescale": args.rescale,
        "reference_dataset": args.reference_dataset,
        "rows": int(len(out)),
        "regions_with_reference": int(out["region_reference_mean"].notna().sum()),
        "regions_missing_reference": int(out["region_reference_mean"].isna().sum()),
    }
    report_path = args.output_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["report"] = str(report_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
