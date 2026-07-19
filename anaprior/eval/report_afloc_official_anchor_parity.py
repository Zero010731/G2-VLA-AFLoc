"""Compare regenerated official AFLoc heatmaps with a saved AFLoc baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _heatmap(payload: Any) -> np.ndarray:
    value = payload.get("hmap") if isinstance(payload, Mapping) else payload
    return np.asarray(value, dtype=np.float64)


def compare_heatmap_sets(
    generated: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    min_matched_cases: int = 1,
    min_coverage: float = 0.99,
    min_pearson_r: float = 0.999,
    max_mean_mae: float = 1.0e-5,
) -> dict[str, Any]:
    correlations: list[float] = []
    maes: list[float] = []
    max_errors: list[float] = []
    missing: list[str] = []
    shape_mismatches = 0

    for generated_key, payload in generated.items():
        metadata = payload if isinstance(payload, Mapping) else {}
        lookup_key = str(metadata.get("hmap_key") or generated_key)
        if lookup_key not in reference:
            missing.append(lookup_key)
            continue
        actual = _heatmap(payload)
        expected = _heatmap(reference[lookup_key])
        if actual.shape != expected.shape:
            shape_mismatches += 1
            continue
        finite = np.isfinite(actual) & np.isfinite(expected)
        if not finite.any():
            continue
        left = actual[finite]
        right = expected[finite]
        mae = float(np.mean(np.abs(left - right)))
        max_error = float(np.max(np.abs(left - right)))
        if float(left.std()) == 0.0 or float(right.std()) == 0.0:
            correlation = 1.0 if np.array_equal(left, right) else 0.0
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
            if abs(correlation - 1.0) <= 1.0e-12:
                correlation = 1.0
        correlations.append(correlation)
        maes.append(mae)
        max_errors.append(max_error)

    total = len(generated)
    matched = len(correlations)
    coverage = float(matched / total) if total else 0.0
    mean_r = float(np.mean(correlations)) if correlations else 0.0
    mean_mae = float(np.mean(maes)) if maes else float("inf")
    result = {
        "status": "ok",
        "generated_cases": total,
        "reference_entries": len(reference),
        "matched_cases": matched,
        "coverage": coverage,
        "mean_pearson_r": mean_r,
        "min_pearson_r": float(np.min(correlations)) if correlations else 0.0,
        "mean_mae": mean_mae,
        "max_abs_error": float(np.max(max_errors)) if max_errors else float("inf"),
        "shape_mismatch_count": shape_mismatches,
        "missing_reference_keys": missing,
        "thresholds": {
            "min_matched_cases": min_matched_cases,
            "min_coverage": min_coverage,
            "min_pearson_r": min_pearson_r,
            "max_mean_mae": max_mean_mae,
        },
    }
    result["parity_passed"] = bool(
        matched >= min_matched_cases
        and coverage >= min_coverage
        and mean_r >= min_pearson_r
        and mean_mae <= max_mean_mae
        and shape_mismatches == 0
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-hmaps", required=True, type=Path)
    parser.add_argument("--reference-hmaps", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-matched-cases", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generated = np.load(args.generated_hmaps, allow_pickle=True).item()
    reference = np.load(args.reference_hmaps, allow_pickle=True).item()
    result = compare_heatmap_sets(
        generated,
        reference,
        min_matched_cases=args.min_matched_cases,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
