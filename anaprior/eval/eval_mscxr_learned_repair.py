"""MS-CXR learned repair orchestration for AnaPrior-Loc.

The core functions in this file are deliberately small and testable. They turn
already-prepared MS-CXR heatmaps, region maps, and learned region scores into the
pre-registered repair method heatmap dictionaries. The heavy dataset/model I/O
can be layered on top without changing the experimental definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anaprior.eval.selective_repair import RepairCase, apply_repair_cases


DEFAULT_CANDIDATES = ("Pneumothorax", "Pleural Effusion")


@dataclass(frozen=True)
class LearnedRepairInput:
    case_id: str
    dicom_id: str
    category: str
    finding: str
    heatmap: np.ndarray
    region_maps: np.ndarray
    regions: list[str]
    phrase: str = ""


RegionScoreTable = dict[tuple[str, str], dict[str, float]]


def parse_csv_list(raw: str | None, default: tuple[str, ...] = DEFAULT_CANDIDATES) -> list[str]:
    if raw is None:
        return list(default)
    values = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def load_region_score_table(path: Path) -> RegionScoreTable:
    table: RegionScoreTable = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dicom_id", "region", "finding", "score_probability"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            dicom_id = str(row["dicom_id"]).strip()
            finding = str(row["finding"]).strip()
            region = str(row["region"]).strip()
            if not dicom_id or not finding or not region:
                continue
            key = (dicom_id, finding.lower())
            score = float(row["score_probability"])
            previous = table.setdefault(key, {}).get(region)
            table[key][region] = score if previous is None else max(float(previous), score)
    return table


def region_scores_for_case(
    table: RegionScoreTable,
    dicom_id: str,
    finding: str,
    regions: list[str],
) -> np.ndarray:
    region_scores = table.get((str(dicom_id), str(finding).strip().lower()), {})
    return np.asarray([float(region_scores.get(region, 0.0)) for region in regions], dtype=np.float32)


def _repair_cases_from_inputs(
    inputs: list[LearnedRepairInput],
    region_score_table: RegionScoreTable,
) -> list[RepairCase]:
    cases: list[RepairCase] = []
    for item in inputs:
        cases.append(
            RepairCase(
                case_id=item.case_id,
                category=item.category,
                heatmap=np.asarray(item.heatmap, dtype=np.float32),
                region_maps=np.asarray(item.region_maps, dtype=np.float32),
                region_scores=region_scores_for_case(
                    region_score_table,
                    dicom_id=item.dicom_id,
                    finding=item.finding,
                    regions=item.regions,
                ),
                regions=list(item.regions),
                phrase=item.phrase or item.finding,
            )
        )
    return cases


def _baseline_hmaps(inputs: list[LearnedRepairInput]) -> dict[str, dict[str, np.ndarray | str]]:
    return {
        item.case_id: {
            "hmap": np.asarray(item.heatmap, dtype=np.float32).copy(),
            "learned_repair": "baseline",
        }
        for item in inputs
    }


def _result_to_hmaps(results) -> dict[str, dict[str, np.ndarray | str]]:
    return {
        case_id: {
            "hmap": result.heatmap,
            "learned_repair": result.status,
        }
        for case_id, result in results.items()
    }


def _counter_dict(stats: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {key: {str(k): int(v) for k, v in value.items()} for key, value in stats.items()}


def build_method_hmaps(
    inputs: list[LearnedRepairInput],
    region_score_table: RegionScoreTable,
    candidate_categories: set[str] | list[str] | tuple[str, ...],
    alpha: float,
    seed: int = 0,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray | str]]], dict[str, dict[str, dict[str, int]]]]:
    """Build all pre-registered learned repair method heatmap dictionaries."""

    repair_cases = _repair_cases_from_inputs(inputs, region_score_table)
    method_specs = [
        ("learned_selective", "candidate", "learned"),
        ("disease_gated_learned", "candidate", "disease_gated"),
        ("disease_pooled_learned", "candidate", "disease_pooled"),
        ("phrase_anatomy_dcem", "candidate", "phrase_anatomy"),
        ("all_class_learned", "all", "learned"),
        ("candidate_shuffled", "candidate", "shuffled"),
        ("candidate_uniform", "candidate", "uniform"),
    ]
    hmaps: dict[str, dict[str, dict[str, np.ndarray | str]]] = {"baseline": _baseline_hmaps(inputs)}
    stats: dict[str, dict[str, dict[str, int]]] = {
        "baseline": {
            "repaired_categories": {},
            "skipped_non_candidate": {},
            "missing_region_objects": {},
            "missing_finding_evidence": {},
            "gate_passed_categories": {},
            "gate_bypassed_categories": {},
        }
    }
    for method, repair_scope, score_mode in method_specs:
        results, method_stats = apply_repair_cases(
            repair_cases,
            candidate_categories=candidate_categories,
            alpha=alpha,
            repair_scope=repair_scope,  # type: ignore[arg-type]
            score_mode=score_mode,  # type: ignore[arg-type]
            seed=seed,
        )
        hmaps[method] = _result_to_hmaps(results)
        stats[method] = _counter_dict(method_stats)
    return hmaps, stats


def save_method_hmaps(
    hmaps: dict[str, dict[str, dict[str, np.ndarray | str]]],
    stats: dict[str, dict[str, dict[str, int]]],
    outdir: Path,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for method, method_hmaps in hmaps.items():
        method_dir = outdir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        hmap_path = method_dir / "hmaps.npy"
        np.save(hmap_path, method_hmaps)
        outputs[method] = str(hmap_path)
    stats_path = outdir / "build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["build_stats"] = str(stats_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pre-registered learned repair heatmaps from prepared MS-CXR "
            "repair inputs. Full dataset extraction is intentionally separate."
        )
    )
    parser.add_argument("--prepared-inputs-npz", required=True, type=Path)
    parser.add_argument("--region-score-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--candidate-categories", default="Pneumothorax,Pleural Effusion")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_prepared_inputs(path: Path) -> list[LearnedRepairInput]:
    payload = np.load(path, allow_pickle=True)
    raw_items = payload["items"].tolist()
    inputs = []
    for item in raw_items:
        inputs.append(
            LearnedRepairInput(
                case_id=str(item["case_id"]),
                dicom_id=str(item["dicom_id"]),
                category=str(item["category"]),
                finding=str(item["finding"]),
                heatmap=np.asarray(item["heatmap"], dtype=np.float32),
                region_maps=np.asarray(item["region_maps"], dtype=np.float32),
                regions=list(item["regions"]),
                phrase=str(item.get("phrase", item.get("finding", ""))),
            )
        )
    return inputs


def main() -> int:
    args = parse_args()
    inputs = _load_prepared_inputs(args.prepared_inputs_npz)
    score_table = load_region_score_table(args.region_score_csv)
    candidate_categories = parse_csv_list(args.candidate_categories)
    hmaps, stats = build_method_hmaps(
        inputs=inputs,
        region_score_table=score_table,
        candidate_categories=candidate_categories,
        alpha=args.alpha,
        seed=args.seed,
    )
    outputs = save_method_hmaps(hmaps, stats, args.outdir)
    result: dict[str, Any] = {
        "status": "ok",
        "prepared_inputs_npz": str(args.prepared_inputs_npz),
        "region_score_csv": str(args.region_score_csv),
        "outdir": str(args.outdir),
        "candidate_categories": candidate_categories,
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "num_cases": len(inputs),
        "outputs": outputs,
    }
    summary_path = args.outdir / "learned_repair_build_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
