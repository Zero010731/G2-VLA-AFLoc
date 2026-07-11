"""MS-CXR oracle region-recoverability profile.

This script is a diagnostic, not a deployable method.  It asks a narrow
question: if the region evidence were perfectly aligned with the MS-CXR lesion
mask, which pathologies can AFLoc heatmaps be repaired by region-level
anatomical evidence?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anaprior.eval.analyze_learned_oracle_gap import (
    compute_region_gt_overlaps,
    dataframe_to_markdown,
    load_mscxr_gt_masks_by_case,
)
from anaprior.eval.eval_mscxr_learned_repair import LearnedRepairInput
from anaprior.eval.score_mscxr_learned_repair_metrics import (
    add_method_metadata,
    coerce_eval_dataframe,
    stable_int,
    summarize_comparison,
)
from anaprior.eval.selective_repair import RepairCase, apply_repair_cases


METRICS = ("cnr", "iou", "dice")
DEFAULT_METHODS = ("baseline", "oracle_repair", "oracle_shuffled", "oracle_uniform")
DEFAULT_COMPARISONS = (
    ("oracle_repair_vs_baseline", "oracle_repair", "baseline"),
    ("oracle_shuffled_vs_baseline", "oracle_shuffled", "baseline"),
    ("oracle_uniform_vs_baseline", "oracle_uniform", "baseline"),
    ("oracle_repair_vs_oracle_shuffled", "oracle_repair", "oracle_shuffled"),
    ("oracle_repair_vs_oracle_uniform", "oracle_repair", "oracle_uniform"),
)


def _load_prepared_inputs(path: Path) -> list[LearnedRepairInput]:
    payload = np.load(path, allow_pickle=True)
    raw_items = payload["items"].tolist()
    inputs: list[LearnedRepairInput] = []
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
            )
        )
    return inputs


def _baseline_hmaps(inputs: list[LearnedRepairInput]) -> dict[str, dict[str, Any]]:
    return {
        item.case_id: {
            "hmap": np.asarray(item.heatmap, dtype=np.float32).copy(),
            "oracle_repair": "baseline",
        }
        for item in inputs
    }


def _counter_dict(stats: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {key: {str(k): int(v) for k, v in value.items()} for key, value in stats.items()}


def _result_to_hmaps(results) -> dict[str, dict[str, Any]]:
    return {
        case_id: {
            "hmap": result.heatmap,
            "oracle_repair": result.status,
        }
        for case_id, result in results.items()
    }


def _oracle_repair_cases_and_table(
    inputs: list[LearnedRepairInput],
    gt_masks_by_case: dict[str, np.ndarray],
) -> tuple[list[RepairCase], pd.DataFrame, dict[str, dict[str, int]]]:
    cases: list[RepairCase] = []
    rows: list[dict[str, Any]] = []
    skipped = {
        "missing_gt_mask": Counter(),
        "empty_region_objects": Counter(),
    }
    for item in inputs:
        gt_mask = gt_masks_by_case.get(item.case_id)
        if gt_mask is None:
            skipped["missing_gt_mask"][item.category] += 1
            continue
        region_maps = np.asarray(item.region_maps, dtype=np.float32)
        if region_maps.ndim != 3 or float(np.nansum(region_maps)) <= 1e-6:
            skipped["empty_region_objects"][item.category] += 1
            continue
        oracle_scores = compute_region_gt_overlaps(region_maps, np.asarray(gt_mask, dtype=np.float32))
        cases.append(
            RepairCase(
                case_id=item.case_id,
                category=item.category,
                heatmap=np.asarray(item.heatmap, dtype=np.float32),
                region_maps=region_maps,
                region_scores=oracle_scores,
            )
        )
        top_region = item.regions[int(np.nanargmax(oracle_scores))] if len(item.regions) else ""
        for region, score in zip(item.regions, oracle_scores):
            rows.append(
                {
                    "case_id": item.case_id,
                    "dicom_id": item.dicom_id,
                    "category": item.category,
                    "finding": item.finding,
                    "region": region,
                    "oracle_score": float(score),
                    "is_oracle_top_region": region == top_region,
                }
            )
    return cases, pd.DataFrame(rows), _counter_dict(skipped)


def build_oracle_recoverability_hmaps(
    inputs: list[LearnedRepairInput],
    gt_masks_by_case: dict[str, np.ndarray],
    alpha: float,
    seed: int = 0,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, int]]], pd.DataFrame]:
    """Build baseline and oracle-region repair heatmaps for all classes."""

    repair_cases, oracle_table, skipped_stats = _oracle_repair_cases_and_table(inputs, gt_masks_by_case)
    all_categories = sorted({case.category for case in repair_cases})
    hmaps: dict[str, dict[str, dict[str, Any]]] = {"baseline": _baseline_hmaps(inputs)}
    stats: dict[str, dict[str, dict[str, int]]] = {
        "baseline": {
            "repaired_categories": {},
            "skipped_non_candidate": {},
            "missing_region_objects": {},
            "missing_finding_evidence": {},
            **skipped_stats,
        }
    }
    method_specs = [
        ("oracle_repair", "learned"),
        ("oracle_shuffled", "shuffled"),
        ("oracle_uniform", "uniform"),
    ]
    for method, score_mode in method_specs:
        results, method_stats = apply_repair_cases(
            repair_cases,
            candidate_categories=all_categories,
            alpha=alpha,
            repair_scope="all",
            score_mode=score_mode,
            seed=seed,
        )
        hmaps[method] = _result_to_hmaps(results)
        stats[method] = {**_counter_dict(method_stats), **skipped_stats}
    return hmaps, stats, oracle_table


def save_oracle_hmaps(
    hmaps: dict[str, dict[str, dict[str, Any]]],
    stats: dict[str, dict[str, dict[str, int]]],
    oracle_table: pd.DataFrame,
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
    oracle_path = outdir / "oracle_region_scores.csv"
    oracle_table.to_csv(oracle_path, index=False)
    stats_path = outdir / "oracle_build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["oracle_region_scores"] = str(oracle_path)
    outputs["build_stats"] = str(stats_path)
    return outputs


def _case_id_from_row(row: pd.Series) -> str:
    return str(row["path"]) + str(row["label_text"])


def _filter_eval_data_to_hmaps(data: pd.DataFrame, hmaps: dict[str, dict[str, Any]]) -> pd.DataFrame:
    available = set(hmaps.keys())
    frame = data.copy()
    frame["_case_id"] = [_case_id_from_row(row) for _, row in frame.iterrows()]
    frame = frame[frame["_case_id"].isin(available)].copy()
    return frame.drop(columns=["_case_id"])


def build_recoverability_profile(
    per_class_summary: pd.DataFrame,
    effect_floor: float = 0.02,
    split: str = "test",
) -> pd.DataFrame:
    """Classify each pathology from oracle repair paired-bootstrap rows."""

    rows: list[dict[str, Any]] = []
    if per_class_summary.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "n",
                "oracle_delta_cnr",
                "oracle_cnr_ci_low",
                "oracle_delta_iou",
                "oracle_iou_ci_low",
                "oracle_delta_dice",
                "oracle_dice_ci_low",
                "verdict",
                "reason",
            ]
        )
    subset = per_class_summary[
        (per_class_summary["comparison"] == "oracle_repair_vs_baseline")
        & (per_class_summary["split"] == split)
    ]
    for category, group in sorted(subset.groupby("category"), key=lambda item: str(item[0])):
        record: dict[str, Any] = {"category": category, "n": 0}
        for metric in METRICS:
            metric_rows = group[group["metric"] == metric]
            if metric_rows.empty:
                record[f"oracle_delta_{metric}"] = np.nan
                record[f"oracle_{metric}_ci_low"] = np.nan
                record[f"oracle_{metric}_ci_high"] = np.nan
                continue
            row = metric_rows.iloc[0]
            record["n"] = int(row.get("n", record["n"]))
            record[f"oracle_delta_{metric}"] = float(row["mean_delta"])
            record[f"oracle_{metric}_ci_low"] = float(row["ci_low"])
            record[f"oracle_{metric}_ci_high"] = float(row["ci_high"])

        cnr = float(record.get("oracle_delta_cnr", np.nan))
        cnr_low = float(record.get("oracle_cnr_ci_low", np.nan))
        iou = float(record.get("oracle_delta_iou", np.nan))
        iou_low = float(record.get("oracle_iou_ci_low", np.nan))
        dice = float(record.get("oracle_delta_dice", np.nan))
        dice_low = float(record.get("oracle_dice_ci_low", np.nan))
        if cnr >= float(effect_floor) and cnr_low > 0:
            verdict = "region_recoverable"
            reason = "oracle_cnr_effect_floor_and_ci_met"
        elif iou_low > 0 and dice_low > 0:
            verdict = "overlap_recoverable_cnr_limited"
            reason = "threshold_overlap_improves_but_cnr_gate_not_met"
        elif cnr > 0 or iou > 0 or dice > 0:
            verdict = "weak_or_uncertain"
            reason = "positive_direction_without_stable_oracle_gate"
        else:
            verdict = "not_region_recoverable"
            reason = "oracle_region_signal_does_not_improve_metrics"
        record["verdict"] = verdict
        record["reason"] = reason
        rows.append(record)
    return pd.DataFrame(rows)


def score_oracle_recoverability(
    hmaps_root: Path,
    outdir: Path,
    methods: list[str],
    dataset: str = "MS_CXR_CLS",
    val_fraction: float = 0.3,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
    margin: bool = False,
    max_cases: int | None = None,
    effect_floor: float = 0.02,
) -> dict[str, Any]:
    from localization.datasets import load_data
    from anaprior.eval.score_mscxr_learned_repair_metrics import evaluate_hmaps

    outdir.mkdir(parents=True, exist_ok=True)
    data = coerce_eval_dataframe(load_data(dataset=dataset), max_cases=max_cases)

    per_case_by_method: dict[str, pd.DataFrame] = {}
    metric_outputs: dict[str, dict[str, str]] = {}
    for method in methods:
        hmap_path = hmaps_root / method / "hmaps.npy"
        if not hmap_path.exists():
            raise FileNotFoundError(hmap_path)
        hmaps = np.load(hmap_path, allow_pickle=True).item()
        method_dir = outdir / method
        eval_data = _filter_eval_data_to_hmaps(data, hmaps)
        if eval_data.empty:
            example_eval_keys = [_case_id_from_row(row) for _, row in data.head(3).iterrows()]
            example_hmap_keys = [str(key) for key in list(hmaps.keys())[:3]]
            raise ValueError(
                "No evaluation rows matched hmap keys "
                f"for method={method}, dataset={dataset}, eval_rows={len(data)}, hmap_keys={len(hmaps)}. "
                f"Example eval keys={example_eval_keys}; example hmap keys={example_hmap_keys}. "
                "Use the same dataset protocol that created the prepared inputs, e.g. DATASET=MS_CXR for phrase keys."
            )
        metric_df, per_case = evaluate_hmaps(eval_data, hmaps, dataset=dataset, save_dir=method_dir, margin=margin)
        per_case = add_method_metadata(per_case, method=method, val_fraction=val_fraction, seed=seed)
        per_case.to_csv(method_dir / "per_case_metric.csv", index=False)
        metric_df.to_csv(method_dir / "metric.csv", index=False)
        per_case_by_method[method] = per_case
        metric_outputs[method] = {
            "metric_csv": str(method_dir / "metric.csv"),
            "per_case_metric_csv": str(method_dir / "per_case_metric.csv"),
        }

    categories = sorted(set(pd.concat(list(per_case_by_method.values()), axis=0)["category"].astype(str)))
    delta_frames = []
    summary_frames = []
    per_class_frames = []
    comparison_specs = [spec for spec in DEFAULT_COMPARISONS if spec[1] in per_case_by_method and spec[2] in per_case_by_method]
    for split in ["val", "test"]:
        for comparison, method_a, method_b in comparison_specs:
            deltas, summary, per_class = summarize_comparison(
                method_per_case=per_case_by_method[method_a],
                baseline_per_case=per_case_by_method[method_b],
                comparison=comparison,
                candidate_categories=categories,
                split=split,
                n_boot=bootstrap_replicates,
                seed=seed + stable_int(f"oracle:{comparison}:{split}") % 100000,
            )
            delta_frames.append(deltas)
            summary_frames.append(summary)
            per_class_frames.append(per_class)

    delta_table = pd.concat(delta_frames, axis=0, ignore_index=True) if delta_frames else pd.DataFrame()
    bootstrap_summary = pd.concat(summary_frames, axis=0, ignore_index=True) if summary_frames else pd.DataFrame()
    per_class_summary = pd.concat(per_class_frames, axis=0, ignore_index=True) if per_class_frames else pd.DataFrame()
    profile = build_recoverability_profile(per_class_summary, effect_floor=effect_floor, split="test")

    all_per_case = pd.concat(list(per_case_by_method.values()), axis=0, ignore_index=True)
    paths = {
        "per_case_metrics": outdir / "all_per_case_metrics.csv",
        "delta_table": outdir / "delta_table.csv",
        "bootstrap_summary": outdir / "bootstrap_summary.csv",
        "per_class_bootstrap_summary": outdir / "per_class_bootstrap_summary.csv",
        "recoverability_profile": outdir / "recoverability_profile.csv",
        "markdown": outdir / "oracle_profile_report.md",
        "summary": outdir / "oracle_recoverability_summary.json",
    }
    all_per_case.to_csv(paths["per_case_metrics"], index=False)
    delta_table.to_csv(paths["delta_table"], index=False)
    bootstrap_summary.to_csv(paths["bootstrap_summary"], index=False)
    per_class_summary.to_csv(paths["per_class_bootstrap_summary"], index=False)
    profile.to_csv(paths["recoverability_profile"], index=False)
    paths["markdown"].write_text(build_markdown_report(profile, bootstrap_summary, per_class_summary), encoding="utf-8")

    result: dict[str, Any] = {
        "status": "ok",
        "hmaps_root": str(hmaps_root),
        "outdir": str(outdir),
        "dataset": dataset,
        "methods": methods,
        "categories": categories,
        "effect_floor": float(effect_floor),
        "num_eval_rows": int(data.shape[0]),
        "method_outputs": metric_outputs,
        "verdict_counts": profile["verdict"].value_counts().to_dict() if not profile.empty else {},
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def build_markdown_report(
    profile: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    per_class_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Stage D Oracle Region-Recoverability Profile",
        "",
        "定义：oracle region evidence 使用 MS-CXR GT mask 在每个 Chest ImaGenome region map 上的覆盖比例，作为 region repair 的诊断上界。",
        "",
        "## Recoverability Profile",
        "",
        dataframe_to_markdown(profile),
        "",
        "## Test Macro Summary",
        "",
    ]
    if bootstrap_summary.empty:
        lines.append("No summary rows.")
    else:
        rows = bootstrap_summary[
            (bootstrap_summary["split"] == "test")
            & (bootstrap_summary["comparison"].isin(["oracle_repair_vs_baseline", "oracle_repair_vs_oracle_shuffled"]))
            & (bootstrap_summary["metric"] == "cnr")
        ].copy()
        lines.append(dataframe_to_markdown(rows))
    lines.extend(["", "## Per-Class Test Deltas", ""])
    if per_class_summary.empty:
        lines.append("No per-class rows.")
    else:
        rows = per_class_summary[
            (per_class_summary["split"] == "test")
            & (per_class_summary["comparison"] == "oracle_repair_vs_baseline")
        ].copy()
        lines.append(dataframe_to_markdown(rows))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and score an 8-class oracle region-recoverability profile.")
    parser.add_argument("--prepared-inputs-npz", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="MS_CXR_CLS")
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--margin", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--effect-floor", type=float, default=0.02)
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = _load_prepared_inputs(args.prepared_inputs_npz)
    if args.max_cases is not None:
        inputs = inputs[: args.max_cases]
    gt_masks = load_mscxr_gt_masks_by_case(max_cases=args.max_cases, dataset=args.dataset)
    hmaps, stats, oracle_table = build_oracle_recoverability_hmaps(
        inputs=inputs,
        gt_masks_by_case=gt_masks,
        alpha=args.alpha,
        seed=args.seed,
    )
    hmaps_root = args.outdir / "oracle_recoverability_hmaps"
    build_outputs = save_oracle_hmaps(hmaps, stats, oracle_table, hmaps_root)
    result: dict[str, Any] = {
        "status": "ok",
        "prepared_inputs_npz": str(args.prepared_inputs_npz),
        "outdir": str(args.outdir),
        "alpha": float(args.alpha),
        "num_inputs": len(inputs),
        "hmaps_root": str(hmaps_root),
        "build_outputs": build_outputs,
    }
    if not args.build_only:
        score_result = score_oracle_recoverability(
            hmaps_root=hmaps_root,
            outdir=args.outdir / "oracle_recoverability_metrics",
            methods=list(DEFAULT_METHODS),
            dataset=args.dataset,
            val_fraction=args.val_fraction,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            margin=args.margin,
            max_cases=args.max_cases,
            effect_floor=args.effect_floor,
        )
        result["score_result"] = score_result
    summary_path = args.outdir / "oracle_recoverability_run_summary.json"
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["summary"] = str(summary_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
