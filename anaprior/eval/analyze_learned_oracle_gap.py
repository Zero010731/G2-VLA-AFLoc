"""Analyze the gap between learned region evidence and oracle localization evidence.

This diagnostic is intentionally post-hoc and non-training.  It compares the
learned region abnormality scores used by Stage C against an oracle signal
computed from MS-CXR ground-truth masks: how much of the lesion mask falls inside
each Chest ImaGenome-derived region map.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from anaprior.eval.eval_mscxr_learned_repair import (
    RegionScoreTable,
    load_region_score_table,
    parse_csv_list,
    region_scores_for_case,
)
from anaprior.eval.disease_conditioned_gate import apply_disease_specific_pooling
from anaprior.eval.phrase_anatomy_router import apply_phrase_anatomy_router
from anaprior.eval.score_mscxr_learned_repair_metrics import coerce_eval_dataframe


DEFAULT_CANDIDATES = ("Pneumothorax", "Pleural Effusion")
METRICS = ("cnr", "iou", "dice")


@dataclass(frozen=True)
class GapInput:
    case_id: str
    dicom_id: str
    category: str
    finding: str
    region_maps: np.ndarray
    regions: list[str]
    phrase: str = ""


def compute_region_gt_overlaps(region_maps: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Return lesion-mass overlap per region.

    The value for region k is:

        sum(region_map_k * gt_mask) / sum(gt_mask)

    So values sum to approximately 1 when region maps cover the whole lesion.
    This answers: "where does the annotated lesion live in region space?"
    """

    maps = np.asarray(region_maps, dtype=np.float32)
    mask = np.asarray(gt_mask, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError(f"region_maps must have shape [K,H,W], got {maps.shape}")
    if mask.shape != maps.shape[1:]:
        raise ValueError(f"gt_mask shape {mask.shape} does not match region maps {maps.shape[1:]}")
    mask = (mask > 0).astype(np.float32)
    lesion_mass = float(mask.sum())
    if lesion_mass <= 0:
        return np.zeros((maps.shape[0],), dtype=np.float32)
    return (maps * mask[None, :, :]).sum(axis=(1, 2)).astype(np.float32) / lesion_mass


def _load_prepared_inputs(path: Path) -> list[GapInput]:
    payload = np.load(path, allow_pickle=True)
    raw_items = payload["items"].tolist()
    inputs: list[GapInput] = []
    for item in raw_items:
        inputs.append(
            GapInput(
                case_id=str(item["case_id"]),
                dicom_id=str(item["dicom_id"]),
                category=str(item["category"]),
                finding=str(item["finding"]),
                region_maps=np.asarray(item["region_maps"], dtype=np.float32),
                regions=list(item["regions"]),
                phrase=str(item.get("phrase", item.get("finding", ""))),
            )
        )
    return inputs


def evidence_scores_for_gap(
    item: GapInput,
    score_table: RegionScoreTable,
    evidence_mode: str = "learned",
) -> np.ndarray:
    """Return region evidence scores used for learned-vs-oracle gap analysis."""

    raw_scores = region_scores_for_case(score_table, item.dicom_id, item.finding, item.regions)
    mode = str(evidence_mode).strip().lower()
    if mode == "learned":
        return raw_scores
    if mode == "disease_pooled":
        return apply_disease_specific_pooling(item.category, item.regions, raw_scores).scores
    if mode == "phrase_anatomy":
        return apply_phrase_anatomy_router(item.category, item.phrase, item.regions, raw_scores).scores
    raise ValueError("evidence_mode must be 'learned', 'disease_pooled', or 'phrase_anatomy'")


def _case_id_from_data_row(row: pd.Series) -> str:
    return str(row["path"]) + str(row["label_text"])


def load_mscxr_gt_masks_by_case(max_cases: int | None = None, dataset: str = "MS_CXR_CLS") -> dict[str, np.ndarray]:
    from localization.datasets import load_data

    data = coerce_eval_dataframe(load_data(dataset=dataset), max_cases=max_cases)
    return {
        _case_id_from_data_row(row): np.asarray(row["gtmasks"], dtype=np.float32)
        for _, row in data.iterrows()
    }


def _metric_delta_table(
    per_case_metrics: pd.DataFrame,
    method: str,
    baseline_method: str,
) -> pd.DataFrame:
    if per_case_metrics.empty:
        return pd.DataFrame(columns=["case_id", "split", "delta_cnr", "delta_iou", "delta_dice"])
    required = {"case_id", "method", *METRICS}
    missing = required - set(per_case_metrics.columns)
    if missing:
        raise ValueError(f"per_case_metrics missing required columns: {sorted(missing)}")

    left = per_case_metrics[per_case_metrics["method"] == method].copy()
    right = per_case_metrics[per_case_metrics["method"] == baseline_method].copy()
    keep_cols = ["case_id", *METRICS]
    optional_cols = [col for col in ["split", "category"] if col in left.columns and col not in keep_cols]
    merged = left[keep_cols + optional_cols].merge(
        right[keep_cols],
        on="case_id",
        suffixes=("_method", "_baseline"),
        how="inner",
    )
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_method"].astype(float) - merged[f"{metric}_baseline"].astype(float)
    out_cols = ["case_id", *optional_cols, *(f"delta_{metric}" for metric in METRICS)]
    return merged[out_cols].copy()


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return float("nan")
    value = valid["x"].corr(valid["y"], method=method)
    return float(value) if pd.notna(value) else float("nan")


def _top_region(values: np.ndarray, regions: list[str]) -> tuple[str, float, int]:
    if len(values) == 0:
        return "", float("nan"), -1
    idx = int(np.nanargmax(values))
    return str(regions[idx]), float(values[idx]), idx


def _score_table_has_case_finding(score_table: RegionScoreTable, dicom_id: str, finding: str) -> bool:
    return (str(dicom_id), str(finding).strip().lower()) in score_table


def _failure_mode(
    top1_hit: bool,
    oracle_top_overlap: float,
    learned_top_oracle_overlap: float,
    delta_cnr: float | None,
    oracle_overlap_floor: float,
) -> str:
    delta = 0.0 if delta_cnr is None or pd.isna(delta_cnr) else float(delta_cnr)
    if oracle_top_overlap < oracle_overlap_floor:
        return "weak_oracle_region_signal"
    if not top1_hit and learned_top_oracle_overlap < oracle_overlap_floor and delta < 0:
        return "oracle_signal_learned_miss"
    if top1_hit and delta < 0:
        return "learned_matches_but_repair_hurts"
    if top1_hit and delta >= 0:
        return "learned_matches_and_repair_helps"
    return "partial_region_mismatch"


def build_gap_tables(
    inputs: list[GapInput],
    gt_masks_by_case: dict[str, np.ndarray],
    score_table: RegionScoreTable,
    per_case_metrics: pd.DataFrame,
    method: str = "learned_selective",
    baseline_method: str = "baseline",
    oracle_overlap_floor: float = 0.25,
    evidence_mode: str = "learned",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build region-level, case-level, and per-class gap tables."""

    delta_table = _metric_delta_table(per_case_metrics, method=method, baseline_method=baseline_method)
    deltas_by_case = {str(row["case_id"]): row for _, row in delta_table.iterrows()}
    region_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for item in inputs:
        gt_mask = gt_masks_by_case.get(item.case_id)
        if gt_mask is None:
            continue
        has_learned_evidence = _score_table_has_case_finding(score_table, item.dicom_id, item.finding)
        learned_scores = evidence_scores_for_gap(item, score_table, evidence_mode=evidence_mode)
        oracle_overlaps = compute_region_gt_overlaps(item.region_maps, gt_mask)
        if has_learned_evidence:
            learned_top_region, learned_top_score, learned_top_idx = _top_region(learned_scores, item.regions)
        else:
            learned_scores = np.full((len(item.regions),), np.nan, dtype=np.float32)
            learned_top_region = "missing_learned_evidence"
            learned_top_score = float("nan")
            learned_top_idx = -1
        oracle_top_region, oracle_top_overlap, oracle_top_idx = _top_region(oracle_overlaps, item.regions)
        learned_top_oracle_overlap = (
            float(oracle_overlaps[learned_top_idx]) if learned_top_idx >= 0 else float("nan")
        )
        oracle_top_learned_score = (
            float(learned_scores[oracle_top_idx]) if oracle_top_idx >= 0 else float("nan")
        )
        top1_hit = bool(learned_top_idx == oracle_top_idx and learned_top_idx >= 0)
        oracle_order = np.argsort(-oracle_overlaps)
        top3_hit = bool(learned_top_idx in set(int(idx) for idx in oracle_order[:3]))
        delta = deltas_by_case.get(item.case_id)

        base_record: dict[str, Any] = {
            "case_id": item.case_id,
            "dicom_id": item.dicom_id,
            "category": item.category,
            "finding": item.finding,
        }
        if delta is not None and "split" in delta.index:
            base_record["split"] = delta["split"]

        delta_values = {
            f"delta_{metric}": (float(delta[f"delta_{metric}"]) if delta is not None else float("nan"))
            for metric in METRICS
        }
        case_rows.append(
            {
                **base_record,
                "learned_top_region": learned_top_region,
                "oracle_top_region": oracle_top_region,
                "has_learned_evidence": has_learned_evidence,
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "learned_top_score": learned_top_score,
                "oracle_top_overlap": oracle_top_overlap,
                "learned_top_oracle_overlap": learned_top_oracle_overlap,
                "oracle_top_learned_score": oracle_top_learned_score,
                "oracle_recovery_ratio": (
                    learned_top_oracle_overlap / oracle_top_overlap if oracle_top_overlap > 0 else float("nan")
                ),
                **delta_values,
                "failure_mode": _failure_mode(
                    top1_hit=top1_hit,
                    oracle_top_overlap=oracle_top_overlap,
                    learned_top_oracle_overlap=learned_top_oracle_overlap,
                    delta_cnr=delta_values["delta_cnr"],
                    oracle_overlap_floor=oracle_overlap_floor,
                ),
            }
        )
        for region, learned_score, oracle_overlap in zip(item.regions, learned_scores, oracle_overlaps):
            region_rows.append(
                {
                    **base_record,
                    "region": region,
                    "has_learned_evidence": has_learned_evidence,
                    "learned_score": float(learned_score),
                    "oracle_gt_overlap": float(oracle_overlap),
                    "is_learned_top_region": region == learned_top_region,
                    "is_oracle_top_region": region == oracle_top_region,
                    "evidence_source": f"{evidence_mode}_vs_oracle_gt_overlap",
                    **delta_values,
                }
            )

    region_table = pd.DataFrame(region_rows)
    case_table = pd.DataFrame(case_rows)
    summary = summarize_gap(region_table, case_table)
    return region_table, case_table, summary


def summarize_gap(region_table: pd.DataFrame, case_table: pd.DataFrame) -> pd.DataFrame:
    if case_table.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["category"]
    if "split" in case_table.columns:
        group_cols = ["split", "category"]
    for keys, cases in case_table.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        category = record["category"]
        regions = region_table[region_table["category"] == category]
        if "split" in record and "split" in regions.columns:
            regions = regions[regions["split"] == record["split"]]
        record.update(
            {
                "n_cases": int(len(cases)),
                "n_regions": int(len(regions)),
                "learned_evidence_case_rate": round(float(cases["has_learned_evidence"].mean()), 6)
                if "has_learned_evidence" in cases.columns
                else float("nan"),
                "top1_hit_rate": round(float(cases["top1_hit"].mean()), 6),
                "top3_hit_rate": round(float(cases["top3_hit"].mean()), 6),
                "mean_oracle_top_overlap": round(float(cases["oracle_top_overlap"].mean()), 6),
                "mean_learned_top_oracle_overlap": round(float(cases["learned_top_oracle_overlap"].mean()), 6),
                "mean_learned_top_score": round(float(cases["learned_top_score"].mean()), 6),
                "mean_oracle_top_learned_score": round(float(cases["oracle_top_learned_score"].mean()), 6),
                "mean_oracle_recovery_ratio": round(float(cases["oracle_recovery_ratio"].mean()), 6),
                "learned_oracle_pearson": round(
                    _corr(regions["learned_score"], regions["oracle_gt_overlap"], "pearson"), 6
                ),
                "learned_oracle_spearman": round(
                    _corr(regions["learned_score"], regions["oracle_gt_overlap"], "spearman"), 6
                ),
            }
        )
        for metric in METRICS:
            record[f"mean_delta_{metric}"] = round(float(cases[f"delta_{metric}"].mean()), 6)
        modes = cases["failure_mode"].value_counts().to_dict()
        record["failure_modes"] = json.dumps({str(k): int(v) for k, v in modes.items()}, ensure_ascii=False)
        rows.append(record)
    return pd.DataFrame(rows)


def build_oracle_learned_confusion(case_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count oracle-top-region to learned-top-region routes.

    The output is a long-form confusion matrix.  Keeping it long-form makes it
    easy to sort, filter by class/split, and render the same CSV on machines
    without plotting dependencies.
    """

    if case_table.empty:
        columns = [
            "category",
            "oracle_top_region",
            "learned_top_region",
            "count",
            "mean_delta_cnr",
            "mean_delta_iou",
            "mean_delta_dice",
            "is_hit",
            "route_fraction_within_category",
        ]
        empty = pd.DataFrame(columns=columns)
        return empty, empty.copy()

    group_cols = ["category", "oracle_top_region", "learned_top_region"]
    if "split" in case_table.columns:
        group_cols = ["split", *group_cols]
    rows: list[dict[str, Any]] = []
    for keys, group in case_table.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        record.update(
            {
                "count": int(len(group)),
                "mean_delta_cnr": round(float(group["delta_cnr"].mean()), 6)
                if "delta_cnr" in group.columns
                else float("nan"),
                "mean_delta_iou": round(float(group["delta_iou"].mean()), 6)
                if "delta_iou" in group.columns
                else float("nan"),
                "mean_delta_dice": round(float(group["delta_dice"].mean()), 6)
                if "delta_dice" in group.columns
                else float("nan"),
                "is_hit": bool(record["oracle_top_region"] == record["learned_top_region"]),
            }
        )
        rows.append(record)
    confusion = pd.DataFrame(rows)
    denom_cols = ["category"]
    if "split" in confusion.columns:
        denom_cols = ["split", "category"]
    denominators = confusion.groupby(denom_cols)["count"].transform("sum")
    confusion["route_fraction_within_category"] = (confusion["count"] / denominators).round(6)
    sort_cols = [col for col in ["split", "category"] if col in confusion.columns] + [
        "is_hit",
        "count",
        "mean_delta_cnr",
    ]
    confusion = confusion.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 3) + [True, False, True])
    top_misroutes = (
        confusion[~confusion["is_hit"]]
        .sort_values(
            [col for col in ["split", "category"] if col in confusion.columns] + ["count", "mean_delta_cnr"],
            ascending=[True] * len([col for col in ["split", "category"] if col in confusion.columns]) + [False, True],
        )
        .reset_index(drop=True)
    )
    return confusion.reset_index(drop=True), top_misroutes


def _test_gap_row(gap_summary: pd.DataFrame, category: str, split: str) -> pd.Series | None:
    if gap_summary.empty:
        return None
    rows = gap_summary[gap_summary["category"] == category]
    if "split" in rows.columns:
        rows = rows[rows["split"] == split]
    if rows.empty:
        return None
    return rows.iloc[0]


def _diagnose_gap(
    oracle_recoverable: bool,
    learned_evidence_case_rate: float,
    learned_delta_cnr: float,
    top1_hit_rate: float,
    top3_hit_rate: float,
    learned_oracle_spearman: float,
    mean_oracle_top_learned_score: float,
    learned_success_floor: float,
    min_evidence_case_rate: float,
    top1_good: float,
    top3_good: float,
    score_floor: float,
) -> tuple[str, str]:
    if not oracle_recoverable:
        return "oracle_not_recoverable", "oracle_region_repair_does_not_pass_gate"
    if pd.isna(learned_evidence_case_rate) or learned_evidence_case_rate < min_evidence_case_rate:
        return "missing_learned_evidence", "no_or_incomplete_region_scores_for_this_finding"
    if learned_delta_cnr >= learned_success_floor:
        return "learned_gap_small", "learned_repair_already_recovers_useful_oracle_signal"
    if top1_hit_rate < top1_good and top3_hit_rate < top3_good:
        return "evidence_misdirected", "learned_top_region_is_far_from_oracle_top_region"
    if top1_hit_rate < top1_good and top3_hit_rate >= top3_good:
        return "region_too_coarse_or_near_miss", "learned_evidence_is_near_oracle_region_but_not_precise"
    if (
        top1_hit_rate >= top1_good
        and top3_hit_rate >= top3_good
        and (pd.isna(mean_oracle_top_learned_score) or mean_oracle_top_learned_score < score_floor)
    ):
        return "predictor_calibration_or_score_scale", "ranking_is_reasonable_but_oracle_region_score_is_too_low"
    if not pd.isna(learned_oracle_spearman) and learned_oracle_spearman < 0:
        return "evidence_misdirected", "learned_oracle_rank_correlation_is_negative"
    return "unresolved_learned_or_fusion_gap", "oracle_passes_but_learned_repair_does_not_explain_failure_cleanly"


def build_gap_diagnosis_table(
    gap_summary: pd.DataFrame,
    oracle_profile: pd.DataFrame,
    split: str = "test",
    effect_floor: float = 0.02,
    learned_success_floor: float = 0.02,
    min_evidence_case_rate: float = 0.8,
    top1_good: float = 0.25,
    top3_good: float = 0.6,
    score_floor: float = 0.2,
) -> pd.DataFrame:
    """Merge oracle upper-bound and learned-vs-oracle evidence into diagnoses."""

    rows: list[dict[str, Any]] = []
    for _, oracle in oracle_profile.sort_values("category").iterrows():
        category = str(oracle["category"])
        gap = _test_gap_row(gap_summary, category, split=split)
        oracle_delta_cnr = float(oracle.get("oracle_delta_cnr", np.nan))
        oracle_cnr_ci_low = float(oracle.get("oracle_cnr_ci_low", np.nan))
        oracle_recoverable = (
            str(oracle.get("verdict", "")) == "region_recoverable"
            or (oracle_delta_cnr >= effect_floor and oracle_cnr_ci_low > 0)
        )

        record: dict[str, Any] = {
            "category": category,
            "n_oracle": int(oracle.get("n", 0)),
            "oracle_delta_cnr": oracle_delta_cnr,
            "oracle_cnr_ci_low": oracle_cnr_ci_low,
            "oracle_delta_iou": float(oracle.get("oracle_delta_iou", np.nan)),
            "oracle_delta_dice": float(oracle.get("oracle_delta_dice", np.nan)),
            "oracle_recoverable": bool(oracle_recoverable),
        }
        if gap is None:
            diagnosis, reason = _diagnose_gap(
                oracle_recoverable=oracle_recoverable,
                learned_evidence_case_rate=0.0,
                learned_delta_cnr=0.0,
                top1_hit_rate=0.0,
                top3_hit_rate=0.0,
                learned_oracle_spearman=float("nan"),
                mean_oracle_top_learned_score=float("nan"),
                learned_success_floor=learned_success_floor,
                min_evidence_case_rate=min_evidence_case_rate,
                top1_good=top1_good,
                top3_good=top3_good,
                score_floor=score_floor,
            )
            record.update(
                {
                    "n_gap_cases": 0,
                    "learned_evidence_case_rate": 0.0,
                    "learned_delta_cnr": np.nan,
                    "oracle_learned_gap_cnr": np.nan,
                    "top1_hit_rate": np.nan,
                    "top3_hit_rate": np.nan,
                    "learned_oracle_spearman": np.nan,
                    "mean_oracle_top_overlap": np.nan,
                    "mean_learned_top_score": np.nan,
                    "mean_oracle_top_learned_score": np.nan,
                    "diagnosis": diagnosis,
                    "reason": reason,
                    "recommended_next_step": _recommended_next_step(diagnosis),
                }
            )
            rows.append(record)
            continue

        learned_delta_cnr = float(gap.get("mean_delta_cnr", np.nan))
        learned_evidence_case_rate = float(gap.get("learned_evidence_case_rate", np.nan))
        top1_hit_rate = float(gap.get("top1_hit_rate", np.nan))
        top3_hit_rate = float(gap.get("top3_hit_rate", np.nan))
        learned_oracle_spearman = float(gap.get("learned_oracle_spearman", np.nan))
        mean_oracle_top_learned_score = float(gap.get("mean_oracle_top_learned_score", np.nan))
        diagnosis, reason = _diagnose_gap(
            oracle_recoverable=oracle_recoverable,
            learned_evidence_case_rate=learned_evidence_case_rate,
            learned_delta_cnr=learned_delta_cnr,
            top1_hit_rate=top1_hit_rate,
            top3_hit_rate=top3_hit_rate,
            learned_oracle_spearman=learned_oracle_spearman,
            mean_oracle_top_learned_score=mean_oracle_top_learned_score,
            learned_success_floor=learned_success_floor,
            min_evidence_case_rate=min_evidence_case_rate,
            top1_good=top1_good,
            top3_good=top3_good,
            score_floor=score_floor,
        )
        record.update(
            {
                "n_gap_cases": int(gap.get("n_cases", 0)),
                "learned_evidence_case_rate": learned_evidence_case_rate,
                "learned_delta_cnr": learned_delta_cnr,
                "oracle_learned_gap_cnr": oracle_delta_cnr - learned_delta_cnr
                if not pd.isna(learned_delta_cnr)
                else np.nan,
                "top1_hit_rate": top1_hit_rate,
                "top3_hit_rate": top3_hit_rate,
                "learned_oracle_spearman": learned_oracle_spearman,
                "mean_oracle_top_overlap": float(gap.get("mean_oracle_top_overlap", np.nan)),
                "mean_learned_top_score": float(gap.get("mean_learned_top_score", np.nan)),
                "mean_oracle_top_learned_score": mean_oracle_top_learned_score,
                "diagnosis": diagnosis,
                "reason": reason,
                "recommended_next_step": _recommended_next_step(diagnosis),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _recommended_next_step(diagnosis: str) -> str:
    mapping = {
        "missing_learned_evidence": "train_or_export_region_predictor_for_this_finding_before_claiming_gap",
        "evidence_misdirected": "inspect_label_bias_feature_bias_and_oracle_to_learned_confusion_routes",
        "region_too_coarse_or_near_miss": "refine_region_granularity_or_split_high_mass_neighbor_regions",
        "predictor_calibration_or_score_scale": "fit_calibration_on_imagenome_heldout_then_freeze_before_mscxr",
        "learned_gap_small": "keep_as_supported_learned_repair_candidate",
        "oracle_not_recoverable": "do_not_prioritize_region_repair_for_this_finding",
        "unresolved_learned_or_fusion_gap": "run_case_level_audit_and_check_fusion_alpha_or_feature_normalization",
    }
    return mapping.get(diagnosis, "manual_review")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a compact markdown table without pandas' optional tabulate dependency."""

    if df.empty:
        return "No rows."
    text_df = df.copy()
    for column in text_df.columns:
        text_df[column] = text_df[column].map(lambda value: "" if pd.isna(value) else str(value))
    columns = [str(column) for column in text_df.columns]
    rows = text_df.to_dict(orient="records")
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }

    def fmt(values: dict[str, str]) -> str:
        return "| " + " | ".join(str(values[column]).ljust(widths[column]) for column in columns) + " |"

    header = fmt({column: column for column in columns})
    separator = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    body = [fmt({column: str(row[column]) for column in columns}) for row in rows]
    return "\n".join([header, separator, *body])


def build_markdown_report(
    summary: pd.DataFrame,
    case_table: pd.DataFrame,
    args: argparse.Namespace,
    diagnosis: pd.DataFrame | None = None,
) -> str:
    confusion, top_misroutes = build_oracle_learned_confusion(case_table)
    lines = [
        "# Stage C Learned-vs-Oracle Gap Analysis",
        "",
        "定义：oracle evidence = MS-CXR GT mask 落入每个 Chest ImaGenome region map 的比例；learned evidence = Stage B predictor 输出的 region abnormality probability。",
        "",
        f"- method: `{args.method}`",
        f"- baseline_method: `{args.baseline_method}`",
        f"- oracle_overlap_floor: `{args.oracle_overlap_floor}`",
        f"- total_cases: `{len(case_table)}`",
        "",
        "## Per-Class Summary",
        "",
    ]
    if summary.empty:
        lines.append("No matched cases.")
    else:
        lines.append(dataframe_to_markdown(summary))
    if diagnosis is not None:
        lines.extend(
            [
                "",
                "## 8-Class Gap Diagnosis",
                "",
                dataframe_to_markdown(diagnosis),
            ]
        )
    if not case_table.empty:
        lines.extend(
            [
                "",
                "## Main Oracle -> Learned Misroutes",
                "",
                dataframe_to_markdown(top_misroutes.head(30)),
                "",
                "## Worst Delta-CNR Cases",
                "",
                dataframe_to_markdown(case_table.sort_values("delta_cnr").head(20)),
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze learned region evidence vs oracle GT-region evidence.")
    parser.add_argument("--prepared-inputs-npz", required=True, type=Path)
    parser.add_argument("--region-score-csv", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--candidate-categories", default="Pneumothorax,Pleural Effusion")
    parser.add_argument("--method", default="learned_selective")
    parser.add_argument("--baseline-method", default="baseline")
    parser.add_argument("--evidence-mode", default="learned", choices=["learned", "disease_pooled", "phrase_anatomy"])
    parser.add_argument("--oracle-profile-csv", type=Path, default=None)
    parser.add_argument("--oracle-overlap-floor", type=float, default=0.25)
    parser.add_argument("--effect-floor", type=float, default=0.02)
    parser.add_argument("--learned-success-floor", type=float, default=0.02)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--dataset", default="MS_CXR_CLS")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_categories = set(parse_csv_list(args.candidate_categories, DEFAULT_CANDIDATES))
    inputs = [item for item in _load_prepared_inputs(args.prepared_inputs_npz) if item.category in candidate_categories]
    score_table = load_region_score_table(args.region_score_csv)
    gt_masks = load_mscxr_gt_masks_by_case(max_cases=args.max_cases, dataset=args.dataset)
    metrics = pd.read_csv(args.metrics_csv)

    region_table, case_table, summary = build_gap_tables(
        inputs=inputs,
        gt_masks_by_case=gt_masks,
        score_table=score_table,
        per_case_metrics=metrics,
        method=args.method,
        baseline_method=args.baseline_method,
        oracle_overlap_floor=args.oracle_overlap_floor,
        evidence_mode=args.evidence_mode,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "region_gap_table": args.outdir / "region_gap_table.csv",
        "case_gap_table": args.outdir / "case_gap_table.csv",
        "per_class_gap_summary": args.outdir / "per_class_gap_summary.csv",
        "gap_diagnosis": args.outdir / "gap_diagnosis.csv",
        "oracle_learned_confusion": args.outdir / "oracle_learned_confusion.csv",
        "top_oracle_learned_misroutes": args.outdir / "top_oracle_learned_misroutes.csv",
        "markdown": args.outdir / "gap_analysis_report.md",
        "summary": args.outdir / "gap_analysis_summary.json",
    }
    confusion, top_misroutes = build_oracle_learned_confusion(case_table)
    diagnosis = None
    if args.oracle_profile_csv is not None:
        oracle_profile = pd.read_csv(args.oracle_profile_csv)
        diagnosis = build_gap_diagnosis_table(
            summary,
            oracle_profile,
            split="test",
            effect_floor=args.effect_floor,
            learned_success_floor=args.learned_success_floor,
        )
        diagnosis.to_csv(paths["gap_diagnosis"], index=False)
    region_table.to_csv(paths["region_gap_table"], index=False)
    case_table.to_csv(paths["case_gap_table"], index=False)
    summary.to_csv(paths["per_class_gap_summary"], index=False)
    confusion.to_csv(paths["oracle_learned_confusion"], index=False)
    top_misroutes.to_csv(paths["top_oracle_learned_misroutes"], index=False)
    paths["markdown"].write_text(build_markdown_report(summary, case_table, args, diagnosis), encoding="utf-8")

    result: dict[str, Any] = {
        "status": "ok",
        "prepared_inputs_npz": str(args.prepared_inputs_npz),
        "region_score_csv": str(args.region_score_csv),
        "metrics_csv": str(args.metrics_csv),
        "candidate_categories": sorted(candidate_categories),
        "method": args.method,
        "baseline_method": args.baseline_method,
        "evidence_mode": args.evidence_mode,
        "num_inputs": len(inputs),
        "num_matched_cases": int(len(case_table)),
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key != "gap_diagnosis" or diagnosis is not None
        },
    }
    if diagnosis is not None:
        result["diagnosis_counts"] = {
            str(k): int(v) for k, v in diagnosis["diagnosis"].value_counts().to_dict().items()
        }
    paths["summary"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
