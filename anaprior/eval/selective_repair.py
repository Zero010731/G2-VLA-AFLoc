"""Core selective repair fusion for learned AnaPrior-Loc.

This module is intentionally I/O-free. It turns per-region abnormality scores
into a repair map and blends that map with an AFLoc heatmap. Dataset loading,
MS-CXR metrics, and predictor inference stay in separate scripts.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from anaprior.eval.disease_conditioned_gate import apply_disease_conditioned_gate, apply_disease_specific_pooling
from anaprior.eval.phrase_anatomy_router import apply_phrase_anatomy_router


RepairScope = Literal["candidate", "all"]
ScoreMode = Literal["learned", "shuffled", "uniform", "disease_gated", "disease_pooled", "phrase_anatomy"]


@dataclass(frozen=True)
class RepairCase:
    case_id: str
    category: str
    heatmap: np.ndarray
    region_maps: np.ndarray
    region_scores: np.ndarray
    regions: list[str] | None = None
    phrase: str = ""


@dataclass(frozen=True)
class RepairResult:
    heatmap: np.ndarray
    status: str


def normalize_unit_map(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    finite = arr[valid]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = hi - lo
    if span <= eps:
        arr[valid] = 0.0
    else:
        arr[valid] = (finite - lo) / span
    return arr


def repair_map_from_region_scores(
    region_maps: np.ndarray,
    finding_scores: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray | None:
    maps = np.asarray(region_maps, dtype=np.float32)
    scores = np.asarray(finding_scores, dtype=np.float32)
    if maps.ndim != 3:
        raise ValueError("region_maps must have shape [R,H,W]")
    if scores.ndim != 1 or scores.shape[0] != maps.shape[0]:
        raise ValueError("finding_scores must have shape [R] matching region_maps")
    if scores.size == 0 or float(np.nanmax(scores)) <= eps:
        return None

    region_mass = np.asarray([float(np.nansum(region_map)) for region_map in maps], dtype=np.float32)
    usable = (scores > eps) & (region_mass > eps)
    if not np.any(usable):
        return None
    weights = np.where(usable, scores, 0.0).astype(np.float32)
    repair = np.tensordot(weights, maps, axes=(0, 0)).astype(np.float32)
    if float(np.nanmax(repair)) <= eps:
        return None
    return normalize_unit_map(repair)


def _stable_int(seed_key: str) -> int:
    digest = hashlib.sha256(str(seed_key).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def shuffle_region_scores(scores: np.ndarray, seed_key: str) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size < 2:
        return arr.copy()
    shift = _stable_int(seed_key) % (arr.size - 1) + 1
    return np.roll(arr, shift).astype(np.float32)


def blend_heatmap(hmap: np.ndarray, repair_map: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base_input = np.asarray(hmap, dtype=np.float32)
    repair_input = np.asarray(repair_map, dtype=np.float32)
    if base_input.shape != repair_input.shape:
        raise ValueError("hmap and repair_map must have the same shape")
    valid = np.isfinite(base_input)
    base = normalize_unit_map(base_input)
    repair = normalize_unit_map(repair_input)
    fused = ((1.0 - alpha) * base) + (alpha * repair)
    fused = fused.astype(np.float32)
    fused[~valid] = np.nan
    return fused


def _candidate_lut(candidate_categories: set[str] | list[str] | tuple[str, ...]) -> set[str]:
    return {str(category).strip().lower() for category in candidate_categories}


def _scores_for_case(case: RepairCase, mode: ScoreMode, seed: int) -> np.ndarray:
    scores = np.asarray(case.region_scores, dtype=np.float32)
    if mode == "learned":
        return scores
    if mode == "shuffled":
        return shuffle_region_scores(scores, seed_key=f"{seed}:{case.case_id}:{case.category}")
    if mode == "uniform":
        region_mass = np.asarray([float(np.nansum(region_map)) for region_map in case.region_maps], dtype=np.float32)
        return (region_mass > 1e-6).astype(np.float32)
    if mode == "disease_gated":
        if case.regions is None:
            return scores
        return apply_disease_conditioned_gate(case.category, case.regions, scores).scores
    if mode == "disease_pooled":
        if case.regions is None:
            return scores
        return apply_disease_specific_pooling(case.category, case.regions, scores).scores
    if mode == "phrase_anatomy":
        if case.regions is None:
            return scores
        return apply_phrase_anatomy_router(case.category, case.phrase, case.regions, scores).scores
    raise ValueError(f"Unsupported score mode: {mode}")


def apply_repair_cases(
    cases: list[RepairCase],
    candidate_categories: set[str] | list[str] | tuple[str, ...],
    alpha: float,
    repair_scope: RepairScope = "candidate",
    score_mode: ScoreMode = "learned",
    seed: int = 0,
) -> tuple[dict[str, RepairResult], dict[str, Counter]]:
    """Apply selective repair to a list of cases.

    `repair_scope="candidate"` touches only the evidence-supported candidate
    classes. `repair_scope="all"` is the required ablation that applies the same
    mechanism to non-candidate classes too.
    """

    if repair_scope not in {"candidate", "all"}:
        raise ValueError("repair_scope must be 'candidate' or 'all'")
    if score_mode not in {"learned", "shuffled", "uniform", "disease_gated", "disease_pooled", "phrase_anatomy"}:
        raise ValueError(
            "score_mode must be 'learned', 'shuffled', 'uniform', 'disease_gated', 'disease_pooled', or 'phrase_anatomy'"
        )

    candidates = _candidate_lut(candidate_categories)
    out: dict[str, RepairResult] = {}
    stats = {
        "repaired_categories": Counter(),
        "skipped_non_candidate": Counter(),
        "missing_region_objects": Counter(),
        "missing_finding_evidence": Counter(),
        "gate_passed_categories": Counter(),
        "gate_bypassed_categories": Counter(),
    }

    for case in cases:
        category_key = str(case.category).strip().lower()
        is_candidate = category_key in candidates
        heatmap = np.asarray(case.heatmap, dtype=np.float32)
        if repair_scope == "candidate" and not is_candidate:
            out[case.case_id] = RepairResult(heatmap=heatmap.copy(), status="baseline_copy")
            stats["skipped_non_candidate"][case.category] += 1
            continue
        if float(alpha) <= 0.0:
            out[case.case_id] = RepairResult(heatmap=heatmap.copy(), status="baseline_copy")
            continue

        region_maps = np.asarray(case.region_maps, dtype=np.float32)
        if region_maps.ndim != 3 or float(np.nansum(region_maps)) <= 1e-6:
            out[case.case_id] = RepairResult(heatmap=heatmap.copy(), status="missing_region_objects")
            stats["missing_region_objects"][case.category] += 1
            continue

        if score_mode in {"disease_gated", "disease_pooled", "phrase_anatomy"} and case.regions is not None:
            raw_scores = np.asarray(case.region_scores, dtype=np.float32)
            if score_mode == "disease_gated":
                gate = apply_disease_conditioned_gate(case.category, case.regions, raw_scores)
                prefix = "disease_gate_bypass"
            elif score_mode == "disease_pooled":
                gate = apply_disease_specific_pooling(case.category, case.regions, raw_scores)
                prefix = "disease_pool_bypass"
            else:
                gate = apply_phrase_anatomy_router(case.category, case.phrase, case.regions, raw_scores)
                prefix = "phrase_anatomy_bypass"
            if gate.status == "bypass":
                out[case.case_id] = RepairResult(
                    heatmap=heatmap.copy(),
                    status=f"{prefix}:{gate.reason}",
                )
                stats["gate_bypassed_categories"][case.category] += 1
                continue
            stats["gate_passed_categories"][case.category] += 1
            scores = gate.scores
        else:
            scores = _scores_for_case(case, mode=score_mode, seed=seed)
        repair_map = repair_map_from_region_scores(region_maps, scores)
        if repair_map is None:
            out[case.case_id] = RepairResult(heatmap=heatmap.copy(), status="missing_finding_evidence")
            stats["missing_finding_evidence"][case.category] += 1
            continue

        out[case.case_id] = RepairResult(
            heatmap=blend_heatmap(heatmap, repair_map, alpha=alpha),
            status="repaired",
        )
        stats["repaired_categories"][case.category] += 1

    return out, stats
