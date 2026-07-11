"""Build DP-MSA-v0 weak training caches.

The cache target is a region-score-weighted anatomy map. It intentionally does
not read MS-CXR boxes, masks, or oracle maps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anaprior.eval.eval_mscxr_learned_repair import (
    _load_prepared_inputs,
    load_region_score_table,
    region_scores_for_case,
)
from anaprior.eval.phrase_subtype import PHRASE_SUBTYPE_TO_ID, phrase_subtype_id


DEFAULT_FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
)


def parse_csv_list(raw: str | None, default: tuple[str, ...] = DEFAULT_FINDINGS) -> list[str]:
    if raw is None:
        return list(default)
    values = [part.strip() for part in str(raw).replace(";", ",").split(",") if part.strip()]
    return values or list(default)


def parse_json_mapping(raw: str | None) -> dict[str, float]:
    if raw is None or not str(raw).strip():
        return {}
    text = str(raw).strip()
    path = Path(text)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("disease beta mapping must be a JSON object")
    return {str(key): float(value) for key, value in payload.items()}


def load_hmap_dict(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    raw = np.load(path, allow_pickle=True).item()
    hmaps = {}
    for case_id, payload in raw.items():
        if isinstance(payload, dict):
            value = payload.get("hmap")
        else:
            value = payload
        if value is None:
            continue
        hmaps[str(case_id)] = np.asarray(value, dtype=np.float32)
    return hmaps


def stable_fraction(value: str, seed: int = 0) -> float:
    digest = hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / float(0xFFFFFFFF)


def hashed_split(case_id: str, valid_fraction: float, seed: int = 0) -> str:
    if valid_fraction <= 0:
        return "train"
    if valid_fraction >= 1:
        return "val"
    return "val" if stable_fraction(case_id, seed=seed) < float(valid_fraction) else "train"


def normalize_map(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    span = hi - lo
    if not np.isfinite(span) or span <= eps:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / span, 0.0, 1.0).astype(np.float32)


def region_score_weighted_target(region_maps: np.ndarray, region_scores: np.ndarray) -> np.ndarray:
    maps = np.asarray(region_maps, dtype=np.float32)
    scores = np.asarray(region_scores, dtype=np.float32).clip(min=0.0)
    weighted = np.sum(maps * scores[:, None, None], axis=0)
    return normalize_map(weighted)[None, :, :]


def mixed_target(base_hmap: np.ndarray, region_target: np.ndarray, beta: float) -> np.ndarray:
    base = np.asarray(base_hmap, dtype=np.float32)
    target = np.asarray(region_target, dtype=np.float32)
    if base.shape != target.shape:
        raise ValueError(f"base_hmap shape {base.shape} does not match region_target shape {target.shape}")
    beta = float(beta)
    return normalize_map(((1.0 - beta) * base[0]) + (beta * target[0]))[None, :, :]


def _split_records(records: list[dict[str, Any]], valid_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    train_records = []
    valid_records = []
    for record in records:
        if hashed_split(str(record["case_id"]), valid_fraction=valid_fraction, seed=seed) == "val":
            valid_records.append(record)
        else:
            train_records.append(record)
    duplicated_singleton = False
    if len(records) == 1:
        train_records = [records[0]]
        valid_records = [records[0]]
        duplicated_singleton = True
    elif records:
        if not train_records:
            train_records.append(valid_records.pop())
        if not valid_records:
            valid_records.append(train_records.pop())
    return train_records, valid_records, duplicated_singleton


def _tensor_payload(
    records: list[dict[str, Any]],
    finding_vocab: dict[str, int],
    region_names: list[str],
    target_mode: str,
    base_method_name: str,
    base_hmaps_npy: Path | None,
    target_mix_beta: float,
    disease_betas: dict[str, float],
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot build DP-MSA cache from zero records")
    return {
        "base_hmaps": torch.from_numpy(np.stack([record["base_hmap"] for record in records], axis=0)).float(),
        "region_maps": torch.from_numpy(np.stack([record["region_maps"] for record in records], axis=0)).float(),
        "region_scores": torch.from_numpy(np.stack([record["region_scores"] for record in records], axis=0)).float(),
        "target_hmaps": torch.from_numpy(np.stack([record["target_hmap"] for record in records], axis=0)).float(),
        "disease_ids": torch.tensor([int(record["disease_id"]) for record in records], dtype=torch.long),
        "subtype_ids": torch.tensor([int(record["subtype_id"]) for record in records], dtype=torch.long),
        "case_ids": [str(record["case_id"]) for record in records],
        "categories": [str(record["category"]) for record in records],
        "phrases": [str(record["phrase"]) for record in records],
        "target_betas": torch.tensor([float(record["target_beta"]) for record in records], dtype=torch.float32),
        "finding_vocab": dict(finding_vocab),
        "subtype_vocab": dict(PHRASE_SUBTYPE_TO_ID),
        "region_names": list(region_names),
        "target_mode": str(target_mode),
        "base_method_name": str(base_method_name),
        "base_hmaps_npy": str(base_hmaps_npy) if base_hmaps_npy is not None else "",
        "target_mix_beta": float(target_mix_beta),
        "disease_betas": dict(disease_betas),
        "uses_mscxr_boxes": False,
    }


def build_dp_msa_training_cache(
    prepared_inputs_npz: Path,
    region_score_csv: Path,
    outdir: Path,
    findings: list[str] | tuple[str, ...] = DEFAULT_FINDINGS,
    valid_fraction: float = 0.2,
    seed: int = 13,
    min_score_sum: float = 1e-6,
    source_split: str = "all",
    source_val_fraction: float = 0.3,
    source_seed: int = 0,
    max_cases: int | None = None,
    base_hmaps_npy: Path | None = None,
    base_method_name: str = "prepared_inputs",
    target_mix_beta: float = 1.0,
    disease_betas: dict[str, float] | None = None,
) -> dict[str, Any]:
    inputs = _load_prepared_inputs(prepared_inputs_npz)
    if max_cases is not None:
        inputs = inputs[: int(max_cases)]
    score_table = load_region_score_table(region_score_csv)
    base_hmaps = load_hmap_dict(base_hmaps_npy)
    disease_betas = {str(key): float(value) for key, value in (disease_betas or {}).items()}
    target_mix_beta = float(target_mix_beta)
    target_mode = "mixed_base_region_score" if base_hmaps or target_mix_beta < 1.0 or disease_betas else "region_score_weighted"
    finding_vocab = {str(finding): idx for idx, finding in enumerate([str(item) for item in findings])}
    allowed_categories = set(finding_vocab)
    source_split = str(source_split).strip().lower()
    if source_split not in {"all", "train", "val", "test"}:
        raise ValueError("source_split must be one of: all, train, val, test")

    records: list[dict[str, Any]] = []
    skipped: dict[str, Counter] = {
        "category_not_requested": Counter(),
        "source_split_filtered": Counter(),
        "region_mismatch": Counter(),
        "low_score_sum": Counter(),
        "missing_base_hmap": Counter(),
    }
    region_names: list[str] | None = None
    for item in inputs:
        category = str(item.category)
        if category not in allowed_categories:
            skipped["category_not_requested"][category] += 1
            continue
        split = hashed_split(str(item.case_id), valid_fraction=source_val_fraction, seed=source_seed)
        split_alias = "test" if split == "train" else split
        if source_split != "all" and source_split not in {split, split_alias}:
            skipped["source_split_filtered"][category] += 1
            continue
        if region_names is None:
            region_names = [str(region) for region in item.regions]
        if [str(region) for region in item.regions] != region_names:
            skipped["region_mismatch"][category] += 1
            continue
        scores = region_scores_for_case(score_table, item.dicom_id, item.finding, item.regions)
        if float(np.sum(np.clip(scores, 0.0, None))) <= float(min_score_sum):
            skipped["low_score_sum"][category] += 1
            continue
        phrase = item.phrase or item.finding
        fallback_base = normalize_map(np.asarray(item.heatmap, dtype=np.float32))[None, :, :]
        if base_hmaps:
            maybe_base = base_hmaps.get(str(item.case_id))
            if maybe_base is None:
                skipped["missing_base_hmap"][category] += 1
                continue
            base_hmap = np.asarray(maybe_base, dtype=np.float32)[None, :, :]
        else:
            base_hmap = fallback_base
        region_target = region_score_weighted_target(item.region_maps, scores)
        beta = float(disease_betas.get(category, target_mix_beta))
        if target_mode == "region_score_weighted":
            target_hmap = region_target
        else:
            target_hmap = mixed_target(base_hmap, region_target, beta=beta)
        records.append(
            {
                "case_id": str(item.case_id),
                "category": category,
                "phrase": str(phrase),
                "base_hmap": base_hmap,
                "region_maps": np.asarray(item.region_maps, dtype=np.float32),
                "region_scores": np.asarray(scores, dtype=np.float32),
                "target_hmap": target_hmap,
                "target_beta": beta,
                "disease_id": int(finding_vocab[category]),
                "subtype_id": int(phrase_subtype_id(str(phrase), category=category)),
            }
        )
    if region_names is None:
        raise ValueError("no prepared inputs with usable region names")
    if not records:
        raise ValueError(
            "no DP-MSA cache rows were built; check findings, source_split, region scores, and min_score_sum"
        )

    train_records, valid_records, duplicated_singleton = _split_records(records, valid_fraction=valid_fraction, seed=seed)
    outdir.mkdir(parents=True, exist_ok=True)
    train_cache = outdir / "train_dp_msa_v0.pt"
    valid_cache = outdir / "valid_dp_msa_v0.pt"
    tensor_kwargs = {
        "finding_vocab": finding_vocab,
        "region_names": region_names,
        "target_mode": target_mode,
        "base_method_name": base_method_name,
        "base_hmaps_npy": base_hmaps_npy,
        "target_mix_beta": target_mix_beta,
        "disease_betas": disease_betas,
    }
    torch.save(_tensor_payload(train_records, **tensor_kwargs), train_cache)
    torch.save(_tensor_payload(valid_records, **tensor_kwargs), valid_cache)

    train_categories = Counter(str(record["category"]) for record in train_records)
    valid_categories = Counter(str(record["category"]) for record in valid_records)
    report = {
        "status": "ok",
        "prepared_inputs_npz": str(prepared_inputs_npz),
        "region_score_csv": str(region_score_csv),
        "outdir": str(outdir),
        "train_cache": str(train_cache),
        "valid_cache": str(valid_cache),
        "target_mode": target_mode,
        "base_method_name": str(base_method_name),
        "base_hmaps_npy": str(base_hmaps_npy) if base_hmaps_npy is not None else "",
        "target_mix_beta": float(target_mix_beta),
        "disease_betas": disease_betas,
        "uses_mscxr_boxes": False,
        "source_split": source_split,
        "source_val_fraction": float(source_val_fraction),
        "source_seed": int(source_seed),
        "valid_fraction": float(valid_fraction),
        "seed": int(seed),
        "min_score_sum": float(min_score_sum),
        "num_inputs": int(len(inputs)),
        "num_records": int(len(records)),
        "num_train": int(len(train_records)),
        "num_valid": int(len(valid_records)),
        "duplicated_singleton": bool(duplicated_singleton),
        "finding_vocab": finding_vocab,
        "subtype_vocab": dict(PHRASE_SUBTYPE_TO_ID),
        "region_names": region_names,
        "train_categories": {str(k): int(v) for k, v in train_categories.items()},
        "valid_categories": {str(k): int(v) for k, v in valid_categories.items()},
        "skipped": {
            name: {str(k): int(v) for k, v in counter.items()}
            for name, counter in skipped.items()
        },
    }
    report_path = outdir / "dp_msa_cache_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DP-MSA-v0 weak training cache from Stage C inputs.")
    parser.add_argument("--prepared-inputs-npz", required=True, type=Path)
    parser.add_argument("--region-score-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--findings", default=",".join(DEFAULT_FINDINGS))
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-score-sum", type=float, default=1e-6)
    parser.add_argument("--source-split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--source-val-fraction", type=float, default=0.3)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--base-hmaps-npy", type=Path, default=None)
    parser.add_argument("--base-method-name", default="prepared_inputs")
    parser.add_argument("--target-mix-beta", type=float, default=1.0)
    parser.add_argument("--disease-beta-json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_dp_msa_training_cache(
        prepared_inputs_npz=args.prepared_inputs_npz,
        region_score_csv=args.region_score_csv,
        outdir=args.outdir,
        findings=parse_csv_list(args.findings, DEFAULT_FINDINGS),
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        min_score_sum=args.min_score_sum,
        source_split=args.source_split,
        source_val_fraction=args.source_val_fraction,
        source_seed=args.source_seed,
        max_cases=args.max_cases,
        base_hmaps_npy=args.base_hmaps_npy,
        base_method_name=args.base_method_name,
        target_mix_beta=args.target_mix_beta,
        disease_betas=parse_json_mapping(args.disease_beta_json),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
