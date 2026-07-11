"""Build MS-CXR DP-MSA repair heatmaps from a frozen checkpoint."""

from __future__ import annotations

import argparse
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
from anaprior.eval.phrase_subtype import phrase_subtype_id
from anaprior.models.dp_msa_adapter import DPMultiScaleSpatialAdapter


def _load_checkpoint(
    path: Path,
    device: str,
    lambda_override: float | None = None,
) -> tuple[DPMultiScaleSpatialAdapter, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "model_config", "finding_vocab", "subtype_vocab", "region_names"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} missing checkpoint fields: {missing}")
    model_config = dict(payload["model_config"])
    if lambda_override is not None:
        model_config["lambda_weight"] = float(lambda_override)
    model = DPMultiScaleSpatialAdapter(**model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    payload = dict(payload)
    payload["model_config"] = model_config
    return model, payload


def _baseline_hmaps(inputs) -> dict[str, dict[str, Any]]:
    return {
        item.case_id: {
            "hmap": np.asarray(item.heatmap, dtype=np.float32).copy(),
            "learned_repair": "baseline",
        }
        for item in inputs
    }


def _load_base_hmaps(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    raw = np.load(path, allow_pickle=True).item()
    out = {}
    for case_id, payload in raw.items():
        if isinstance(payload, dict):
            value = payload.get("hmap")
        else:
            value = payload
        if value is not None:
            out[str(case_id)] = np.asarray(value, dtype=np.float32)
    return out


def build_dp_msa_hmaps(
    prepared_inputs_npz: Path,
    region_score_csv: Path,
    checkpoint: Path,
    device: str = "cpu",
    method_name: str = "dp_msa",
    lambda_override: float | None = None,
    base_hmaps_npy: Path | None = None,
    base_method_name: str = "prepared_inputs",
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, int]]]]:
    inputs = _load_prepared_inputs(prepared_inputs_npz)
    score_table = load_region_score_table(region_score_csv)
    base_hmaps = _load_base_hmaps(base_hmaps_npy)
    method_name = str(method_name).strip()
    if not method_name:
        raise ValueError("method_name must be non-empty")
    model, payload = _load_checkpoint(checkpoint, device=device, lambda_override=lambda_override)
    finding_vocab = {str(k): int(v) for k, v in payload["finding_vocab"].items()}
    subtype_vocab = {str(k): int(v) for k, v in payload["subtype_vocab"].items()}
    expected_regions = list(payload["region_names"])
    torch_device = torch.device(device)
    hmaps: dict[str, dict[str, dict[str, Any]]] = {"baseline": _baseline_hmaps(inputs), method_name: {}}
    stats = {
        "baseline": {
            "repaired_categories": {},
            "missing_finding_vocab": {},
            "region_mismatch": {},
        },
        method_name: {
            "repaired_categories": Counter(),
            "missing_finding_vocab": Counter(),
            "region_mismatch": Counter(),
            "missing_base_hmap": Counter(),
            "base_method_name": Counter(),
        },
    }
    with torch.no_grad():
        for item in inputs:
            if item.category not in finding_vocab:
                hmaps[method_name][item.case_id] = {
                    "hmap": np.asarray(item.heatmap, dtype=np.float32).copy(),
                    "learned_repair": "baseline_copy:missing_finding_vocab",
                }
                stats[method_name]["missing_finding_vocab"][item.category] += 1
                continue
            if list(item.regions) != expected_regions:
                hmaps[method_name][item.case_id] = {
                    "hmap": np.asarray(item.heatmap, dtype=np.float32).copy(),
                    "learned_repair": "baseline_copy:region_mismatch",
                }
                stats[method_name]["region_mismatch"][item.category] += 1
                continue
            if base_hmaps:
                base_hmap = base_hmaps.get(str(item.case_id))
                if base_hmap is None:
                    hmaps[method_name][item.case_id] = {
                        "hmap": np.asarray(item.heatmap, dtype=np.float32).copy(),
                        "learned_repair": "baseline_copy:missing_base_hmap",
                    }
                    stats[method_name]["missing_base_hmap"][item.category] += 1
                    continue
                stats[method_name]["base_method_name"][base_method_name] += 1
            else:
                base_hmap = np.asarray(item.heatmap, dtype=np.float32)
                stats[method_name]["base_method_name"]["prepared_inputs"] += 1
            scores = region_scores_for_case(score_table, item.dicom_id, item.finding, item.regions)
            subtype = phrase_subtype_id(item.phrase or item.finding, category=item.category)
            subtype = int(subtype if subtype in set(subtype_vocab.values()) else subtype_vocab.get("other", 0))
            out = model(
                torch.from_numpy(np.asarray(base_hmap, dtype=np.float32))[None, None].to(torch_device),
                torch.from_numpy(np.asarray(item.region_maps, dtype=np.float32))[None].to(torch_device),
                torch.from_numpy(np.asarray(scores, dtype=np.float32))[None].to(torch_device),
                torch.tensor([finding_vocab[item.category]], dtype=torch.long, device=torch_device),
                torch.tensor([subtype], dtype=torch.long, device=torch_device),
            )
            hmaps[method_name][item.case_id] = {
                "hmap": out.final_heatmap[0, 0].detach().cpu().numpy().astype(np.float32),
                "learned_repair": f"{method_name}_repaired",
            }
            stats[method_name]["repaired_categories"][item.category] += 1
    stats_out = {
        method: {key: {str(k): int(v) for k, v in value.items()} for key, value in method_stats.items()}
        for method, method_stats in stats.items()
    }
    return hmaps, stats_out


def save_dp_msa_hmaps(
    hmaps: dict[str, dict[str, dict[str, Any]]],
    stats: dict[str, dict[str, dict[str, int]]],
    outdir: Path,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for method, method_hmaps in hmaps.items():
        method_dir = outdir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        path = method_dir / "hmaps.npy"
        np.save(path, method_hmaps)
        outputs[method] = str(path)
    stats_path = outdir / "build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["build_stats"] = str(stats_path)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DP-MSA repair heatmaps for MS-CXR evaluation.")
    parser.add_argument("--prepared-inputs-npz", required=True, type=Path)
    parser.add_argument("--region-score-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method-name", default="dp_msa")
    parser.add_argument("--lambda-override", type=float, default=None)
    parser.add_argument("--base-hmaps-npy", type=Path, default=None)
    parser.add_argument("--base-method-name", default="prepared_inputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hmaps, stats = build_dp_msa_hmaps(
        prepared_inputs_npz=args.prepared_inputs_npz,
        region_score_csv=args.region_score_csv,
        checkpoint=args.checkpoint,
        device=args.device,
        method_name=args.method_name,
        lambda_override=args.lambda_override,
        base_hmaps_npy=args.base_hmaps_npy,
        base_method_name=args.base_method_name,
    )
    outputs = save_dp_msa_hmaps(hmaps, stats, args.outdir)
    summary = {
        "status": "ok",
        "prepared_inputs_npz": str(args.prepared_inputs_npz),
        "region_score_csv": str(args.region_score_csv),
        "checkpoint": str(args.checkpoint),
        "outdir": str(args.outdir),
        "method_name": str(args.method_name),
        "lambda_override": args.lambda_override,
        "base_hmaps_npy": str(args.base_hmaps_npy) if args.base_hmaps_npy is not None else "",
        "base_method_name": str(args.base_method_name),
        "num_cases": len(hmaps["baseline"]),
        "outputs": outputs,
    }
    summary_path = args.outdir / "dp_msa_repair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
