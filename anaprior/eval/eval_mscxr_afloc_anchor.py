"""Export frozen AFLoc phrase-patch anchors on raw localization datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from anaprior.eval.eval_mscxr_afloc_mrsg import (
    load_dataset_rows,
    normalize_eval_rows,
    normalize_output_heatmap,
)
from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder
from anaprior.models.afloc_mrsg.anchor import compute_afloc_phrase_anchor


DEFAULT_METHOD_NAME = "afloc_anchor"
SUPPORTED_DATASETS = ("MS_CXR", "CHEXLOCALIZE")


@dataclass(frozen=True)
class AnchorEvalResult:
    hmaps: dict[str, dict[str, Any]]
    case_diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def load_anchor_runtime(
    *,
    afloc_checkpoint: Path,
    bert_type: Path,
    device: str,
    hf_local_files_only: bool,
) -> dict[str, Any]:
    if not afloc_checkpoint.is_file():
        raise FileNotFoundError(f"afloc_checkpoint not found: {afloc_checkpoint}")
    if not bert_type.is_dir():
        raise FileNotFoundError(f"bert_type directory not found: {bert_type}")
    os.environ["AFLOC_BERT_TYPE"] = str(bert_type)
    if hf_local_files_only:
        os.environ["AFLOC_HF_LOCAL_FILES_ONLY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    encoder = FrozenAFLocMRSGEncoder.from_checkpoint(afloc_checkpoint, device=device)
    encoder.eval()
    return {
        "afloc_checkpoint": afloc_checkpoint,
        "bert_type": bert_type,
        "encoder": encoder,
    }


def default_encode_anchor_case(
    row: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    encoder = runtime["encoder"]
    afloc_model = getattr(encoder, "afloc", None)
    if afloc_model is None or not hasattr(afloc_model, "process_img"):
        raise RuntimeError("AFLoc runtime must expose process_img(paths, device, flag=0)")
    image = afloc_model.process_img([str(row["path"])], device, flag=0)
    image_features = encoder.encode_images(image)
    phrase = str(row["label_text"])
    description = str(row["category"]) or phrase
    phrase_features = encoder.encode_phrases(
        [phrase],
        [description],
        device=device,
    )
    anchor, confidence, _ = compute_afloc_phrase_anchor(image_features, phrase_features)
    return {
        "hmap": anchor[0, 0].detach().cpu().numpy().astype(np.float32),
        "confidence": confidence[0, 0].detach().cpu().numpy().astype(np.float32),
        "scale_agreement": float(confidence.mean().detach().cpu().item()),
    }


def build_mscxr_afloc_anchor_hmaps(
    *,
    data_rows: Any,
    dataset: str,
    afloc_checkpoint: Path,
    encode_case: Callable[[Mapping[str, Any], Mapping[str, Any], str], Mapping[str, Any]] = default_encode_anchor_case,
    runtime: Mapping[str, Any] | None = None,
    device: str = "cpu",
    method_name: str = DEFAULT_METHOD_NAME,
    split: str = "test",
    max_cases: int | None = None,
) -> AnchorEvalResult:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    normalized_rows = normalize_eval_rows(data_rows, dataset=dataset, max_cases=max_cases)
    runtime_bundle: dict[str, Any] = dict(runtime or {})
    runtime_bundle.setdefault("afloc_checkpoint", Path(afloc_checkpoint))
    if encode_case is default_encode_anchor_case and "encoder" not in runtime_bundle:
        raise ValueError("runtime with frozen AFLoc encoder is required for real anchor inference")

    hmaps: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    zero_variance_count = 0
    hmap_stds: list[float] = []
    confidence_means: list[float] = []
    duplicate_keys: dict[str, int] = {}
    for row in normalized_rows:
        duplicate_keys[str(row["hmap_key"])] = duplicate_keys.get(str(row["hmap_key"]), 0) + 1
        encoded = dict(encode_case(row, runtime_bundle, device))
        raw_hmap = np.asarray(encoded["hmap"], dtype=np.float32)
        raw_std = float(np.nanstd(raw_hmap))
        hmap = normalize_output_heatmap(raw_hmap)
        confidence = np.asarray(encoded.get("confidence", np.ones_like(raw_hmap)), dtype=np.float32)
        if raw_std <= 1.0e-8:
            zero_variance_count += 1
        hmap_stds.append(float(hmap.std()))
        confidence_means.append(float(np.nanmean(confidence)))
        case_id = str(row["case_id"])
        hmaps[case_id] = {
            "case_id": case_id,
            "path": str(row["path"]),
            "label_text": str(row["label_text"]),
            "category": str(row["category"]),
            "hmap": hmap,
        }
        diagnostics.append(
            {
                "case_id": case_id,
                "path": str(row["path"]),
                "label_text": str(row["label_text"]),
                "category": str(row["category"]),
                "hmap_key": str(row["hmap_key"]),
                "duplicate_index": int(row["duplicate_index"]),
                "hmap_mean": float(hmap.mean()),
                "hmap_std": float(hmap.std()),
                "confidence_mean": float(np.nanmean(confidence)),
                "scale_agreement": float(encoded.get("scale_agreement", np.nanmean(confidence))),
            }
        )

    checkpoint_path = Path(afloc_checkpoint)
    summary = {
        "status": "ok",
        "dataset": str(dataset),
        "split": str(split),
        "method_name": str(method_name),
        "num_cases": int(len(normalized_rows)),
        "num_hmaps": int(len(hmaps)),
        "num_duplicate_hmap_keys": int(sum(count > 1 for count in duplicate_keys.values())),
        "zero_variance_count": int(zero_variance_count),
        "mean_hmap_std": float(np.mean(hmap_stds)) if hmap_stds else 0.0,
        "mean_anchor_confidence": float(np.mean(confidence_means)) if confidence_means else 0.0,
        "afloc_checkpoint": str(checkpoint_path),
        "afloc_checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else "",
        "git_commit": current_git_commit(),
        "uses_spatial_annotations": False,
        "uses_dcem": False,
        "uses_region_predictor": False,
        "uses_validation_gate": False,
        "uses_mrsg_checkpoint": False,
        "checkpoint_selection_participant": False,
    }
    return AnchorEvalResult(hmaps=hmaps, case_diagnostics=diagnostics, summary=summary)


def save_mscxr_afloc_anchor_outputs(
    result: AnchorEvalResult,
    *,
    outdir: Path,
    method_name: str = DEFAULT_METHOD_NAME,
) -> dict[str, str]:
    method_dir = outdir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    hmaps_path = method_dir / "hmaps.npy"
    diagnostics_path = outdir / "anchor_case_diagnostics.json"
    summary_path = outdir / "anchor_eval_summary.json"
    np.save(hmaps_path, result.hmaps)
    diagnostics_path.write_text(
        json.dumps(result.case_diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = dict(result.summary)
    summary["outputs"] = {
        "hmaps": str(hmaps_path),
        "case_diagnostics": str(diagnostics_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "hmaps": str(hmaps_path),
        "case_diagnostics": str(diagnostics_path),
        "summary": str(summary_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen AFLoc phrase-patch anchor heatmaps.")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--afloc-checkpoint", required=True, type=Path)
    parser.add_argument("--bert-type", required=True, type=Path)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--ms-cxr-json", type=Path, default=None)
    parser.add_argument("--mimic-img-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_dataset_rows(
        args.dataset,
        split=args.split,
        max_cases=args.max_cases,
        ms_cxr_json=args.ms_cxr_json,
        mimic_img_dir=args.mimic_img_dir,
    )
    runtime = load_anchor_runtime(
        afloc_checkpoint=args.afloc_checkpoint,
        bert_type=args.bert_type,
        device=args.device,
        hf_local_files_only=args.hf_local_files_only,
    )
    result = build_mscxr_afloc_anchor_hmaps(
        data_rows=rows,
        dataset=args.dataset,
        afloc_checkpoint=args.afloc_checkpoint,
        runtime=runtime,
        device=args.device,
        method_name=args.method_name,
        split=args.split,
        max_cases=args.max_cases,
    )
    outputs = save_mscxr_afloc_anchor_outputs(
        result,
        outdir=args.outdir,
        method_name=args.method_name,
    )
    payload = dict(result.summary)
    payload["outputs"] = outputs
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
