"""Export the exact frozen-AFLoc heatmap used by the original localization pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from anaprior.eval.eval_mscxr_afloc_mrsg import load_dataset_rows, normalize_eval_rows
from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder
from anaprior.features.afloc_official_heatmap import compute_official_afloc_heatmap


DEFAULT_METHOD_NAME = "afloc_official_anchor"


@dataclass(frozen=True)
class OfficialAnchorResult:
    hmaps: dict[str, dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]


def load_official_runtime(
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
    return {"encoder": encoder, "afloc_checkpoint": afloc_checkpoint}


def default_encode_official_case(
    row: Mapping[str, Any],
    runtime: Mapping[str, Any],
    device: str,
) -> np.ndarray:
    encoder = runtime["encoder"]
    model = encoder.afloc
    model.eval()
    model.cfg.data.image.imsize = 224
    flag = model.cfg.data.image.flag
    image = model.process_img(
        [str(row["path"])],
        device,
        equalize_hist=False,
        flag=flag,
    )
    text = model.process_text(str(row["label_text"]), device)
    with torch.no_grad():
        local, _, _, _ = model.image_encoder_forward(image.to(device))
        text_output = model.text_encoder_forward(
            text["caption_ids"].to(device),
            text["attention_mask"].to(device),
            text["token_type_ids"].to(device),
        )
    heatmap = compute_official_afloc_heatmap(local, text_output["report_embeddings"])
    return heatmap.cpu().numpy().astype(np.float32, copy=False)


def build_official_anchor_hmaps(
    *,
    data_rows: Any,
    dataset: str,
    afloc_checkpoint: Path,
    encode_case: Callable[[Mapping[str, Any], Mapping[str, Any], str], np.ndarray] = default_encode_official_case,
    runtime: Mapping[str, Any] | None = None,
    device: str = "cpu",
    split: str = "test",
    max_cases: int | None = None,
    method_name: str = DEFAULT_METHOD_NAME,
) -> OfficialAnchorResult:
    rows = normalize_eval_rows(data_rows, dataset=dataset, max_cases=max_cases)
    runtime_bundle = dict(runtime or {})
    runtime_bundle.setdefault("afloc_checkpoint", Path(afloc_checkpoint))
    if encode_case is default_encode_official_case and "encoder" not in runtime_bundle:
        raise ValueError("runtime with frozen AFLoc encoder is required")

    hmaps: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        heatmap = np.asarray(encode_case(row, runtime_bundle, device), dtype=np.float32)
        if heatmap.ndim != 2:
            raise ValueError("official AFLoc heatmap must be 2D")
        case_id = str(row["case_id"])
        payload = {
            "case_id": case_id,
            "path": str(row["path"]),
            "label_text": str(row["label_text"]),
            "category": str(row["category"]),
            "hmap_key": str(row["hmap_key"]),
            "hmap": heatmap,
        }
        hmaps[case_id] = payload
        diagnostics.append({
            "case_id": case_id,
            "hmap_key": str(row["hmap_key"]),
            "shape": list(heatmap.shape),
            "mean": float(np.nanmean(heatmap)),
            "std": float(np.nanstd(heatmap)),
        })

    return OfficialAnchorResult(
        hmaps=hmaps,
        diagnostics=diagnostics,
        summary={
            "status": "ok",
            "dataset": dataset,
            "split": split,
            "method_name": method_name,
            "num_cases": len(rows),
            "num_hmaps": len(hmaps),
            "afloc_checkpoint": str(afloc_checkpoint),
            "official_formula": "official_iel_x_global_report_gaussian_1.5_bilinear",
            "raw_similarity_preserved": True,
            "uses_spatial_annotations": False,
            "uses_dcem": False,
            "uses_region_predictor": False,
            "uses_mrsg_checkpoint": False,
        },
    )


def save_official_anchor_outputs(
    result: OfficialAnchorResult,
    *,
    outdir: Path,
    method_name: str = DEFAULT_METHOD_NAME,
) -> dict[str, str]:
    method_dir = outdir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    hmaps_path = method_dir / "hmaps.npy"
    summary_path = outdir / "official_anchor_eval_summary.json"
    diagnostics_path = outdir / "official_anchor_case_diagnostics.json"
    np.save(hmaps_path, result.hmaps)
    diagnostics_path.write_text(json.dumps(result.diagnostics, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    return {
        "hmaps": str(hmaps_path),
        "summary": str(summary_path),
        "diagnostics": str(diagnostics_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="MS_CXR", choices=("MS_CXR", "MS_CXR_CLS"))
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--afloc-checkpoint", required=True, type=Path)
    parser.add_argument("--bert-type", required=True, type=Path)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--ms-cxr-json", required=True, type=Path)
    parser.add_argument("--mimic-img-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_rows = load_dataset_rows(
        args.dataset,
        split=args.split,
        max_cases=args.max_cases,
        ms_cxr_json=args.ms_cxr_json,
        mimic_img_dir=args.mimic_img_dir,
    )
    runtime = load_official_runtime(
        afloc_checkpoint=args.afloc_checkpoint,
        bert_type=args.bert_type,
        device=args.device,
        hf_local_files_only=args.hf_local_files_only,
    )
    result = build_official_anchor_hmaps(
        data_rows=data_rows,
        dataset=args.dataset,
        afloc_checkpoint=args.afloc_checkpoint,
        runtime=runtime,
        device=args.device,
        split=args.split,
        max_cases=args.max_cases,
        method_name=args.method_name,
    )
    outputs = save_official_anchor_outputs(result, outdir=args.outdir, method_name=args.method_name)
    print(json.dumps({**result.summary, "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
