"""Extract AFLoc spatial feature maps for dense DP-MSA training/evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from anaprior.eval.eval_mscxr_learned_repair import _load_prepared_inputs
from anaprior.features.extract_region_features import build_afloc_extractor


SpatialFeatureExtractor = Callable[[str, Optional[Path]], torch.Tensor]


def image_path_from_case_id(case_id: str) -> Path | None:
    """Extract an image path prefix from a Stage C case id."""

    text = str(case_id)
    match = re.search(r"\.(jpg|jpeg|png)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return Path(text[: match.end()])


def build_dp_msa_spatial_feature_cache(
    prepared_inputs_npz: Path,
    output_path: Path,
    extract_spatial_features: SpatialFeatureExtractor,
    max_cases: int | None = None,
    feature_dtype: str = "float16",
    progress_every: int = 100,
) -> dict[str, object]:
    if feature_dtype not in {"float16", "float32"}:
        raise ValueError("feature_dtype must be float16 or float32")
    inputs = _load_prepared_inputs(prepared_inputs_npz)
    if max_cases is not None:
        inputs = inputs[: int(max_cases)]
    if not inputs:
        raise ValueError("no prepared inputs loaded")

    features = []
    case_ids = []
    dicom_ids = []
    image_paths = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, item in enumerate(inputs, start=1):
        case_id = str(item.case_id)
        if case_id in seen:
            continue
        seen.add(case_id)
        image_path = image_path_from_case_id(case_id)
        try:
            tensor = extract_spatial_features(str(item.dicom_id), image_path)
        except Exception as exc:  # pragma: no cover - real extractor failures.
            errors.append({"case_id": case_id, "dicom_id": str(item.dicom_id), "error": str(exc)})
            continue
        if tensor.ndim != 3:
            raise ValueError("extract_spatial_features must return [C,H,W]")
        features.append(tensor.detach().cpu().float())
        case_ids.append(case_id)
        dicom_ids.append(str(item.dicom_id))
        image_paths.append(str(image_path) if image_path is not None else "")
        if progress_every > 0 and idx % int(progress_every) == 0:
            print(
                "[extract_dp_msa_spatial_features] "
                f"processed={idx}/{len(inputs)} cached={len(case_ids)} errors={len(errors)}",
                flush=True,
            )

    if not features:
        raise ValueError("no spatial features were extracted")
    stacked = torch.stack(features, dim=0)
    if feature_dtype == "float16":
        stacked = stacked.to(torch.float16)
    else:
        stacked = stacked.to(torch.float32)

    payload = {
        "case_ids": case_ids,
        "dicom_ids": dicom_ids,
        "image_paths": image_paths,
        "spatial_features": stacked,
        "feature_dtype": feature_dtype,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report = {
        "status": "ok" if not errors else "partial",
        "prepared_inputs_npz": str(prepared_inputs_npz),
        "output_path": str(output_path),
        "num_inputs": int(len(inputs)),
        "num_cached": int(len(case_ids)),
        "feature_shape": list(stacked.shape),
        "feature_dtype": feature_dtype,
        "errors": errors[:20],
        "num_errors": int(len(errors)),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AFLoc spatial feature maps for dense DP-MSA.")
    parser.add_argument("--prepared-inputs-npz", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--bert-type", default=None, help="Local ClinicalBERT path override for AFLoc.")
    parser.add_argument("--hf-local-files-only", default="1", help="Set AFLOC_HF_LOCAL_FILES_ONLY before loading AFLoc.")
    parser.add_argument("--feature-level", choices=("img_emb_l", "img_emb_l2"), default="img_emb_l")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bert_type:
        os.environ["AFLOC_BERT_TYPE"] = str(args.bert_type)
    if args.hf_local_files_only:
        os.environ["AFLOC_HF_LOCAL_FILES_ONLY"] = str(args.hf_local_files_only)
    extractor = build_afloc_extractor(args.ckpt, args.device, args.feature_level)
    report = build_dp_msa_spatial_feature_cache(
        prepared_inputs_npz=args.prepared_inputs_npz,
        output_path=args.output_path,
        extract_spatial_features=extractor,
        max_cases=args.max_cases,
        feature_dtype=args.feature_dtype,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
