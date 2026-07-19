"""Run standalone AFLoc-MRSG evaluation on raw localization datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from anaprior.features.afloc_preprocessing import (
    extract_afloc_image_preprocessing,
    preprocess_afloc_image_from_path,
)
from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder
from anaprior.models.afloc_mrsg import AFLocMRSG, MRSGConfig


DEFAULT_METHOD_NAME = "afloc_mrsg"
SUPPORTED_DATASETS = ("MS_CXR", "CHEXLOCALIZE")
FORBIDDEN_ARG_NAMES = (
    "prepared_inputs_npz",
    "base_hmaps_npy",
    "region_maps_npy",
    "region_score_csv",
    "validation_gate",
    "validation_gate_source_method",
    "validation_gate_fallback_method",
    "validation_gate_method_name",
    "lambda_override",
)


@dataclass(frozen=True)
class MRSGEvalResult:
    hmaps: dict[str, dict[str, Any]]
    case_diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]


def build_hmap_lookup_key(path: Any, label_text: Any) -> str:
    return str(path) + str(label_text)


def stable_case_id(
    dataset: str,
    path: Any,
    label_text: Any,
    category: Any,
    duplicate_index: int,
) -> str:
    raw = "|".join(
        [
            str(dataset),
            str(path),
            str(label_text),
            str(category),
            str(int(duplicate_index)),
        ]
    )
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    stem = Path(str(path)).stem or "case"
    return f"{str(dataset).lower()}_{stem}_{int(duplicate_index)}_{digest}"


def coerce_rows(data_rows: Any, max_cases: int | None = None) -> pd.DataFrame:
    if isinstance(data_rows, pd.DataFrame):
        frame = data_rows.copy()
    elif isinstance(data_rows, dict):
        frame = pd.DataFrame(data_rows)
    else:
        frame = pd.DataFrame(list(data_rows))
    if max_cases is not None:
        frame = frame.iloc[:max_cases].copy()
    return frame


def normalize_eval_rows(
    data_rows: Any,
    dataset: str,
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    frame = coerce_rows(data_rows, max_cases=max_cases)
    required = {"path", "label_text", "category"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"evaluation data is missing required columns: {missing}")

    duplicates: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        path = str(record["path"])
        label_text = str(record["label_text"])
        category = str(record["category"])
        hmap_key = build_hmap_lookup_key(path, label_text)
        duplicate_index = int(duplicates.get(hmap_key, 0))
        duplicates[hmap_key] = duplicate_index + 1
        rows.append(
            {
                "case_id": stable_case_id(
                    dataset=dataset,
                    path=path,
                    label_text=label_text,
                    category=category,
                    duplicate_index=duplicate_index,
                ),
                "dataset": str(dataset),
                "path": path,
                "label_text": label_text,
                "category": category,
                "hmap_key": hmap_key,
                "duplicate_index": duplicate_index,
            }
        )
    return rows


def normalize_output_heatmap(hmap: np.ndarray) -> np.ndarray:
    array = np.asarray(hmap, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("standalone AFLoc-MRSG heatmaps must be 2D")
    finite = np.isfinite(array)
    out = np.zeros_like(array, dtype=np.float32)
    if not finite.any():
        return _resize_heatmap_for_scoring(out)
    values = array[finite]
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return _resize_heatmap_for_scoring(out)
    out[finite] = (values - low) / (high - low)
    return _resize_heatmap_for_scoring(out)


def _resize_heatmap_for_scoring(hmap: np.ndarray) -> np.ndarray:
    if hmap.shape == (224, 224):
        return hmap.astype(np.float32, copy=False)
    tensor = torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy().astype(np.float32, copy=False)


def load_checkpoint_reference(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a mapping checkpoint payload")
    required = {"model_config", "model_state_dict"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} missing checkpoint fields: {missing}")
    if payload.get("model_config") is None:
        raise ValueError(f"{path} missing checkpoint field: model_config")
    if payload.get("model_state_dict") is None:
        raise ValueError(f"{path} missing checkpoint field: model_state_dict")
    model_config = dict(payload["model_config"])
    MRSGConfig(**model_config)
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{path} model_state_dict must be a mapping")
    image_channels = payload.get("image_channels")
    if image_channels is not None and len(tuple(image_channels)) != 3:
        raise ValueError(f"{path} image_channels must contain exactly three feature levels")
    return {"path": path, "payload": dict(payload)}


def load_runtime(
    *,
    afloc_checkpoint: Path,
    checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    reference = load_checkpoint_reference(checkpoint)
    payload = reference["payload"]
    image_channels = payload.get("image_channels")
    if image_channels is None:
        raise ValueError(f"{checkpoint} missing checkpoint field: image_channels")
    config = MRSGConfig(**dict(payload["model_config"]))
    model = AFLocMRSG(
        config,
        image_channels=tuple(int(channel) for channel in image_channels),
    )
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        raise ValueError(
            f"{checkpoint} incompatible with AFLocMRSG state dict; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    encoder = FrozenAFLocMRSGEncoder.from_checkpoint(afloc_checkpoint, device=device)
    encoder.to(torch_device)
    encoder.eval()
    return {
        "afloc_checkpoint": afloc_checkpoint,
        "checkpoint": checkpoint,
        "reference": reference,
        "afloc_encoder": encoder,
        "mrsg_model": model,
    }


def default_encode_case(
    row: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    runtime = checkpoint["runtime"]
    encoder = runtime["afloc_encoder"]
    afloc_model = getattr(encoder, "afloc", None)
    if afloc_model is None or not hasattr(afloc_model, "process_img"):
        raise RuntimeError(
            "AFLoc runtime must expose afloc_encoder.afloc.process_img(paths, device, flag=0) "
            "to honor the loaded checkpoint preprocessing contract."
        )
    try:
        tensor = afloc_model.process_img([row["path"]], device, flag=0)
    except TypeError as exc:
        raise RuntimeError(
            "AFLoc runtime process_img must accept (paths, device, flag=0) for checkpoint-driven preprocessing."
        ) from exc
    tensor = tensor.to(torch.device(device))
    image_gray = None
    try:
        preprocessing = extract_afloc_image_preprocessing(afloc_model)
    except ValueError:
        preprocessing = None
    if preprocessing is not None:
        _, image_gray_tensor = preprocess_afloc_image_from_path(row["path"], preprocessing)
        image_gray = image_gray_tensor.unsqueeze(0).to(device=torch.device(device), dtype=tensor.dtype)
    phrase = str(row["label_text"])
    disease_description = str(row["category"]) or phrase
    with torch.no_grad():
        image_features, phrase_features = encoder(
            tensor,
            [phrase],
            [disease_description],
            device=device,
            image_gray=image_gray,
        )
        output = runtime["mrsg_model"](image_features, phrase_features)
    return {
        "hmap": output.final_heatmap[0, 0].detach().cpu().numpy().astype(np.float32),
        "query_route_weights": output.query_route_weights[0].detach().cpu().numpy().astype(np.float32).tolist(),
        "query_reliability": output.query_reliability[0].detach().cpu().numpy().astype(np.float32).tolist(),
    }


def build_mscxr_afloc_mrsg_hmaps(
    *,
    data_rows: Any,
    dataset: str,
    checkpoint: Path,
    afloc_checkpoint: Path | None = None,
    encode_case: Callable[[Mapping[str, Any], Mapping[str, Any], str], Mapping[str, Any]] = default_encode_case,
    device: str = "cpu",
    method_name: str = DEFAULT_METHOD_NAME,
    split: str = "test",
    max_cases: int | None = None,
) -> MRSGEvalResult:
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")

    normalized_rows = normalize_eval_rows(data_rows, dataset=dataset, max_cases=max_cases)
    checkpoint_ref = load_checkpoint_reference(Path(checkpoint))
    checkpoint_bundle: dict[str, Any] = {
        "path": Path(checkpoint),
        "reference": checkpoint_ref,
        "afloc_checkpoint": Path(afloc_checkpoint) if afloc_checkpoint is not None else None,
    }
    if encode_case is default_encode_case:
        if afloc_checkpoint is None:
            raise ValueError("afloc_checkpoint is required for real AFLoc-MRSG inference")
        checkpoint_bundle["runtime"] = load_runtime(
            afloc_checkpoint=Path(afloc_checkpoint),
            checkpoint=Path(checkpoint),
            device=device,
        )

    hmaps: dict[str, dict[str, Any]] = {}
    case_diagnostics: list[dict[str, Any]] = []
    for row in normalized_rows:
        encoded = dict(encode_case(row, checkpoint_bundle, device))
        hmap = normalize_output_heatmap(np.asarray(encoded["hmap"], dtype=np.float32))
        case_id = str(row["case_id"])
        hmaps[case_id] = {
            "case_id": case_id,
            "path": str(row["path"]),
            "label_text": str(row["label_text"]),
            "category": str(row["category"]),
            "hmap": hmap,
        }
        if "query_route_weights" in encoded:
            hmaps[case_id]["query_route_weights"] = list(encoded["query_route_weights"])
        if "query_reliability" in encoded:
            hmaps[case_id]["query_reliability"] = list(encoded["query_reliability"])
        case_diagnostics.append(
            {
                "case_id": case_id,
                "dataset": str(dataset),
                "path": str(row["path"]),
                "label_text": str(row["label_text"]),
                "category": str(row["category"]),
                "hmap_key": str(row["hmap_key"]),
                "duplicate_index": int(row["duplicate_index"]),
                "hmap_min": round(float(hmap.min()), 6),
                "hmap_max": round(float(hmap.max()), 6),
                "query_route_weights": list(encoded.get("query_route_weights", [])),
                "query_reliability": list(encoded.get("query_reliability", [])),
            }
        )

    duplicate_hmap_keys = {
        row["hmap_key"]
        for row in normalized_rows
        if sum(1 for item in normalized_rows if item["hmap_key"] == row["hmap_key"]) > 1
    }
    summary = {
        "status": "ok",
        "dataset": str(dataset),
        "split": str(split),
        "method_name": str(method_name),
        "checkpoint": str(checkpoint),
        "afloc_checkpoint": str(afloc_checkpoint) if afloc_checkpoint is not None else "",
        "num_cases": int(len(normalized_rows)),
        "num_hmaps": int(len(hmaps)),
        "num_duplicate_hmap_keys": int(len(duplicate_hmap_keys)),
        "external_evaluation": bool(dataset == "CHEXLOCALIZE"),
        "uses_dcem": False,
        "uses_region_predictor": False,
        "uses_validation_gate": False,
        "checkpoint_selection_participant": False,
    }
    return MRSGEvalResult(hmaps=hmaps, case_diagnostics=case_diagnostics, summary=summary)


def save_mscxr_afloc_mrsg_outputs(
    result: MRSGEvalResult,
    *,
    outdir: Path,
    method_name: str = DEFAULT_METHOD_NAME,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    method_dir = outdir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    hmaps_path = method_dir / "hmaps.npy"
    np.save(hmaps_path, result.hmaps)
    diagnostics_path = outdir / "mrsg_case_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(result.case_diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path = outdir / "mrsg_eval_summary.json"
    summary_payload = dict(result.summary)
    summary_payload["outputs"] = {
        "hmaps": str(hmaps_path),
        "case_diagnostics": str(diagnostics_path),
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "hmaps": str(hmaps_path),
        "case_diagnostics": str(diagnostics_path),
        "summary": str(summary_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standalone AFLoc-MRSG heatmaps on raw localization data.")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--afloc-checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method-name", default=DEFAULT_METHOD_NAME)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--ms-cxr-json", type=Path, default=None)
    parser.add_argument("--mimic-img-dir", type=Path, default=None)
    parser.add_argument("--prepared-inputs-npz", type=Path, default=None)
    parser.add_argument("--base-hmaps-npy", type=Path, default=None)
    parser.add_argument("--region-maps-npy", type=Path, default=None)
    parser.add_argument("--region-score-csv", type=Path, default=None)
    parser.add_argument("--validation-gate", action="store_true")
    parser.add_argument("--validation-gate-source-method", default=None)
    parser.add_argument("--validation-gate-fallback-method", default=None)
    parser.add_argument("--validation-gate-method-name", default=None)
    parser.add_argument("--lambda-override", type=float, default=None)
    return parser.parse_args(argv)


def validate_forbidden_args(args: argparse.Namespace) -> None:
    provided: list[str] = []
    for name in FORBIDDEN_ARG_NAMES:
        value = getattr(args, name)
        if isinstance(value, bool):
            if value:
                provided.append(f"--{name.replace('_', '-')}")
            continue
        if value not in (None, ""):
            provided.append(f"--{name.replace('_', '-')}")
    if provided:
        raise ValueError(
            "standalone AFLoc-MRSG raw evaluation does not accept repair-pipeline inputs: "
            + ", ".join(provided)
        )


def load_dataset_rows(
    dataset: str,
    split: str,
    max_cases: int | None = None,
    ms_cxr_json: Path | None = None,
    mimic_img_dir: Path | None = None,
) -> Any:
    from localization.datasets import load_data

    kwargs: dict[str, Any] = {}
    if dataset == "CHEXLOCALIZE":
        kwargs["split"] = split
    if dataset in {"MS_CXR", "MS_CXR_CLS"}:
        if ms_cxr_json is not None:
            kwargs["ms_cxr_json"] = ms_cxr_json
        if mimic_img_dir is not None:
            kwargs["mimic_img_dir"] = mimic_img_dir
    data = load_data(dataset=dataset, **kwargs)
    return coerce_rows(data, max_cases=max_cases)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_forbidden_args(args)
    rows = load_dataset_rows(
        args.dataset,
        split=args.split,
        max_cases=args.max_cases,
        ms_cxr_json=args.ms_cxr_json,
        mimic_img_dir=args.mimic_img_dir,
    )
    result = build_mscxr_afloc_mrsg_hmaps(
        data_rows=rows,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        afloc_checkpoint=args.afloc_checkpoint,
        device=args.device,
        method_name=args.method_name,
        split=args.split,
        max_cases=args.max_cases,
    )
    outputs = save_mscxr_afloc_mrsg_outputs(
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
