"""Extract and cache AFLoc region features for predictor training."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from anaprior.features.region_pooling import RegionBox, pool_region_features


FeatureExtractor = Callable[[str, Optional[Path]], torch.Tensor]


@dataclass(frozen=True)
class FeatureCacheConfig:
    region_table_csv: Path
    output_path: Path
    image_root: Path | None = None
    image_ext: str = ".jpg"
    image_size: tuple[int, int] = (224, 224)
    max_rows: int | None = None
    progress_every: int = 500
    feature_dtype: str = "float32"
    metadata_mode: str = "full"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache pooled AFLoc region features.")
    parser.add_argument("--region-table-csv", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--ckpt", type=Path, default=None, help="AFLoc checkpoint for real extraction.")
    parser.add_argument("--image-root", type=Path, default=None, help="Root containing MIMIC-CXR jpg files.")
    parser.add_argument("--image-ext", default=".jpg")
    parser.add_argument("--feature-level", choices=("img_emb_l", "img_emb_l2"), default="img_emb_l")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--feature-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--metadata-mode", choices=("full", "none"), default="full")
    return parser.parse_args()


def read_region_table(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def finding_vocab_from_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    findings = sorted({row["finding"] for row in rows})
    return {finding: idx for idx, finding in enumerate(findings)}


def _float_value(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing {key} for row with dicom_id={row.get('dicom_id')}")
    return float(value)


def row_has_bbox(row: dict[str, str]) -> bool:
    return all(str(row.get(key, "")).strip() for key in ("x1", "y1", "x2", "y2"))


def box_from_row(row: dict[str, str]) -> RegionBox:
    return RegionBox(
        name=str(row["region"]),
        x1=_float_value(row, "x1"),
        y1=_float_value(row, "y1"),
        x2=_float_value(row, "x2"),
        y2=_float_value(row, "y2"),
    )


def image_path_for_row(row: dict[str, str], image_root: Path | None, image_ext: str) -> Path | None:
    if image_root is None:
        return None
    dicom_id = str(row["dicom_id"])
    subject_id = str(row.get("subject_id", ""))
    study_id = str(row.get("study_id", ""))
    if subject_id and study_id:
        prefix = f"p{subject_id[:2]}"
        return image_root / prefix / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}{image_ext}"
    return image_root / f"{dicom_id}{image_ext}"


def build_feature_cache_from_table(
    config: FeatureCacheConfig,
    extract_local_features: FeatureExtractor,
    progress_logger: Optional[Callable[[str], None]] = None,
) -> dict[str, object]:
    if config.feature_dtype not in {"float32", "float16"}:
        raise ValueError("feature_dtype must be 'float32' or 'float16'")
    if config.metadata_mode not in {"full", "none"}:
        raise ValueError("metadata_mode must be 'full' or 'none'")

    rows = read_region_table(config.region_table_csv, max_rows=config.max_rows)
    finding_vocab = finding_vocab_from_rows(rows)
    rows_by_dicom: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_dicom[str(row["dicom_id"])].append(row)
    total_dicoms = len(rows_by_dicom)
    progress_every = max(int(config.progress_every), 0)
    if progress_logger is not None:
        progress_logger(
            "[extract_region_features] "
            f"loaded_rows={len(rows)} unique_dicoms={total_dicoms} findings={len(finding_vocab)}"
        )

    region_features = []
    labels = []
    finding_ids = []
    valid_mask = []
    cell_counts = []
    region_names = []
    metadata = []
    extraction_errors: list[dict[str, str]] = []
    skipped_missing_bbox: list[dict[str, str]] = []

    for dicom_index, (dicom_id, dicom_rows) in enumerate(rows_by_dicom.items(), start=1):
        image_path = image_path_for_row(dicom_rows[0], config.image_root, config.image_ext)
        try:
            local_features = extract_local_features(dicom_id, image_path)
        except Exception as exc:  # pragma: no cover - exercised by real CLI failures.
            extraction_errors.append({"dicom_id": dicom_id, "error": str(exc)})
            continue

        rows_with_bbox = []
        for row in dicom_rows:
            if row_has_bbox(row):
                rows_with_bbox.append(row)
            else:
                skipped_missing_bbox.append(
                    {
                        "dicom_id": str(row.get("dicom_id", "")),
                        "region": str(row.get("region", "")),
                        "finding": str(row.get("finding", "")),
                    }
                )
        if not rows_with_bbox:
            continue

        boxes = [box_from_row(row) for row in rows_with_bbox]
        pooled = pool_region_features(local_features, boxes, image_size=config.image_size)
        pooled_features = pooled.features
        if pooled_features.ndim != 2:
            raise ValueError("extract_local_features must return a single-image [C,H,W] tensor")

        for idx, row in enumerate(rows_with_bbox):
            region_features.append(pooled_features[idx].detach().cpu())
            labels.append(float(row["label"]))
            finding_ids.append(finding_vocab[row["finding"]])
            valid_mask.append(bool(pooled.valid_mask[idx].item()))
            cell_counts.append(int(pooled.cell_counts[idx].item()))
            region_names.append(str(row["region"]))
            metadata.append(
                {
                    "dicom_id": dicom_id,
                    "subject_id": row.get("subject_id", ""),
                    "study_id": row.get("study_id", ""),
                    "region": row.get("region", ""),
                    "finding": row.get("finding", ""),
                    "label_source": row.get("label_source", ""),
                }
            )
        if progress_logger is not None and (
            dicom_index == 1
            or (progress_every > 0 and dicom_index % progress_every == 0)
            or dicom_index == total_dicoms
        ):
            progress_logger(
                "[extract_region_features] "
                f"processed_dicoms={dicom_index}/{total_dicoms} "
                f"cached_rows={len(labels)} "
                f"skipped_missing_bbox_rows={len(skipped_missing_bbox)} "
                f"extraction_errors={len(extraction_errors)}"
            )

    if region_features:
        feature_tensor = torch.stack(region_features, dim=0)
    else:
        feature_tensor = torch.empty((0, 0), dtype=torch.float32)
    if config.feature_dtype == "float16":
        feature_tensor = feature_tensor.to(torch.float16)
    else:
        feature_tensor = feature_tensor.to(torch.float32)

    payload = {
        "region_features": feature_tensor,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "finding_ids": torch.tensor(finding_ids, dtype=torch.long),
        "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
        "cell_counts": torch.tensor(cell_counts, dtype=torch.long),
        "finding_vocab": finding_vocab,
        "region_names": region_names,
    }
    if config.metadata_mode == "full":
        payload["metadata"] = metadata
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, config.output_path)

    report = {
        "status": "ok" if not extraction_errors and not skipped_missing_bbox else "partial",
        "region_table_csv": str(config.region_table_csv),
        "output_path": str(config.output_path),
        "rows": len(labels),
        "dicoms": len(rows_by_dicom),
        "feature_dim": int(feature_tensor.shape[1]) if feature_tensor.ndim == 2 and feature_tensor.numel() else 0,
        "feature_dtype": config.feature_dtype,
        "metadata_mode": config.metadata_mode,
        "finding_vocab": finding_vocab,
        "extraction_errors": extraction_errors,
        "skipped_missing_bbox_rows": len(skipped_missing_bbox),
        "missing_bbox_examples": skipped_missing_bbox[:20],
    }
    report_path = config.output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def exit_code_from_report(report: dict[str, object]) -> int:
    """Return CLI exit code.

    Missing bbox rows are common in Chest ImaGenome-derived tables and are not
    fatal when at least some rows were cached. Real extraction errors remain
    fatal so shell runners stop before using an incomplete feature file.
    """

    extraction_errors = report.get("extraction_errors", [])
    if extraction_errors:
        return 2
    if int(report.get("rows", 0)) <= 0:
        return 2
    return 0


def build_afloc_extractor(ckpt: Path, device: str, feature_level: str) -> FeatureExtractor:
    """Build a real AFLoc extractor. Kept behind CLI so unit tests stay light."""

    from PIL import Image
    import torchvision.transforms as transforms

    from afloc.builder import load_model

    model = load_model(str(ckpt), device=device)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    def extract(_dicom_id: str, image_path: Path | None) -> torch.Tensor:
        if image_path is None:
            raise ValueError("image_path is required for real AFLoc extraction")
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb_l, img_emb_l2, _img_emb_lf, _img_emb_g = model.image_encoder_forward(tensor)
        selected = img_emb_l if feature_level == "img_emb_l" else img_emb_l2
        return selected.squeeze(0).detach().cpu()

    return extract


def main() -> int:
    args = parse_args()
    if args.ckpt is None:
        raise SystemExit("--ckpt is required for CLI feature extraction")
    extractor = build_afloc_extractor(args.ckpt, args.device, args.feature_level)
    report = build_feature_cache_from_table(
        FeatureCacheConfig(
            region_table_csv=args.region_table_csv,
            output_path=args.output_path,
            image_root=args.image_root,
            image_ext=args.image_ext,
            max_rows=args.max_rows,
            progress_every=args.progress_every,
            feature_dtype=args.feature_dtype,
            metadata_mode=args.metadata_mode,
        ),
        extract_local_features=extractor,
        progress_logger=lambda message: print(message, flush=True),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return exit_code_from_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
