import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from anaprior.features.extract_region_features import (
    FeatureCacheConfig,
    build_feature_cache_from_table,
    exit_code_from_report,
    finding_vocab_from_rows,
)


def write_region_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "subject_id",
        "study_id",
        "dicom_id",
        "image_id",
        "view",
        "region",
        "object_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "finding",
        "label",
        "label_source",
    ]
    rows = [
        {
            "split": "train",
            "subject_id": "10000001",
            "study_id": "50000001",
            "dicom_id": "dicom-a",
            "image_id": "dicom-a",
            "view": "PA",
            "region": "left half",
            "object_id": "dicom-a_left half",
            "x1": "0",
            "y1": "0",
            "x2": "112",
            "y2": "224",
            "width": "112",
            "height": "224",
            "finding": "Pneumothorax",
            "label": "1",
            "label_source": "explicit_yes",
        },
        {
            "split": "train",
            "subject_id": "10000001",
            "study_id": "50000001",
            "dicom_id": "dicom-a",
            "image_id": "dicom-a",
            "view": "PA",
            "region": "right half",
            "object_id": "dicom-a_right half",
            "x1": "112",
            "y1": "0",
            "x2": "224",
            "y2": "224",
            "width": "112",
            "height": "224",
            "finding": "Pleural Effusion",
            "label": "0",
            "label_source": "explicit_no",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_region_table_with_missing_bbox(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "subject_id",
        "study_id",
        "dicom_id",
        "image_id",
        "view",
        "region",
        "object_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "finding",
        "label",
        "label_source",
    ]
    rows = [
        {
            "split": "train",
            "subject_id": "10000001",
            "study_id": "50000001",
            "dicom_id": "dicom-a",
            "image_id": "dicom-a",
            "view": "PA",
            "region": "left half",
            "object_id": "dicom-a_left half",
            "x1": "0",
            "y1": "0",
            "x2": "112",
            "y2": "224",
            "width": "112",
            "height": "224",
            "finding": "Pneumothorax",
            "label": "1",
            "label_source": "explicit_yes",
        },
        {
            "split": "train",
            "subject_id": "10000001",
            "study_id": "50000001",
            "dicom_id": "dicom-a",
            "image_id": "dicom-a",
            "view": "PA",
            "region": "region without box",
            "object_id": "",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "width": "",
            "height": "",
            "finding": "Pleural Effusion",
            "label": "1",
            "label_source": "explicit_yes",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_finding_vocab_is_deterministic(tmp_path: Path) -> None:
    table_path = tmp_path / "region_table.csv"
    write_region_table(table_path)
    rows = list(csv.DictReader(table_path.open(encoding="utf-8")))

    assert finding_vocab_from_rows(rows) == {"Pleural Effusion": 0, "Pneumothorax": 1}


def test_build_feature_cache_uses_one_extraction_per_dicom(tmp_path: Path) -> None:
    table_path = tmp_path / "region_table.csv"
    output_path = tmp_path / "features.pt"
    write_region_table(table_path)
    calls = []

    def fake_extract(dicom_id: str, image_path: Path | None) -> torch.Tensor:
        calls.append((dicom_id, image_path))
        assert dicom_id == "dicom-a"
        return torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    report = build_feature_cache_from_table(
        config=FeatureCacheConfig(
            region_table_csv=table_path,
            output_path=output_path,
            image_root=None,
            image_ext=".jpg",
            image_size=(224, 224),
        ),
        extract_local_features=fake_extract,
    )

    assert report["status"] == "ok"
    assert report["rows"] == 2
    assert report["dicoms"] == 1
    assert len(calls) == 1
    payload = torch.load(output_path, weights_only=False)
    assert payload["region_features"].shape == (2, 1)
    assert payload["labels"].tolist() == [1.0, 0.0]
    assert payload["finding_ids"].tolist() == [1, 0]
    assert payload["region_names"] == ["left half", "right half"]
    assert payload["metadata"][0]["dicom_id"] == "dicom-a"


def test_build_feature_cache_skips_rows_without_bbox(tmp_path: Path) -> None:
    table_path = tmp_path / "region_table_missing_bbox.csv"
    output_path = tmp_path / "features.pt"
    write_region_table_with_missing_bbox(table_path)

    def fake_extract(_dicom_id: str, _image_path: Path | None) -> torch.Tensor:
        return torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    report = build_feature_cache_from_table(
        config=FeatureCacheConfig(
            region_table_csv=table_path,
            output_path=output_path,
            image_root=None,
            image_ext=".jpg",
            image_size=(224, 224),
        ),
        extract_local_features=fake_extract,
    )

    assert report["status"] == "partial"
    assert report["rows"] == 1
    assert report["skipped_missing_bbox_rows"] == 1
    assert report["missing_bbox_examples"][0]["dicom_id"] == "dicom-a"
    payload = torch.load(output_path, weights_only=False)
    assert payload["region_features"].shape == (1, 1)
    assert payload["region_names"] == ["left half"]


def test_build_feature_cache_can_store_half_precision_features(tmp_path: Path) -> None:
    table_path = tmp_path / "region_table.csv"
    output_path = tmp_path / "features.pt"
    write_region_table(table_path)

    def fake_extract(_dicom_id: str, _image_path: Path | None) -> torch.Tensor:
        return torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    report = build_feature_cache_from_table(
        config=FeatureCacheConfig(
            region_table_csv=table_path,
            output_path=output_path,
            image_root=None,
            image_ext=".jpg",
            image_size=(224, 224),
            feature_dtype="float16",
        ),
        extract_local_features=fake_extract,
    )

    payload = torch.load(output_path, weights_only=False)
    assert payload["region_features"].dtype == torch.float16
    assert report["feature_dtype"] == "float16"


def test_build_feature_cache_can_omit_metadata_for_training_cache(tmp_path: Path) -> None:
    table_path = tmp_path / "region_table.csv"
    output_path = tmp_path / "features.pt"
    write_region_table(table_path)

    def fake_extract(_dicom_id: str, _image_path: Path | None) -> torch.Tensor:
        return torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    report = build_feature_cache_from_table(
        config=FeatureCacheConfig(
            region_table_csv=table_path,
            output_path=output_path,
            image_root=None,
            image_ext=".jpg",
            image_size=(224, 224),
            metadata_mode="none",
        ),
        extract_local_features=fake_extract,
    )

    payload = torch.load(output_path, weights_only=False)
    assert "metadata" not in payload
    assert report["metadata_mode"] == "none"


def test_partial_report_with_only_missing_bbox_rows_is_nonfatal() -> None:
    report = {
        "status": "partial",
        "rows": 10,
        "extraction_errors": [],
        "skipped_missing_bbox_rows": 2,
    }

    assert exit_code_from_report(report) == 0


def test_partial_report_with_extraction_errors_is_fatal() -> None:
    report = {
        "status": "partial",
        "rows": 10,
        "extraction_errors": [{"dicom_id": "x", "error": "missing image"}],
        "skipped_missing_bbox_rows": 0,
    }

    assert exit_code_from_report(report) == 2


def test_feature_extraction_progress_logger_receives_updates(tmp_path: Path) -> None:
    table_path = tmp_path / "region_table.csv"
    output_path = tmp_path / "features.pt"
    write_region_table(table_path)
    messages: list[str] = []

    def fake_extract(_dicom_id: str, _image_path: Path | None) -> torch.Tensor:
        return torch.arange(16, dtype=torch.float32).view(1, 4, 4)

    build_feature_cache_from_table(
        config=FeatureCacheConfig(
            region_table_csv=table_path,
            output_path=output_path,
            image_root=None,
            image_ext=".jpg",
            image_size=(224, 224),
            progress_every=1,
        ),
        extract_local_features=fake_extract,
        progress_logger=messages.append,
    )

    assert any("processed_dicoms=1/1" in message for message in messages)
