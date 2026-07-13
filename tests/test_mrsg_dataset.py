from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from anaprior.models.afloc_mrsg.teacher import GeometryTransform


def _write_manifest_and_image(
    tmp_path: Path,
    *,
    image_name: str = "scan.png",
    manifest_image_path: str | None = None,
) -> Path:
    image_path = tmp_path / image_name
    width = height = 8
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            rgb[row, col, 0] = row * 16 + col
            rgb[row, col, 1] = col * 16
            rgb[row, col, 2] = 255 - row * 16
    Image.fromarray(rgb, mode="RGB").save(image_path)

    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "image_path": str(image_path) if manifest_image_path is None else manifest_image_path,
                "subject_id": "11000001",
                "study_id": "51000001",
                "dicom_id": "dicom-1",
                "phrase": "small right apical pneumothorax",
                "finding": "Pneumothorax",
                "disease_description": "air in the pleural space",
                "negative_phrases": ["small left pleural effusion"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_dataset_returns_geometry_tracked_views_and_laterality_metadata(tmp_path: Path) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset, geometry_transform_from_metadata

    dataset = MRSGDataset(
        manifest_path=_write_manifest_and_image(tmp_path),
        image_size=(6, 6),
        crop_size=(4, 4),
        seed=5,
        geometry_prob=1.0,
        horizontal_flip_prob=1.0,
        weak_noise_std=0.0,
        strong_noise_std=0.0,
        equivariance_horizontal_flip_prob=1.0,
    )

    sample = dataset[0]

    assert sample["original_image"].shape == (3, 6, 6)
    assert sample["weak_image"].shape == (3, 4, 4)
    assert sample["strong_image"].shape == (3, 4, 4)
    assert sample["equivariance_image"].shape == (3, 4, 4)
    for key in ("original_image", "weak_image", "strong_image", "equivariance_image"):
        assert sample[key].dtype == torch.float32
        assert torch.all(sample[key] >= 0.0)
        assert torch.all(sample[key] <= 1.0)

    assert set(sample["geometry"]) == {
        "horizontal_flip",
        "crop_top",
        "crop_left",
        "crop_height",
        "crop_width",
    }
    assert set(sample["equivariance_transform"]) == set(sample["geometry"])
    assert geometry_transform_from_metadata(sample["geometry"]) == GeometryTransform(
        horizontal_flip=True,
        crop_top=0,
        crop_left=2,
        crop_height=4,
        crop_width=4,
    )
    assert geometry_transform_from_metadata(sample["equivariance_transform"]) == (
        geometry_transform_from_metadata(sample["geometry"])
    )
    assert torch.equal(sample["weak_image"], sample["strong_image"])
    assert sample["phrase"] == "small left apical pneumothorax"
    assert sample["negative_phrases"] == ["small right pleural effusion"]
    assert sample["equivariance_phrase"] == "small left apical pneumothorax"
    assert sample["equivariance_negative_phrases"] == ["small right pleural effusion"]


def test_dataset_uses_shared_geometry_for_weak_and_strong_but_distinct_photometric_noise(
    tmp_path: Path,
) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset, geometry_transform_from_metadata

    dataset = MRSGDataset(
        manifest_path=_write_manifest_and_image(tmp_path),
        image_size=(6, 6),
        crop_size=(4, 4),
        seed=5,
        geometry_prob=1.0,
        horizontal_flip_prob=1.0,
        weak_noise_std=0.0,
        strong_noise_std=0.15,
        equivariance_horizontal_flip_prob=0.0,
    )

    sample = dataset[0]

    assert geometry_transform_from_metadata(sample["geometry"]).horizontal_flip is True
    assert geometry_transform_from_metadata(sample["equivariance_transform"]).horizontal_flip is False
    assert sample["phrase"] == "small left apical pneumothorax"
    assert sample["equivariance_phrase"] == "small right apical pneumothorax"
    assert not torch.equal(sample["weak_image"], sample["strong_image"])
    assert torch.equal(sample["weak_geometry_image"], sample["geometry_applied_image"])
    assert torch.equal(sample["strong_geometry_image"], sample["geometry_applied_image"])


def test_dataset_is_deterministic_for_the_same_seed_and_varies_across_seeds(tmp_path: Path) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset, geometry_transform_from_metadata

    manifest = _write_manifest_and_image(tmp_path)
    first = MRSGDataset(manifest_path=manifest, image_size=(6, 6), crop_size=(4, 4), seed=11)
    second = MRSGDataset(manifest_path=manifest, image_size=(6, 6), crop_size=(4, 4), seed=11)
    third = MRSGDataset(manifest_path=manifest, image_size=(6, 6), crop_size=(4, 4), seed=12)

    first_sample = first[0]
    second_sample = second[0]
    third_sample = third[0]

    assert geometry_transform_from_metadata(first_sample["geometry"]) == (
        geometry_transform_from_metadata(second_sample["geometry"])
    )
    assert geometry_transform_from_metadata(first_sample["equivariance_transform"]) == (
        geometry_transform_from_metadata(second_sample["equivariance_transform"])
    )
    assert torch.equal(first_sample["weak_image"], second_sample["weak_image"])
    assert torch.equal(first_sample["strong_image"], second_sample["strong_image"])
    assert (
        geometry_transform_from_metadata(third_sample["geometry"])
        != geometry_transform_from_metadata(first_sample["geometry"])
        or not torch.equal(third_sample["weak_image"], first_sample["weak_image"])
    )


def test_dataset_fails_clearly_for_missing_and_corrupt_images(tmp_path: Path) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset

    missing_manifest = tmp_path / "missing.jsonl"
    missing_manifest.write_text(
        json.dumps(
            {
                "image_path": str(tmp_path / "missing.png"),
                "subject_id": "11000001",
                "study_id": "51000001",
                "dicom_id": "dicom-1",
                "phrase": "small right apical pneumothorax",
                "finding": "Pneumothorax",
                "disease_description": "air in the pleural space",
                "negative_phrases": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing image"):
        MRSGDataset(manifest_path=missing_manifest)[0]

    corrupt_image = tmp_path / "corrupt.png"
    corrupt_image.write_bytes(b"not-an-image")
    corrupt_manifest = tmp_path / "corrupt.jsonl"
    corrupt_manifest.write_text(
        json.dumps(
            {
                "image_path": str(corrupt_image),
                "subject_id": "11000001",
                "study_id": "51000001",
                "dicom_id": "dicom-2",
                "phrase": "small right apical pneumothorax",
                "finding": "Pneumothorax",
                "disease_description": "air in the pleural space",
                "negative_phrases": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Failed to load image"):
        MRSGDataset(manifest_path=corrupt_manifest)[0]


def test_dataset_resolves_relative_paths_under_explicit_image_root_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset

    image_root = tmp_path / "images"
    relative_image = image_root / "nested" / "scan.png"
    relative_image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(relative_image)

    relative_manifest = _write_manifest_and_image(
        tmp_path,
        image_name="unused-relative.png",
        manifest_image_path="nested/scan.png",
    )
    relative_sample = MRSGDataset(
        manifest_path=relative_manifest,
        image_root=image_root,
        image_size=(6, 6),
    )[0]
    assert relative_sample["original_image"].shape == (3, 6, 6)

    absolute_manifest = _write_manifest_and_image(tmp_path, image_name="absolute.png")
    absolute_sample = MRSGDataset(
        manifest_path=absolute_manifest,
        image_root=image_root,
        image_size=(6, 6),
    )[0]
    assert absolute_sample["original_image"].shape == (3, 6, 6)

    traversal_manifest = _write_manifest_and_image(
        tmp_path,
        image_name="unused-traversal.png",
        manifest_image_path="../escape.png",
    )
    with pytest.raises(ValueError, match="outside image_root"):
        MRSGDataset(
            manifest_path=traversal_manifest,
            image_root=image_root,
            image_size=(6, 6),
        )[0]


def test_default_dataloader_collates_geometry_metadata_and_reconstructs_transform(
    tmp_path: Path,
) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset, geometry_transform_from_metadata

    manifest = tmp_path / "train.jsonl"
    rows = []
    for index in range(2):
        image_path = tmp_path / f"scan-{index}.png"
        Image.new("RGB", (8, 8), color=(32 + index, 64, 96)).save(image_path)
        rows.append(
            {
                "image_path": str(image_path),
                "subject_id": f"1100000{index + 1}",
                "study_id": f"5100000{index + 1}",
                "dicom_id": f"dicom-{index + 1}",
                "phrase": "small right apical pneumothorax",
                "finding": "Pneumothorax",
                "disease_description": "air in the pleural space",
                "negative_phrases": ["small left pleural effusion"],
            }
        )
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = MRSGDataset(
        manifest_path=manifest,
        image_size=(6, 6),
        crop_size=(4, 4),
        seed=5,
        geometry_prob=1.0,
        horizontal_flip_prob=1.0,
        weak_noise_std=0.0,
        strong_noise_std=0.0,
        equivariance_horizontal_flip_prob=1.0,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["original_image"].shape == (2, 3, 6, 6)
    assert batch["weak_image"].shape == (2, 3, 4, 4)
    assert set(batch["geometry"]) == {
        "horizontal_flip",
        "crop_top",
        "crop_left",
        "crop_height",
        "crop_width",
    }
    assert all(torch.is_tensor(value) for value in batch["geometry"].values())
    assert all(torch.is_tensor(value) for value in batch["equivariance_transform"].values())

    first_geometry = geometry_transform_from_metadata(
        {key: value[0] for key, value in batch["geometry"].items()}
    )
    assert first_geometry == GeometryTransform(
        horizontal_flip=True,
        crop_top=0,
        crop_left=2,
        crop_height=4,
        crop_width=4,
    )


def test_dataset_epoch_control_is_deterministic_per_epoch_and_changes_augmentations(
    tmp_path: Path,
) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset

    manifest = _write_manifest_and_image(tmp_path)
    first = MRSGDataset(
        manifest_path=manifest,
        image_size=(6, 6),
        crop_size=(4, 4),
        seed=11,
        geometry_prob=1.0,
        horizontal_flip_prob=0.5,
        weak_noise_std=0.0,
        strong_noise_std=0.15,
        equivariance_horizontal_flip_prob=0.5,
    )
    second = MRSGDataset(
        manifest_path=manifest,
        image_size=(6, 6),
        crop_size=(4, 4),
        seed=11,
        geometry_prob=1.0,
        horizontal_flip_prob=0.5,
        weak_noise_std=0.0,
        strong_noise_std=0.15,
        equivariance_horizontal_flip_prob=0.5,
    )

    first.set_epoch(3)
    second.set_epoch(3)
    sample_a = first[0]
    sample_b = first[0]
    sample_c = second[0]

    assert sample_a["subject_id"] == sample_c["subject_id"]
    assert sample_a["study_id"] == sample_c["study_id"]
    assert sample_a["dicom_id"] == sample_c["dicom_id"]
    assert torch.equal(sample_a["strong_image"], sample_b["strong_image"])
    assert torch.equal(sample_a["strong_image"], sample_c["strong_image"])
    assert torch.equal(sample_a["equivariance_image"], sample_c["equivariance_image"])
    assert sample_a["equivariance_phrase"] == sample_c["equivariance_phrase"]

    first.set_epoch(4)
    shifted = first[0]

    assert shifted["subject_id"] == sample_a["subject_id"]
    assert shifted["study_id"] == sample_a["study_id"]
    assert shifted["dicom_id"] == sample_a["dicom_id"]
    assert (
        not torch.equal(shifted["strong_image"], sample_a["strong_image"])
        or not torch.equal(shifted["equivariance_image"], sample_a["equivariance_image"])
        or shifted["equivariance_phrase"] != sample_a["equivariance_phrase"]
    )
