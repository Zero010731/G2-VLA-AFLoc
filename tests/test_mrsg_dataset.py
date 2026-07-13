from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from anaprior.models.afloc_mrsg.teacher import GeometryTransform


def _write_manifest_and_image(tmp_path: Path, *, image_name: str = "scan.png") -> Path:
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
                "image_path": str(image_path),
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
    from anaprior.data.mrsg_dataset import MRSGDataset

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

    assert isinstance(sample["geometry"], GeometryTransform)
    assert isinstance(sample["equivariance_transform"], GeometryTransform)
    assert sample["geometry"] == GeometryTransform(
        horizontal_flip=True,
        crop_top=0,
        crop_left=2,
        crop_height=4,
        crop_width=4,
    )
    assert sample["equivariance_transform"] == sample["geometry"]
    assert torch.equal(sample["weak_image"], sample["strong_image"])
    assert sample["phrase"] == "small left apical pneumothorax"
    assert sample["negative_phrases"] == ["small right pleural effusion"]
    assert sample["equivariance_phrase"] == "small left apical pneumothorax"
    assert sample["equivariance_negative_phrases"] == ["small right pleural effusion"]


def test_dataset_uses_shared_geometry_for_weak_and_strong_but_distinct_photometric_noise(
    tmp_path: Path,
) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset

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

    assert sample["geometry"].horizontal_flip is True
    assert sample["equivariance_transform"].horizontal_flip is False
    assert sample["phrase"] == "small left apical pneumothorax"
    assert sample["equivariance_phrase"] == "small right apical pneumothorax"
    assert not torch.equal(sample["weak_image"], sample["strong_image"])
    assert torch.equal(sample["weak_geometry_image"], sample["geometry_applied_image"])
    assert torch.equal(sample["strong_geometry_image"], sample["geometry_applied_image"])


def test_dataset_is_deterministic_for_the_same_seed_and_varies_across_seeds(tmp_path: Path) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset

    manifest = _write_manifest_and_image(tmp_path)
    first = MRSGDataset(manifest_path=manifest, image_size=(6, 6), crop_size=(4, 4), seed=11)
    second = MRSGDataset(manifest_path=manifest, image_size=(6, 6), crop_size=(4, 4), seed=11)
    third = MRSGDataset(manifest_path=manifest, image_size=(6, 6), crop_size=(4, 4), seed=12)

    first_sample = first[0]
    second_sample = second[0]
    third_sample = third[0]

    assert first_sample["geometry"] == second_sample["geometry"]
    assert first_sample["equivariance_transform"] == second_sample["equivariance_transform"]
    assert torch.equal(first_sample["weak_image"], second_sample["weak_image"])
    assert torch.equal(first_sample["strong_image"], second_sample["strong_image"])
    assert (
        third_sample["geometry"] != first_sample["geometry"]
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
