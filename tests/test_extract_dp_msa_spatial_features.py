from pathlib import Path

import numpy as np
import torch

from anaprior.features.extract_dp_msa_spatial_features import (
    build_dp_msa_spatial_feature_cache,
    image_path_from_case_id,
)


def _prepared_inputs(path: Path) -> None:
    items = [
        {
            "case_id": "/tmp/a.jpgFindings suggesting Pneumonia.",
            "dicom_id": "a",
            "category": "Pneumonia",
            "finding": "Pneumonia",
            "phrase": "Findings suggesting Pneumonia.",
            "heatmap": np.zeros((2, 2), dtype=np.float32),
            "region_maps": np.zeros((2, 2, 2), dtype=np.float32),
            "regions": ["left", "right"],
        },
        {
            "case_id": "/tmp/b.jpgFindings suggesting Edema.",
            "dicom_id": "b",
            "category": "Edema",
            "finding": "Edema",
            "phrase": "Findings suggesting Edema.",
            "heatmap": np.zeros((2, 2), dtype=np.float32),
            "region_maps": np.zeros((2, 2, 2), dtype=np.float32),
            "regions": ["left", "right"],
        },
    ]
    np.savez(path, items=np.array(items, dtype=object))


def test_image_path_from_case_id_extracts_image_prefix() -> None:
    assert image_path_from_case_id("/mnt/x/a.jpgFindings suggesting Pneumonia.") == Path("/mnt/x/a.jpg")
    assert image_path_from_case_id("/mnt/x/a.pngfoo") == Path("/mnt/x/a.png")


def test_build_dp_msa_spatial_feature_cache_writes_case_aligned_tensor(tmp_path: Path) -> None:
    prepared = tmp_path / "inputs.npz"
    output = tmp_path / "spatial.pt"
    _prepared_inputs(prepared)

    def fake_extract(dicom_id: str, image_path: Path | None) -> torch.Tensor:
        base = 1.0 if dicom_id == "a" else 2.0
        assert image_path is not None
        return torch.full((3, 4, 4), base)

    report = build_dp_msa_spatial_feature_cache(prepared, output, fake_extract, feature_dtype="float32")

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert report["status"] == "ok"
    assert payload["case_ids"] == ["/tmp/a.jpgFindings suggesting Pneumonia.", "/tmp/b.jpgFindings suggesting Edema."]
    assert payload["spatial_features"].shape == (2, 3, 4, 4)
    assert torch.allclose(payload["spatial_features"][0], torch.ones(3, 4, 4))
