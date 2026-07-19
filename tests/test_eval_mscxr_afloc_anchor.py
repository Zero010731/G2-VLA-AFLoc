from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anaprior.eval.eval_mscxr_afloc_anchor import (
    build_mscxr_afloc_anchor_hmaps,
    save_mscxr_afloc_anchor_outputs,
)


def test_anchor_eval_uses_only_image_phrase_and_category(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []

    def fake_encode(row, runtime, device):
        seen.append(dict(row))
        assert runtime["afloc_checkpoint"] == tmp_path / "afloc.ckpt"
        assert device == "cpu"
        return {
            "hmap": np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32),
            "confidence": np.ones((2, 2), dtype=np.float32),
            "scale_agreement": 1.0,
        }

    result = build_mscxr_afloc_anchor_hmaps(
        data_rows=[
            {
                "path": str(tmp_path / "case.jpg"),
                "label_text": "right basilar opacity",
                "category": "Lung Opacity",
                "gtmasks": np.ones((4, 4), dtype=np.float32),
                "boxes": [[0, 0, 2, 2]],
            }
        ],
        dataset="MS_CXR",
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=fake_encode,
        device="cpu",
    )

    assert set(seen[0]) == {
        "case_id",
        "category",
        "dataset",
        "duplicate_index",
        "hmap_key",
        "label_text",
        "path",
    }
    payload = next(iter(result.hmaps.values()))
    assert payload["hmap"].shape == (224, 224)
    assert float(payload["hmap"].min()) == pytest.approx(0.0)
    assert float(payload["hmap"].max()) == pytest.approx(1.0)
    assert result.summary["method_name"] == "afloc_anchor"
    assert result.summary["uses_spatial_annotations"] is False
    assert result.summary["uses_dcem"] is False
    assert result.summary["uses_mrsg_checkpoint"] is False


def test_anchor_eval_assigns_stable_unique_duplicate_ids(tmp_path: Path) -> None:
    rows = [
        {
            "path": str(tmp_path / "dup.jpg"),
            "label_text": "pleural effusion",
            "category": "Pleural Effusion",
        },
        {
            "path": str(tmp_path / "dup.jpg"),
            "label_text": "pleural effusion",
            "category": "Pleural Effusion",
        },
    ]

    def fake_encode(row, runtime, device):
        value = float(row["duplicate_index"] + 1)
        return {
            "hmap": np.array([[0.0, value], [0.5, 0.25]], dtype=np.float32),
            "confidence": np.full((2, 2), 0.5, dtype=np.float32),
        }

    first = build_mscxr_afloc_anchor_hmaps(
        data_rows=rows,
        dataset="MS_CXR",
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=fake_encode,
    )
    second = build_mscxr_afloc_anchor_hmaps(
        data_rows=rows,
        dataset="MS_CXR",
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=fake_encode,
    )

    assert list(first.hmaps) == list(second.hmaps)
    assert len(first.hmaps) == 2
    assert first.summary["num_duplicate_hmap_keys"] == 1


def test_anchor_eval_counts_zero_variance_and_saves_standard_outputs(tmp_path: Path) -> None:
    result = build_mscxr_afloc_anchor_hmaps(
        data_rows=[
            {
                "path": str(tmp_path / "constant.jpg"),
                "label_text": "edema",
                "category": "Edema",
            }
        ],
        dataset="MS_CXR",
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=lambda row, runtime, device: {
            "hmap": np.ones((2, 2), dtype=np.float32),
            "confidence": np.full((2, 2), 0.4, dtype=np.float32),
        },
    )
    outputs = save_mscxr_afloc_anchor_outputs(result, outdir=tmp_path / "out")

    assert result.summary["zero_variance_count"] == 1
    assert Path(outputs["hmaps"]).name == "hmaps.npy"
    assert Path(outputs["hmaps"]).parent.name == "afloc_anchor"
    assert Path(outputs["case_diagnostics"]).exists()
    assert Path(outputs["summary"]).exists()
