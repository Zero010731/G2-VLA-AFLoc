from __future__ import annotations

from pathlib import Path

import numpy as np

from anaprior.eval.eval_mscxr_afloc_official_anchor import (
    build_official_anchor_hmaps,
    save_official_anchor_outputs,
)


def test_official_export_preserves_raw_similarity_and_excludes_gt(tmp_path: Path) -> None:
    seen = []

    def fake_encode(row, runtime, device):
        seen.append(dict(row))
        return np.array([[-2.0, 0.5], [1.0, 3.0]], dtype=np.float32)

    result = build_official_anchor_hmaps(
        data_rows=[{
            "path": str(tmp_path / "case.jpg"),
            "label_text": "right basilar opacity",
            "category": "Lung Opacity",
            "gtmasks": np.ones((224, 224), dtype=np.float32),
            "boxes": [[1, 2, 3, 4]],
        }],
        dataset="MS_CXR",
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=fake_encode,
    )

    assert "gtmasks" not in seen[0]
    assert "boxes" not in seen[0]
    payload = next(iter(result.hmaps.values()))
    np.testing.assert_array_equal(
        payload["hmap"],
        np.array([[-2.0, 0.5], [1.0, 3.0]], dtype=np.float32),
    )
    assert payload["hmap_key"] == str(tmp_path / "case.jpg") + "right basilar opacity"
    assert result.summary["method_name"] == "afloc_official_anchor"
    assert result.summary["uses_spatial_annotations"] is False


def test_official_export_saves_standard_method_directory(tmp_path: Path) -> None:
    result = build_official_anchor_hmaps(
        data_rows=[{
            "path": str(tmp_path / "case.jpg"),
            "label_text": "edema",
            "category": "Edema",
        }],
        dataset="MS_CXR",
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=lambda row, runtime, device: np.ones((4, 4), dtype=np.float32),
    )

    outputs = save_official_anchor_outputs(result, outdir=tmp_path / "out")

    assert Path(outputs["hmaps"]).parent.name == "afloc_official_anchor"
    assert Path(outputs["summary"]).is_file()
