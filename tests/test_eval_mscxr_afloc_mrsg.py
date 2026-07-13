from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from anaprior.eval.eval_mscxr_afloc_mrsg import (
    build_mscxr_afloc_mrsg_hmaps,
    main,
)
from tests.mrsg_test_utils import test_config as make_test_config


def write_fake_mrsg_checkpoint(tmp_path: Path, **overrides) -> Path:
    payload = {
        "model_config": asdict(make_test_config()),
        "model_state_dict": {},
        "image_channels": [32, 64, 128],
    }
    payload.update(overrides)
    path = tmp_path / "mrsg.pt"
    torch.save(payload, path)
    return path


def test_mscxr_mrsg_eval_uses_phrase_and_image_only(tmp_path: Path) -> None:
    ckpt = write_fake_mrsg_checkpoint(tmp_path)
    seen_rows: list[dict[str, object]] = []

    def fake_encode_case(row, checkpoint, device):
        seen_rows.append(dict(row))
        assert checkpoint["path"] == ckpt
        assert device == "cpu"
        return {
            "hmap": np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
            "query_route_weights": [0.4, 0.2, 0.2, 0.2],
        }

    result = build_mscxr_afloc_mrsg_hmaps(
        data_rows=[
            {
                "path": str(tmp_path / "case-pna.jpg"),
                "label_text": "right basilar pneumonia",
                "category": "Pneumonia",
                "gtmasks": np.ones((2, 2), dtype=np.uint8),
                "boxes": [[0, 0, 1, 1]],
            }
        ],
        dataset="MS_CXR",
        checkpoint=ckpt,
        encode_case=fake_encode_case,
        device="cpu",
        method_name="afloc_mrsg",
    )

    assert len(seen_rows) == 1
    assert set(seen_rows[0]) == {"case_id", "category", "dataset", "duplicate_index", "hmap_key", "label_text", "path"}
    assert set(result.hmaps) == {seen_rows[0]["case_id"]}
    payload = result.hmaps[str(seen_rows[0]["case_id"])]
    assert payload["hmap"].shape == (2, 2)
    assert float(payload["hmap"].min()) == pytest.approx(0.0)
    assert float(payload["hmap"].max()) == pytest.approx(1.0)
    assert payload["path"] == str(tmp_path / "case-pna.jpg")
    assert payload["label_text"] == "right basilar pneumonia"
    assert result.summary["uses_dcem"] is False
    assert result.summary["uses_region_predictor"] is False
    assert result.summary["external_evaluation"] is False


def test_duplicate_rows_get_stable_unique_case_ids(tmp_path: Path) -> None:
    ckpt = write_fake_mrsg_checkpoint(tmp_path)

    def fake_encode_case(row, checkpoint, device):
        return {"hmap": np.full((2, 2), float(row["duplicate_index"]), dtype=np.float32)}

    rows = [
        {
            "path": str(tmp_path / "dup.jpg"),
            "label_text": "right pleural effusion",
            "category": "Pleural Effusion",
        },
        {
            "path": str(tmp_path / "dup.jpg"),
            "label_text": "right pleural effusion",
            "category": "Pleural Effusion",
        },
    ]

    first = build_mscxr_afloc_mrsg_hmaps(
        data_rows=rows,
        dataset="MS_CXR",
        checkpoint=ckpt,
        encode_case=fake_encode_case,
        device="cpu",
    )
    second = build_mscxr_afloc_mrsg_hmaps(
        data_rows=rows,
        dataset="MS_CXR",
        checkpoint=ckpt,
        encode_case=fake_encode_case,
        device="cpu",
    )

    first_ids = [item["case_id"] for item in first.case_diagnostics]
    second_ids = [item["case_id"] for item in second.case_diagnostics]
    assert first_ids == second_ids
    assert len(set(first_ids)) == 2
    assert first.summary["num_duplicate_hmap_keys"] == 1
    assert first.case_diagnostics[0]["hmap_key"] == first.case_diagnostics[1]["hmap_key"]


def test_build_mscxr_afloc_mrsg_hmaps_validates_checkpoint_fields(tmp_path: Path) -> None:
    bad = write_fake_mrsg_checkpoint(tmp_path, model_config=None)

    with pytest.raises(ValueError, match="model_config"):
        build_mscxr_afloc_mrsg_hmaps(
            data_rows=[
                {
                    "path": str(tmp_path / "case.jpg"),
                    "label_text": "opacity",
                    "category": "Pneumonia",
                }
            ],
            dataset="MS_CXR",
            checkpoint=bad,
            encode_case=lambda row, checkpoint, device: {"hmap": np.zeros((2, 2), dtype=np.float32)},
            device="cpu",
        )


def test_chexlocalize_summary_marks_external_evaluation(tmp_path: Path) -> None:
    ckpt = write_fake_mrsg_checkpoint(tmp_path)

    result = build_mscxr_afloc_mrsg_hmaps(
        data_rows=[
            {
                "path": str(tmp_path / "chex.jpg"),
                "label_text": "Findings suggesting Pleural Effusion.",
                "category": "Pleural Effusion",
            }
        ],
        dataset="CHEXLOCALIZE",
        checkpoint=ckpt,
        encode_case=lambda row, checkpoint, device: {"hmap": np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)},
        device="cpu",
    )

    assert result.summary["external_evaluation"] is True
    assert result.summary["split"] == "test"
    assert result.hmaps[result.case_diagnostics[0]["case_id"]]["hmap"].shape == (2, 2)


def test_cli_rejects_prepared_and_gate_arguments_before_loading_models(tmp_path: Path) -> None:
    ckpt = write_fake_mrsg_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="prepared-inputs-npz"):
        main(
            [
                "--dataset",
                "MS_CXR",
                "--afloc-checkpoint",
                str(tmp_path / "afloc.ckpt"),
                "--checkpoint",
                str(ckpt),
                "--outdir",
                str(tmp_path / "out"),
                "--prepared-inputs-npz",
                str(tmp_path / "legacy.npz"),
            ]
        )

    with pytest.raises(ValueError, match="validation-gate"):
        main(
            [
                "--dataset",
                "MS_CXR",
                "--afloc-checkpoint",
                str(tmp_path / "afloc.ckpt"),
                "--checkpoint",
                str(ckpt),
                "--outdir",
                str(tmp_path / "out"),
                "--validation-gate",
            ]
        )
