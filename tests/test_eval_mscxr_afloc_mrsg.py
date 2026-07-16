from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from anaprior.eval.eval_mscxr_afloc_mrsg import (
    build_mscxr_afloc_mrsg_hmaps,
    default_encode_case,
    load_dataset_rows,
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


def test_load_dataset_rows_passes_explicit_mscxr_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = tmp_path / "mscxr.json"
    image_root = tmp_path / "jpg"
    seen: dict[str, object] = {}

    def fake_load_data(dataset, **kwargs):
        seen["dataset"] = dataset
        seen.update(kwargs)
        return [
            {
                "path": str(image_root / "case.jpg"),
                "label_text": "opacity",
                "category": "Lung Opacity",
            }
        ]

    monkeypatch.setitem(
        sys.modules,
        "localization.datasets",
        SimpleNamespace(load_data=fake_load_data),
    )
    rows = load_dataset_rows(
        "MS_CXR",
        split="test",
        ms_cxr_json=annotation,
        mimic_img_dir=image_root,
    )

    assert len(rows) == 1
    assert seen == {
        "dataset": "MS_CXR",
        "ms_cxr_json": annotation,
        "mimic_img_dir": image_root,
    }

def test_default_encode_case_uses_afloc_process_img_contract(tmp_path: Path) -> None:
    ckpt = write_fake_mrsg_checkpoint(tmp_path)
    image_path = tmp_path / "process-img.jpg"
    image_path.write_bytes(b"not-an-image-needed")
    processed = torch.full((1, 3, 7, 5), 0.25, dtype=torch.float32)
    seen: dict[str, object] = {}

    class FakeRuntimeAFLoc:
        def process_img(self, paths, device, flag=0):
            seen["paths"] = list(paths)
            seen["device"] = device
            seen["flag"] = flag
            return processed.clone()

    class FakeRuntimeEncoder:
        def __init__(self) -> None:
            self.afloc = FakeRuntimeAFLoc()

        def __call__(self, images, phrases, disease_descriptions, device, image_gray=None):
            seen["images"] = images.detach().clone()
            seen["phrases"] = list(phrases)
            seen["descriptions"] = list(disease_descriptions)
            seen["encoder_device"] = device
            seen["image_gray"] = None if image_gray is None else image_gray.detach().clone()
            return (
                type(
                    "ImageFeatures",
                    (),
                    {
                        "image_gray": (
                            images.mean(dim=1, keepdim=True)
                            if image_gray is None
                            else image_gray
                        )
                    },
                )(),
                object(),
            )

    class FakeMRSGModel:
        def __call__(self, image_features, phrase_features):
            assert torch.equal(image_features.image_gray, processed.mean(dim=1, keepdim=True))
            heatmap = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]], dtype=torch.float32)
            return type(
                "Output",
                (),
                {
                    "final_heatmap": heatmap,
                    "query_route_weights": torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
                    "query_reliability": torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32),
                },
            )()

    runtime = {
        "afloc_encoder": FakeRuntimeEncoder(),
        "mrsg_model": FakeMRSGModel(),
    }
    checkpoint_bundle = {
        "path": ckpt,
        "reference": {"path": ckpt, "payload": {}},
        "runtime": runtime,
        "afloc_checkpoint": tmp_path / "afloc.ckpt",
    }

    result = build_mscxr_afloc_mrsg_hmaps(
        data_rows=[
            {
                "path": str(image_path),
                "label_text": "right pleural effusion",
                "category": "Pleural Effusion",
            }
        ],
        dataset="MS_CXR",
        checkpoint=ckpt,
        afloc_checkpoint=tmp_path / "afloc.ckpt",
        encode_case=lambda row, checkpoint, device: default_encode_case(row, checkpoint_bundle, device),
        device="cpu",
    )

    assert seen["paths"] == [str(image_path)]
    assert seen["flag"] == 0
    assert seen["device"] == "cpu"
    assert torch.equal(seen["images"], processed)
    assert seen["phrases"] == ["right pleural effusion"]
    assert seen["descriptions"] == ["Pleural Effusion"]
    assert seen["image_gray"] is None
    assert result.case_diagnostics[0]["hmap_max"] == pytest.approx(1.0)


def test_default_encode_case_fails_clearly_without_afloc_process_img(tmp_path: Path) -> None:
    ckpt = write_fake_mrsg_checkpoint(tmp_path)
    checkpoint_bundle = {
        "path": ckpt,
        "reference": {"path": ckpt, "payload": {}},
        "runtime": {
            "afloc_encoder": object(),
            "mrsg_model": object(),
        },
        "afloc_checkpoint": tmp_path / "afloc.ckpt",
    }

    with pytest.raises(RuntimeError, match="afloc_encoder\\.afloc\\.process_img"):
        default_encode_case(
            {
                "path": str(tmp_path / "missing-process-img.jpg"),
                "label_text": "opacity",
                "category": "Pneumonia",
            },
            checkpoint_bundle,
            "cpu",
        )


def test_default_encode_case_supplies_preprocessed_grayscale_override_when_cfg_is_available(
    tmp_path: Path,
) -> None:
    ckpt = write_fake_mrsg_checkpoint(tmp_path)
    image_path = tmp_path / "cfg-process-img.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8), mode="L").save(image_path)
    processed = torch.full((1, 3, 8, 8), 0.25, dtype=torch.float32)
    seen: dict[str, object] = {}

    class FakeRuntimeAFLoc:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(
                data=SimpleNamespace(image=SimpleNamespace(imsize=8)),
                transforms=SimpleNamespace(center_crop=None, random_crop=None, norm="half"),
            )

        def process_img(self, paths, device, flag=0):
            seen["paths"] = list(paths)
            seen["device"] = device
            seen["flag"] = flag
            return processed.clone()

    class FakeRuntimeEncoder:
        def __init__(self) -> None:
            self.afloc = FakeRuntimeAFLoc()

        def __call__(self, images, phrases, disease_descriptions, device, image_gray=None):
            seen["image_gray"] = None if image_gray is None else image_gray.detach().clone()
            return (
                type(
                    "ImageFeatures",
                    (),
                    {
                        "image_gray": (
                            images.mean(dim=1, keepdim=True)
                            if image_gray is None
                            else image_gray
                        )
                    },
                )(),
                object(),
            )

    class FakeMRSGModel:
        def __call__(self, image_features, phrase_features):
            assert image_features.image_gray.shape == (1, 1, 8, 8)
            assert float(image_features.image_gray.min()) >= 0.0
            assert float(image_features.image_gray.max()) <= 1.0
            heatmap = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
            return type(
                "Output",
                (),
                {
                    "final_heatmap": heatmap,
                    "query_route_weights": torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32),
                    "query_reliability": torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
                },
            )()

    checkpoint_bundle = {
        "path": ckpt,
        "reference": {"path": ckpt, "payload": {}},
        "runtime": {
            "afloc_encoder": FakeRuntimeEncoder(),
            "mrsg_model": FakeMRSGModel(),
        },
        "afloc_checkpoint": tmp_path / "afloc.ckpt",
    }

    default_encode_case(
        {
            "path": str(image_path),
            "label_text": "opacity",
            "category": "Pneumonia",
        },
        checkpoint_bundle,
        "cpu",
    )

    assert seen["paths"] == [str(image_path)]
    assert seen["flag"] == 0
    assert isinstance(seen["image_gray"], torch.Tensor)
