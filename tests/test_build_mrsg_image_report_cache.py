from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from tests.mrsg_test_utils import (
    write_descriptions,
    write_fake_mimic_csv,
    write_fake_mscxr_json,
)


def test_manifest_builder_writes_leakage_free_phrase_rows(tmp_path: Path) -> None:
    from anaprior.train.build_mrsg_image_report_cache import build_mrsg_image_report_cache

    report = build_mrsg_image_report_cache(
        mimic_csv=write_fake_mimic_csv(tmp_path),
        mscxr_json=write_fake_mscxr_json(tmp_path),
        descriptions_json=write_descriptions(tmp_path),
        outdir=tmp_path / "out",
        valid_fraction=0.5,
        seed=13,
    )

    assert report["uses_spatial_annotations"] is False
    assert report["uses_dcem"] is False
    assert report["sanity"]["mscxr_overlap"] == 0
    assert report["sanity"]["train_valid_subject_overlap"] == 0
    assert report["train_rows"] >= 1
    assert report["valid_rows"] >= 1

    train_jsonl = Path(report["train_jsonl"])
    valid_jsonl = Path(report["valid_jsonl"])
    protocol_manifest = Path(report["protocol_manifest"])
    assert train_jsonl.exists()
    assert valid_jsonl.exists()
    assert protocol_manifest.exists()

    train_rows = [json.loads(line) for line in train_jsonl.read_text(encoding="utf-8").splitlines()]
    valid_rows = [json.loads(line) for line in valid_jsonl.read_text(encoding="utf-8").splitlines()]
    all_rows = train_rows + valid_rows
    assert all_rows

    for row in all_rows:
        assert set(row) == {
            "image_path",
            "subject_id",
            "study_id",
            "dicom_id",
            "phrase",
            "finding",
            "disease_description",
            "negative_phrases",
        }
        payload = json.dumps(row).lower()
        for forbidden in ("box", "bbox", "mask", "region", "coordinates", "oracle", "dcem"):
            assert f'"{forbidden}"' not in payload
        assert isinstance(row["negative_phrases"], list)
        assert all(isinstance(item, str) and item.strip() for item in row["negative_phrases"])

    train_subjects = {row["subject_id"] for row in train_rows}
    valid_subjects = {row["subject_id"] for row in valid_rows}
    assert train_subjects.isdisjoint(valid_subjects)
    assert {row["dicom_id"] for row in all_rows} == {"train", "valid"}


def test_manifest_builder_rejects_recursive_forbidden_spatial_keys(tmp_path: Path) -> None:
    from anaprior.train.build_mrsg_image_report_cache import build_mrsg_image_report_cache

    mimic_csv = tmp_path / "mimic.csv"
    pd.DataFrame(
        [
            {
                "path": "files/p11/p11000001/s51000001/train.jpg",
                "report": "small right pneumothorax",
                "metadata": json.dumps({"nested": [{"oracle": [1, 2, 3]}]}),
            }
        ]
    ).to_csv(mimic_csv, index=False)

    with pytest.raises(ValueError, match="forbidden spatial"):
        build_mrsg_image_report_cache(
            mimic_csv=mimic_csv,
            mscxr_json=write_fake_mscxr_json(tmp_path),
            descriptions_json=write_descriptions(tmp_path),
            outdir=tmp_path / "out",
            valid_fraction=0.5,
            seed=13,
            passthrough_columns=("metadata",),
        )


def test_spatial_audit_treats_bracketed_report_text_as_plain_text() -> None:
    from anaprior.train.build_mrsg_image_report_cache import (
        _assert_no_forbidden_spatial_fields,
    )

    _assert_no_forbidden_spatial_fields(
        {
            "phrase": "[** deidentified date **] right basilar opacity",
            "negative_phrases": ["{not JSON} no pleural effusion"],
        }
    )


def test_manifest_builder_skips_negated_and_uncertain_phrases(tmp_path: Path) -> None:
    from anaprior.train.build_mrsg_image_report_cache import build_mrsg_image_report_cache

    mimic_csv = tmp_path / "mimic.csv"
    pd.DataFrame(
        [
            {
                "path": "files/p11/p11000001/s51000001/train.jpg",
                "report": "possible small right apical pneumothorax. no pleural effusion.",
            },
            {
                "path": "files/p12/p12000001/s52000001/valid.jpg",
                "report": "left basilar opacity.",
            },
        ]
    ).to_csv(mimic_csv, index=False)

    report = build_mrsg_image_report_cache(
        mimic_csv=mimic_csv,
        mscxr_json=write_fake_mscxr_json(tmp_path),
        descriptions_json=write_descriptions(tmp_path),
        outdir=tmp_path / "out",
        valid_fraction=0.5,
        seed=13,
    )

    train_rows = [
        json.loads(line)
        for line in Path(report["train_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    valid_rows = [
        json.loads(line)
        for line in Path(report["valid_jsonl"]).read_text(encoding="utf-8").splitlines()
    ]
    all_rows = train_rows + valid_rows

    assert len(all_rows) == 1
    assert all_rows[0]["phrase"] == "left basilar opacity."
    assert all_rows[0]["finding"] == "Lung Opacity"
    assert report["train_rows"] + report["valid_rows"] == 1


def test_manifest_builder_is_deterministic_and_audits_patient_overlap(tmp_path: Path) -> None:
    from anaprior.train.build_mrsg_image_report_cache import build_mrsg_image_report_cache

    mimic_csv = tmp_path / "mimic.csv"
    pd.DataFrame(
        [
            {
                "path": "files/p11/p11000001/s51000001/a.jpg",
                "report": "small right apical pneumothorax",
            },
            {
                "path": "files/p11/p11000001/s51000002/b.jpg",
                "report": "small right apical pneumothorax",
            },
            {
                "path": "files/p12/p12000001/s52000001/c.jpg",
                "report": "left basilar opacity",
            },
        ]
    ).to_csv(mimic_csv, index=False)

    first = build_mrsg_image_report_cache(
        mimic_csv=mimic_csv,
        mscxr_json=write_fake_mscxr_json(tmp_path),
        descriptions_json=write_descriptions(tmp_path),
        outdir=tmp_path / "out-a",
        valid_fraction=0.5,
        seed=7,
    )
    second = build_mrsg_image_report_cache(
        mimic_csv=mimic_csv,
        mscxr_json=write_fake_mscxr_json(tmp_path),
        descriptions_json=write_descriptions(tmp_path),
        outdir=tmp_path / "out-b",
        valid_fraction=0.5,
        seed=7,
    )

    assert Path(first["train_jsonl"]).read_text(encoding="utf-8") == Path(
        second["train_jsonl"]
    ).read_text(encoding="utf-8")
    assert Path(first["valid_jsonl"]).read_text(encoding="utf-8") == Path(
        second["valid_jsonl"]
    ).read_text(encoding="utf-8")
    assert first["sanity"]["train_valid_subject_overlap"] == 0
    assert first["protocol"]["sanity"]["train_valid_subject_overlap"] == 0


def test_builder_output_loads_with_dataset_image_root(tmp_path: Path) -> None:
    from anaprior.data.mrsg_dataset import MRSGDataset
    from anaprior.train.build_mrsg_image_report_cache import build_mrsg_image_report_cache

    image_root = tmp_path / "images"
    image_path = image_root / "files" / "p11" / "p11000001" / "s51000001" / "train.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(32, 64, 96)).save(image_path)

    mimic_csv = tmp_path / "mimic.csv"
    pd.DataFrame(
        [
            {
                "path": "files/p11/p11000001/s51000001/train.jpg",
                "report": "small right apical pneumothorax.",
            }
        ]
    ).to_csv(mimic_csv, index=False)

    report = build_mrsg_image_report_cache(
        mimic_csv=mimic_csv,
        mscxr_json=write_fake_mscxr_json(tmp_path),
        descriptions_json=write_descriptions(tmp_path),
        outdir=tmp_path / "out",
        valid_fraction=0.5,
        seed=13,
    )

    dataset = MRSGDataset(
        manifest_path=report["train_jsonl"] if report["train_rows"] else report["valid_jsonl"],
        image_root=image_root,
        image_size=(6, 6),
        crop_size=(4, 4),
        seed=7,
    )

    sample = dataset[0]

    assert sample["subject_id"] == "11000001"
    assert sample["study_id"] == "51000001"
    assert sample["dicom_id"] == "train"
    assert sample["original_image"].shape == (3, 6, 6)


def test_manifest_builder_cli_help_and_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from anaprior.train.build_mrsg_image_report_cache import main

    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_text = capsys.readouterr().out
    assert "Build MRSG image-report JSONL manifests" in help_text

    exit_code = main(
        [
            "--mimic-csv",
            str(write_fake_mimic_csv(tmp_path)),
            "--mscxr-json",
            str(write_fake_mscxr_json(tmp_path)),
            "--descriptions-json",
            str(write_descriptions(tmp_path)),
            "--outdir",
            str(tmp_path / "cli-out"),
            "--valid-fraction",
            "0.5",
            "--seed",
            "13",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert Path(payload["train_jsonl"]).exists()
    assert Path(payload["valid_jsonl"]).exists()
