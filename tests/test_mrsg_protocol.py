from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from anaprior.data.mrsg_protocol import (
    MIMICIdentity,
    ProtocolSplitResult,
    build_mscxr_exclusion_set,
    filter_and_split_mimic_rows,
    identity_from_path,
    write_protocol_manifest,
)


def test_identity_from_path_ignores_mimic_shard_and_handles_both_separators() -> None:
    expected = MIMICIdentity("10000001", "50000001", "abc-def")

    assert identity_from_path("files/p10/p10000001/s50000001/abc-def.jpg") == expected
    assert identity_from_path(r"files\p10\p10000001\s50000001\abc-def.jpg") == expected


def test_build_exclusion_set_accepts_coco_payload_and_explicit_ids() -> None:
    payload = {
        "images": [
            {"path": "files/p10/p10000001/s50000001/path-dicom.jpg"},
            {
                "subject_id": "p11000001",
                "study_id": "s51000001",
                "dicom_id": "EXPLICIT-DICOM",
                "path": r"files\p11\p11000001\s51000001\explicit-dicom.jpg",
            },
        ]
    }

    exclusions = build_mscxr_exclusion_set(payload)

    assert exclusions.subjects == frozenset({"10000001", "11000001"})
    assert exclusions.studies == frozenset({"50000001", "51000001"})
    assert exclusions.dicoms == frozenset({"path-dicom", "explicit-dicom"})
    assert "files/p11/p11000001/s51000001/explicit-dicom.jpg" in exclusions.paths


def test_protocol_excludes_any_matching_identity_or_normalized_path() -> None:
    rows = pd.DataFrame(
        [
            {"path": "files/p10/p10000001/s60000001/by-subject.jpg", "report": "a"},
            {"path": "files/p10/p10000002/s50000001/by-study.jpg", "report": "b"},
            {"path": "files/p10/p10000003/s60000003/excluded.jpg", "report": "c"},
            {"path": r"files\p10\p10000004\s60000004\by-path.jpg", "report": "d"},
            {"path": "files/p11/p11000001/s51000001/train.jpg", "report": "left opacity"},
            {"path": "files/p11/p11000005/s51000005/valid.jpg", "report": "cardiomegaly"},
        ]
    )

    result = filter_and_split_mimic_rows(
        rows,
        excluded_subjects={"p10000001"},
        excluded_studies={"s50000001"},
        excluded_dicoms={"EXCLUDED"},
        excluded_paths={"files/p10/p10000004/s60000004/by-path.jpg"},
        valid_fraction=0.5,
        seed=13,
    )

    assert set(result.all_rows["subject_id"]) == {"11000001", "11000005"}
    assert set(result.train_rows["subject_id"]) == {"11000001"}
    assert set(result.valid_rows["subject_id"]) == {"11000005"}
    assert set(result.train_rows["subject_id"]).isdisjoint(result.valid_rows["subject_id"])
    assert result.num_rows_before_exclusion == 6
    assert result.num_excluded_mscxr_rows == 4


def test_protocol_is_deterministic_and_keeps_patients_in_one_split() -> None:
    rows = pd.DataFrame(
        [
            {"path": "files/p11/p11000001/s51000001/a.jpg", "report": "opacity"},
            {"path": "files/p11/p11000001/s51000002/b.jpg", "report": "opacity"},
            {"path": "files/p11/p11000005/s51000005/c.jpg", "report": "edema"},
        ]
    )

    first = filter_and_split_mimic_rows(rows, valid_fraction=0.5, seed=13)
    second = filter_and_split_mimic_rows(rows.sample(frac=1.0), valid_fraction=0.5, seed=13)

    assert dict(zip(first.all_rows["path"], first.all_rows["split"])) == dict(
        zip(second.all_rows["path"], second.all_rows["split"])
    )
    assert first.all_rows.groupby("subject_id")["split"].nunique().max() == 1


@pytest.mark.parametrize(
    "rows, message",
    [
        (pd.DataFrame([{"path": "files/p11/p11000001/s51000001/a.jpg"}]), "report"),
        (
            pd.DataFrame(
                [{"path": "files/p11/p11000001/s51000001/a.jpg", "report": "  "}]
            ),
            "report",
        ),
        (
            pd.DataFrame(
                [
                    {
                        "path": "files/p11/p11000001/s51000001/a.jpg",
                        "report": "opacity",
                        "region_map": "forbidden",
                    }
                ]
            ),
            "prohibited",
        ),
    ],
)
def test_protocol_rejects_missing_reports_and_spatial_supervision(rows, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        filter_and_split_mimic_rows(rows)


def test_protocol_rejects_unparseable_identity_and_invalid_fraction() -> None:
    rows = pd.DataFrame([{"path": "not-a-mimic-path.jpg", "report": "opacity"}])

    with pytest.raises(ValueError, match="identity"):
        filter_and_split_mimic_rows(rows)
    with pytest.raises(ValueError, match="valid_fraction"):
        filter_and_split_mimic_rows(
            pd.DataFrame(
                [{"path": "files/p11/p11000001/s51000001/a.jpg", "report": "opacity"}]
            ),
            valid_fraction=1.0,
        )


def test_protocol_rejects_identity_columns_that_conflict_with_path() -> None:
    rows = pd.DataFrame(
        [
            {
                "path": "files/p11/p11000001/s51000001/a.jpg",
                "subject_id": "11000002",
                "report": "opacity",
            }
        ]
    )

    with pytest.raises(ValueError, match="conflicts with path"):
        filter_and_split_mimic_rows(rows)


def test_manifest_records_annotation_free_integrity_hashes(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {"path": "files/p10/p10000001/s50000001/excluded.jpg", "report": "pneumonia"},
            {"path": "files/p11/p11000001/s51000001/train.jpg", "report": "opacity"},
            {"path": "files/p11/p11000005/s51000005/valid.jpg", "report": "edema"},
        ]
    )
    exclusions = build_mscxr_exclusion_set(
        [{"path": "files/p10/p10000001/s50000001/excluded.jpg"}]
    )
    result = filter_and_split_mimic_rows(
        rows,
        excluded_subjects=exclusions.subjects,
        excluded_studies=exclusions.studies,
        excluded_dicoms=exclusions.dicoms,
        excluded_paths=exclusions.paths,
        valid_fraction=0.5,
        seed=13,
    )
    output = tmp_path / "protocol.json"

    manifest = write_protocol_manifest(output, result, exclusions)

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert manifest["uses_spatial_annotations"] is False
    assert manifest["uses_dcem"] is False
    assert manifest["num_rows_before_exclusion"] == 3
    assert manifest["num_rows_after_exclusion"] == 2
    assert manifest["num_excluded_mscxr_rows"] == 1
    for key in (
        "train_subjects_sha256",
        "valid_subjects_sha256",
        "exclusion_ids_sha256",
    ):
        assert len(manifest[key]) == 64
        int(manifest[key], 16)
        assert manifest[key] == manifest[key].lower()
    assert manifest["train_subjects_sha256"] == hashlib.sha256(b"11000001").hexdigest()
    assert manifest["sanity"] == {
        "train_valid_subject_overlap": 0,
        "mscxr_overlap": 0,
    }


def test_manifest_rejects_overlapping_subject_splits(tmp_path) -> None:
    shared = pd.DataFrame(
        [
            {
                "subject_id": "11000001",
                "study_id": "51000001",
                "dicom_id": "a",
                "normalized_path": "files/p11/p11000001/s51000001/a.jpg",
            }
        ]
    )
    result = ProtocolSplitResult(
        all_rows=shared,
        train_rows=shared,
        valid_rows=shared,
        num_rows_before_exclusion=1,
        num_excluded_mscxr_rows=0,
        valid_fraction=0.1,
        seed=13,
    )
    exclusions = build_mscxr_exclusion_set(
        [{"path": "files/p99/p99999999/s59999999/other.jpg"}]
    )

    with pytest.raises(ValueError, match="overlap"):
        write_protocol_manifest(tmp_path / "invalid.json", result, exclusions)


def test_fully_excluded_input_produces_auditable_empty_result(tmp_path) -> None:
    rows = pd.DataFrame(
        [{"path": "files/p10/p10000001/s50000001/a.jpg", "report": "opacity"}]
    )
    exclusions = build_mscxr_exclusion_set(
        [{"path": "files/p10/p10000001/s50000001/a.jpg"}]
    )

    result = filter_and_split_mimic_rows(
        rows,
        excluded_subjects=exclusions.subjects,
        excluded_studies=exclusions.studies,
        excluded_dicoms=exclusions.dicoms,
        excluded_paths=exclusions.paths,
    )
    manifest = write_protocol_manifest(tmp_path / "empty.json", result, exclusions)

    assert result.train_rows.empty
    assert result.valid_rows.empty
    assert manifest["num_rows_after_exclusion"] == 0
    assert manifest["sanity"]["mscxr_overlap"] == 0
