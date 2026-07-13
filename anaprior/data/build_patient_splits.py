"""Build patient-disjoint Chest ImaGenome splits for AnaPrior-Loc.

This is the first Stage B guardrail: Chest ImaGenome training/validation rows
used for the Region Abnormality Predictor must exclude every MS-CXR patient.
The script reads the MS-CXR COCO-style JSON, extracts patient IDs from image
paths, filters Chest ImaGenome silver split CSVs, and writes a leakage report.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PATIENT_RE = re.compile(r"(?:^|[\\/])p(?P<patient>\d{5,})(?:[\\/]|$)")
DEFAULT_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create MS-CXR-excluded Chest ImaGenome patient splits."
    )
    parser.add_argument(
        "--mscxr-json",
        required=True,
        type=Path,
        help="Path to MS_CXR_Local_Alignment_v1.1.0_radgraph_phrase.json.",
    )
    parser.add_argument(
        "--chest-imagenome-root",
        required=True,
        type=Path,
        help="Root directory of chest-imagenome_1.0.0.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Directory for mscxr_patients.txt, clean CSVs, and leakage_report.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260704,
        help="Seed used only when --repartition-train-valid is set.",
    )
    parser.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.1,
        help="Held-out patient fraction used only with --repartition-train-valid.",
    )
    parser.add_argument(
        "--repartition-train-valid",
        action="store_true",
        help=(
            "Ignore Chest ImaGenome's original train/valid split after filtering "
            "and create a new patient-level train/valid split from both files."
        ),
    )
    parser.add_argument(
        "--fail-on-source-overlap",
        action="store_true",
        help="Exit with an error if raw Chest ImaGenome rows include MS-CXR patients.",
    )
    return parser.parse_args()


def normalize_patient_id(value: object) -> str:
    text = str(value).strip()
    if text.startswith("p") and text[1:].isdigit():
        return text[1:]
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def patient_from_path(path_value: object) -> str | None:
    if not path_value:
        return None
    match = PATIENT_RE.search(str(path_value))
    if not match:
        return None
    return normalize_patient_id(match.group("patient"))


def extract_mscxr_patients(mscxr_json: Path) -> set[str]:
    with mscxr_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    patients: set[str] = set()
    missing = 0
    for image in payload.get("images", []):
        patient = patient_from_path(image.get("path"))
        if patient is None:
            patient = patient_from_path(image.get("previous_path"))
        if patient is None:
            missing += 1
            continue
        patients.add(patient)

    if not patients:
        raise ValueError(f"No MS-CXR patients could be extracted from {mscxr_json}")
    if missing:
        print(f"warning: {missing} MS-CXR images had no parseable patient path")
    return patients


def split_csv_path(root: Path, split: str) -> Path:
    return root / "silver_dataset" / "splits" / f"{split}.csv"


def read_split_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    if "subject_id" not in fieldnames:
        raise ValueError(f"CSV missing subject_id column: {path}")
    return fieldnames, rows


def write_split_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def patient_set(rows: Iterable[dict[str, str]]) -> set[str]:
    return {normalize_patient_id(row["subject_id"]) for row in rows}


def filter_rows(rows: list[dict[str, str]], excluded_patients: set[str]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if normalize_patient_id(row["subject_id"]) not in excluded_patients
    ]


def repartition_train_valid(
    train_rows: list[dict[str, str]],
    valid_rows: list[dict[str, str]],
    heldout_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("--heldout-fraction must be between 0 and 1")

    rows_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in [*train_rows, *valid_rows]:
        rows_by_patient[normalize_patient_id(row["subject_id"])].append(row)

    patients = sorted(rows_by_patient)
    rng = random.Random(seed)
    rng.shuffle(patients)
    heldout_count = max(1, int(round(len(patients) * heldout_fraction)))
    heldout_patients = set(patients[:heldout_count])

    new_train: list[dict[str, str]] = []
    new_valid: list[dict[str, str]] = []
    for patient in patients:
        if patient in heldout_patients:
            new_valid.extend(rows_by_patient[patient])
        else:
            new_train.extend(rows_by_patient[patient])
    return new_train, new_valid


def summarize_split(rows: list[dict[str, str]], mscxr_patients: set[str]) -> dict[str, int]:
    patients = patient_set(rows)
    return {
        "rows": len(rows),
        "patients": len(patients),
        "mscxr_patient_overlap": len(patients & mscxr_patients),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    mscxr_patients = extract_mscxr_patients(args.mscxr_json)

    (args.outdir / "mscxr_patients.txt").write_text(
        "\n".join(sorted(mscxr_patients)) + "\n",
        encoding="utf-8",
    )

    fieldnames_by_split: dict[str, list[str]] = {}
    raw_rows_by_split: dict[str, list[dict[str, str]]] = {}
    clean_rows_by_split: dict[str, list[dict[str, str]]] = {}
    report: dict[str, object] = {
        "status": "ok",
        "mscxr_json": str(args.mscxr_json),
        "chest_imagenome_root": str(args.chest_imagenome_root),
        "outdir": str(args.outdir),
        "mscxr_patients": len(mscxr_patients),
        "splits": {},
        "repartition_train_valid": bool(args.repartition_train_valid),
    }

    for split in DEFAULT_SPLITS:
        path = split_csv_path(args.chest_imagenome_root, split)
        if not path.exists():
            raise FileNotFoundError(path)
        fieldnames, raw_rows = read_split_csv(path)
        clean_rows = filter_rows(raw_rows, mscxr_patients)
        fieldnames_by_split[split] = fieldnames
        raw_rows_by_split[split] = raw_rows
        clean_rows_by_split[split] = clean_rows
        report["splits"][split] = {
            "source_csv": str(path),
            "raw": summarize_split(raw_rows, mscxr_patients),
            "clean": summarize_split(clean_rows, mscxr_patients),
            "excluded_rows": len(raw_rows) - len(clean_rows),
            "excluded_patients": len(patient_set(raw_rows) & mscxr_patients),
        }

    if args.repartition_train_valid:
        new_train, new_valid = repartition_train_valid(
            clean_rows_by_split["train"],
            clean_rows_by_split["valid"],
            args.heldout_fraction,
            args.seed,
        )
        clean_rows_by_split["train"] = new_train
        clean_rows_by_split["valid"] = new_valid
        report["heldout_fraction"] = args.heldout_fraction
        report["seed"] = args.seed

    train_patients = patient_set(clean_rows_by_split["train"])
    valid_patients = patient_set(clean_rows_by_split["valid"])
    test_patients = patient_set(clean_rows_by_split["test"])
    report["sanity"] = {
        "train_mscxr_overlap": len(train_patients & mscxr_patients),
        "valid_mscxr_overlap": len(valid_patients & mscxr_patients),
        "test_mscxr_overlap": len(test_patients & mscxr_patients),
        "train_valid_overlap": len(train_patients & valid_patients),
        "train_test_overlap": len(train_patients & test_patients),
        "valid_test_overlap": len(valid_patients & test_patients),
    }

    if report["sanity"]["train_mscxr_overlap"] or report["sanity"]["valid_mscxr_overlap"]:
        report["status"] = "leakage_found"

    for split, rows in clean_rows_by_split.items():
        write_split_csv(args.outdir / f"imagenome_{split}_clean.csv", fieldnames_by_split[split], rows)

    report_path = args.outdir / "leakage_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    raw_source_overlap = sum(
        report["splits"][split]["raw"]["mscxr_patient_overlap"] for split in DEFAULT_SPLITS
    )
    if args.fail_on_source_overlap and raw_source_overlap:
        raise SystemExit(
            f"Raw Chest ImaGenome splits contain {raw_source_overlap} MS-CXR patient overlaps; "
            f"filtered splits were still written to {args.outdir}"
        )

    return report


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

