"""Build leakage-free MRSG image-report JSONL manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from anaprior.data.mrsg_phrases import build_counterfactuals, load_disease_descriptions, mine_report_phrases
from anaprior.data.mrsg_protocol import (
    ProtocolSplitResult,
    build_mscxr_exclusion_set,
    filter_and_split_mimic_rows,
    write_protocol_manifest,
)


FORBIDDEN_SPATIAL_KEYS = frozenset(
    {"box", "bbox", "mask", "region", "coordinates", "oracle", "dcem"}
)


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped[0] in "[{"


def _decode_structured_value(value: Any) -> Any:
    if isinstance(value, str) and _looks_like_json(value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).strip().lower())


def _find_forbidden_spatial_keys(value: Any, prefix: str = "") -> list[str]:
    forbidden: list[str] = []
    value = _decode_structured_value(value)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if _normalize_key(key) in FORBIDDEN_SPATIAL_KEYS:
                forbidden.append(key_path)
            forbidden.extend(_find_forbidden_spatial_keys(nested, key_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            forbidden.extend(_find_forbidden_spatial_keys(nested, f"{prefix}[{index}]"))
    return forbidden


def _assert_no_forbidden_spatial_fields(payload: Mapping[str, Any]) -> None:
    forbidden: list[str] = []
    for key, value in payload.items():
        if _normalize_key(key) in FORBIDDEN_SPATIAL_KEYS:
            forbidden.append(str(key))
        forbidden.extend(_find_forbidden_spatial_keys(value, str(key)))
    if forbidden:
        unique = ", ".join(sorted(set(forbidden)))
        raise ValueError(f"Found forbidden spatial supervision fields: {unique}")


def _manifest_row(
    *,
    row: Mapping[str, Any],
    phrase_text: str,
    finding: str,
    description: str,
    negative_phrases: Sequence[str],
) -> dict[str, Any]:
    payload = {
        "image_path": str(row["normalized_path"]),
        "subject_id": str(row["subject_id"]),
        "study_id": str(row["study_id"]),
        "dicom_id": str(row["dicom_id"]),
        "phrase": phrase_text,
        "finding": finding,
        "disease_description": description,
        "negative_phrases": list(negative_phrases),
    }
    _assert_no_forbidden_spatial_fields(payload)
    return payload


def _build_rows(
    split_rows: pd.DataFrame,
    descriptions: Mapping[str, str],
    passthrough_columns: Sequence[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for _, row in split_rows.iterrows():
        raw = row.to_dict()
        for column in passthrough_columns:
            if column in raw:
                _assert_no_forbidden_spatial_fields({column: raw[column]})
        phrases = [
            item
            for item in mine_report_phrases(str(raw["report"]))
            if (not item.negated) and (not item.uncertain)
        ]
        for phrase in phrases:
            negatives = build_counterfactuals(phrase, str(raw["report"]), max_negatives=8)
            output.append(
                _manifest_row(
                    row=raw,
                    phrase_text=phrase.text,
                    finding=phrase.finding,
                    description=descriptions.get(phrase.finding, phrase.finding.lower()),
                    negative_phrases=[item.text for item in negatives],
                )
            )
    return output


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    ordered_rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(ordered_rows)


def _rebalance_if_needed(split: ProtocolSplitResult) -> ProtocolSplitResult:
    if split.all_rows.empty or (not split.train_rows.empty and not split.valid_rows.empty):
        return split

    all_rows = split.all_rows.copy()
    donor_split = "train" if split.valid_rows.empty else "valid"
    receiver_split = "valid" if donor_split == "train" else "train"
    donor_rows = all_rows.loc[all_rows["split"] == donor_split]
    subject_id = sorted(str(value) for value in donor_rows["subject_id"].unique())[-1]
    all_rows.loc[all_rows["subject_id"] == subject_id, "split"] = receiver_split
    train_rows = all_rows.loc[all_rows["split"] == "train"].reset_index(drop=True)
    valid_rows = all_rows.loc[all_rows["split"] == "valid"].reset_index(drop=True)
    all_rows = all_rows.reset_index(drop=True)
    return ProtocolSplitResult(
        all_rows=all_rows,
        train_rows=train_rows,
        valid_rows=valid_rows,
        num_rows_before_exclusion=split.num_rows_before_exclusion,
        num_excluded_mscxr_rows=split.num_excluded_mscxr_rows,
        valid_fraction=split.valid_fraction,
        seed=split.seed,
    )


def build_mrsg_image_report_cache(
    *,
    mimic_csv: Path,
    mscxr_json: Path,
    descriptions_json: Path | None,
    outdir: Path,
    valid_fraction: float = 0.2,
    seed: int = 13,
    passthrough_columns: Sequence[str] = (),
) -> dict[str, Any]:
    rows = pd.read_csv(mimic_csv)
    exclusions = build_mscxr_exclusion_set(mscxr_json)
    split = filter_and_split_mimic_rows(
        rows,
        excluded_subjects=exclusions.subjects,
        excluded_studies=exclusions.studies,
        excluded_dicoms=exclusions.dicoms,
        excluded_paths=exclusions.paths,
        valid_fraction=valid_fraction,
        seed=seed,
        path_column="path",
        report_column="report",
    )
    split = _rebalance_if_needed(split)
    descriptions = load_disease_descriptions(descriptions_json)

    outdir.mkdir(parents=True, exist_ok=True)
    protocol_path = outdir / "mrsg_protocol_manifest.json"
    protocol = write_protocol_manifest(protocol_path, split, exclusions)
    train_rows = _build_rows(split.train_rows, descriptions, passthrough_columns)
    valid_rows = _build_rows(split.valid_rows, descriptions, passthrough_columns)

    train_path = outdir / "train_mrsg.jsonl"
    valid_path = outdir / "valid_mrsg.jsonl"
    train_count = _write_jsonl(train_path, train_rows)
    valid_count = _write_jsonl(valid_path, valid_rows)

    report = {
        "status": "ok",
        "mimic_csv": str(mimic_csv),
        "mscxr_json": str(mscxr_json),
        "descriptions_json": "" if descriptions_json is None else str(descriptions_json),
        "outdir": str(outdir),
        "train_jsonl": str(train_path),
        "valid_jsonl": str(valid_path),
        "protocol_manifest": str(protocol_path),
        "train_rows": int(train_count),
        "valid_rows": int(valid_count),
        "uses_spatial_annotations": False,
        "uses_dcem": False,
        "protocol": protocol,
        "sanity": dict(protocol["sanity"]),
    }
    (outdir / "mrsg_image_report_cache_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MRSG image-report JSONL manifests.")
    parser.add_argument("--mimic-csv", required=True, type=Path)
    parser.add_argument("--mscxr-json", required=True, type=Path)
    parser.add_argument("--descriptions-json", default=None, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--passthrough-columns", nargs="*", default=())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_mrsg_image_report_cache(
        mimic_csv=args.mimic_csv,
        mscxr_json=args.mscxr_json,
        descriptions_json=args.descriptions_json,
        outdir=args.outdir,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        passthrough_columns=tuple(args.passthrough_columns),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
