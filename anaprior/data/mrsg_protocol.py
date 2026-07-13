"""Leakage-proof image-report protocol for box-free AFLoc-MRSG training."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

import pandas as pd


_IDENTIFIER_PART = re.compile(r"^(?P<prefix>[ps])(?P<identifier>\d+)$", re.IGNORECASE)
_FORBIDDEN_COLUMN_PARTS = frozenset(
    {
        "bbox",
        "box",
        "boxes",
        "mask",
        "masks",
        "segmentation",
        "region",
        "regions",
        "scenegraph",
        "oracle",
        "dcem",
        "heatmap",
        "heatmaps",
    }
)


@dataclass(frozen=True)
class MIMICIdentity:
    subject_id: str
    study_id: str
    dicom_id: str


@dataclass(frozen=True)
class MSCXRExclusionSet:
    subjects: FrozenSet[str]
    studies: FrozenSet[str]
    dicoms: FrozenSet[str]
    paths: FrozenSet[str]


@dataclass(frozen=True)
class ProtocolSplitResult:
    all_rows: pd.DataFrame
    train_rows: pd.DataFrame
    valid_rows: pd.DataFrame
    num_rows_before_exclusion: int
    num_excluded_mscxr_rows: int
    valid_fraction: float
    seed: int

    @property
    def num_rows_after_exclusion(self) -> int:
        return len(self.all_rows)


def _is_missing(value: object) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_identifier(value: object, prefix: Optional[str] = None) -> str:
    if value is None or _is_missing(value):
        return ""
    text = str(value).strip().lower()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if prefix and text.startswith(prefix.lower()) and text[1:].isdigit():
        text = text[1:]
    return text


def _normalize_dicom(value: object) -> str:
    text = _normalize_identifier(value)
    if not text:
        return ""
    filename = text.replace("\\", "/").rsplit("/", 1)[-1]
    return filename.rsplit(".", 1)[0] if "." in filename else filename


def normalize_mimic_path(path: object) -> str:
    """Return a case-normalized POSIX path, anchored at ``files/`` when present."""

    if path is None or _is_missing(path):
        return ""
    text = str(path).strip().replace("\\", "/")
    if not text:
        return ""
    text = re.sub(r"/+", "/", text)
    normalized = posixpath.normpath(text).lower()
    if normalized == ".":
        return ""
    files_index = normalized.find("/files/")
    if files_index >= 0:
        return normalized[files_index + 1 :]
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def identity_from_path(path: str) -> MIMICIdentity:
    """Parse a MIMIC identity without confusing shard ``p10`` for a patient."""

    normalized = normalize_mimic_path(path)
    parts = [part for part in normalized.split("/") if part]
    subjects: List[str] = []
    studies: List[str] = []
    for part in parts:
        match = _IDENTIFIER_PART.fullmatch(part)
        if match is None:
            continue
        identifier = match.group("identifier")
        if match.group("prefix").lower() == "p":
            subjects.append(identifier)
        else:
            studies.append(identifier)

    # MIMIC paths include a short shard (for example p10) before the patient.
    subject = max(subjects, key=len) if subjects else ""
    study = max(studies, key=len) if studies else ""
    dicom = _normalize_dicom(parts[-1]) if parts else ""
    return MIMICIdentity(subject, study, dicom)


def _record_paths(record: Mapping[str, Any]) -> List[str]:
    paths: List[str] = []
    for key in ("path", "file_name", "image_path", "previous_path"):
        value = record.get(key)
        if value is not None and not _is_missing(value) and str(value).strip():
            paths.append(str(value))
    return paths


def _load_exclusion_records(
    source: Union[Path, str, Mapping[str, Any], Iterable[Union[str, Mapping[str, Any]]]],
) -> List[Union[str, Mapping[str, Any]]]:
    if isinstance(source, Path):
        with source.open("r", encoding="utf-8") as handle:
            return _load_exclusion_records(json.load(handle))
    if isinstance(source, str):
        candidate = Path(source)
        if candidate.suffix.lower() == ".json" and candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return _load_exclusion_records(json.load(handle))
        return [source]
    if isinstance(source, Mapping):
        images = source.get("images")
        if isinstance(images, list):
            return list(images)
        return [source]
    return list(source)


def build_mscxr_exclusion_set(
    source: Union[Path, str, Mapping[str, Any], Iterable[Union[str, Mapping[str, Any]]]],
) -> MSCXRExclusionSet:
    """Collect every usable MS-CXR identity and normalized image path."""

    subjects = set()
    studies = set()
    dicoms = set()
    paths = set()
    records = _load_exclusion_records(source)

    for record in records:
        if isinstance(record, str):
            record_paths = [record]
            explicit = {}
        elif isinstance(record, Mapping):
            record_paths = _record_paths(record)
            explicit = record
        else:
            raise TypeError("MS-CXR records must be path strings or mappings")

        explicit_subject = _normalize_identifier(explicit.get("subject_id"), "p")
        explicit_study = _normalize_identifier(explicit.get("study_id"), "s")
        explicit_dicom = _normalize_dicom(explicit.get("dicom_id"))
        if explicit_subject:
            subjects.add(explicit_subject)
        if explicit_study:
            studies.add(explicit_study)
        if explicit_dicom:
            dicoms.add(explicit_dicom)

        for raw_path in record_paths:
            normalized_path = normalize_mimic_path(raw_path)
            if not normalized_path:
                continue
            paths.add(normalized_path)
            identity = identity_from_path(normalized_path)
            if identity.subject_id:
                subjects.add(identity.subject_id)
            if identity.study_id:
                studies.add(identity.study_id)
            if identity.dicom_id:
                dicoms.add(identity.dicom_id)

    if not subjects and not studies and not dicoms and not paths:
        raise ValueError("No usable MS-CXR exclusion identities were found")
    return MSCXRExclusionSet(
        subjects=frozenset(subjects),
        studies=frozenset(studies),
        dicoms=frozenset(dicoms),
        paths=frozenset(paths),
    )


def patient_split(subject_id: str, valid_fraction: float, seed: int) -> str:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    normalized_subject = _normalize_identifier(subject_id, "p")
    if not normalized_subject:
        raise ValueError("subject_id is required for patient splitting")
    digest = hashlib.md5(f"{seed}:{normalized_subject}".encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "valid" if unit < valid_fraction else "train"


def _forbidden_columns(columns: Iterable[object]) -> List[str]:
    forbidden = []
    for column in columns:
        name = str(column).strip().lower()
        compact = re.sub(r"[^a-z0-9]+", "", name)
        parts = set(filter(None, re.split(r"[^a-z0-9]+", name)))
        if parts & _FORBIDDEN_COLUMN_PARTS or any(
            token in compact
            for token in ("scenegraph", "regionpredictor", "spatialannotation", "oracle", "dcem")
        ):
            forbidden.append(str(column))
    return forbidden


def _normalized_exclusions(
    excluded_subjects: Iterable[object],
    excluded_studies: Iterable[object],
    excluded_dicoms: Iterable[object],
    excluded_paths: Iterable[object],
) -> MSCXRExclusionSet:
    return MSCXRExclusionSet(
        subjects=frozenset(
            value
            for value in (_normalize_identifier(item, "p") for item in excluded_subjects)
            if value
        ),
        studies=frozenset(
            value
            for value in (_normalize_identifier(item, "s") for item in excluded_studies)
            if value
        ),
        dicoms=frozenset(
            value for value in (_normalize_dicom(item) for item in excluded_dicoms) if value
        ),
        paths=frozenset(
            value
            for value in (normalize_mimic_path(item) for item in excluded_paths)
            if value
        ),
    )


def _matches_exclusion(
    identity: MIMICIdentity,
    normalized_path: str,
    exclusions: MSCXRExclusionSet,
) -> bool:
    return bool(
        identity.subject_id in exclusions.subjects
        or identity.study_id in exclusions.studies
        or identity.dicom_id in exclusions.dicoms
        or normalized_path in exclusions.paths
    )


def _identity_from_row(row: pd.Series, path_column: str) -> Tuple[MIMICIdentity, str]:
    raw_path = row.get(path_column, "")
    normalized_path = normalize_mimic_path(raw_path)
    parsed = identity_from_path(normalized_path)
    explicit_subject = _normalize_identifier(row.get("subject_id"), "p")
    explicit_study = _normalize_identifier(row.get("study_id"), "s")
    explicit_dicom = _normalize_dicom(row.get("dicom_id"))
    for name, explicit, path_value in (
        ("subject_id", explicit_subject, parsed.subject_id),
        ("study_id", explicit_study, parsed.study_id),
        ("dicom_id", explicit_dicom, parsed.dicom_id),
    ):
        if explicit and path_value and explicit != path_value:
            raise ValueError(f"{name} conflicts with path identity")
    subject = explicit_subject or parsed.subject_id
    study = explicit_study or parsed.study_id
    dicom = explicit_dicom or parsed.dicom_id
    identity = MIMICIdentity(subject, study, dicom)
    if not all((identity.subject_id, identity.study_id, identity.dicom_id)):
        raise ValueError(
            "Every MRSG row must have a parseable subject, study, and DICOM identity"
        )
    return identity, normalized_path


def filter_and_split_mimic_rows(
    rows: pd.DataFrame,
    excluded_subjects: Iterable[object] = (),
    excluded_studies: Iterable[object] = (),
    excluded_dicoms: Iterable[object] = (),
    excluded_paths: Iterable[object] = (),
    valid_fraction: float = 0.1,
    seed: int = 20260713,
    path_column: str = "path",
    report_column: str = "report",
) -> ProtocolSplitResult:
    """Validate, exclude, and deterministically split MIMIC rows by patient."""

    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    forbidden = _forbidden_columns(rows.columns)
    if forbidden:
        raise ValueError(f"MRSG rows contain prohibited supervision columns: {forbidden}")
    if report_column not in rows.columns:
        raise ValueError(f"MRSG rows are missing required report column: {report_column}")
    missing_reports = rows[report_column].map(
        lambda value: _is_missing(value) or not str(value).strip()
    )
    if bool(missing_reports.any()):
        raise ValueError(f"MRSG rows contain missing report text in column: {report_column}")

    exclusions = _normalized_exclusions(
        excluded_subjects,
        excluded_studies,
        excluded_dicoms,
        excluded_paths,
    )
    kept_records = []
    for _, row in rows.iterrows():
        identity, normalized_path = _identity_from_row(row, path_column)
        if _matches_exclusion(identity, normalized_path, exclusions):
            continue
        record = row.to_dict()
        record["subject_id"] = identity.subject_id
        record["study_id"] = identity.study_id
        record["dicom_id"] = identity.dicom_id
        record["normalized_path"] = normalized_path
        record["split"] = patient_split(identity.subject_id, valid_fraction, seed)
        kept_records.append(record)

    output_columns = list(rows.columns)
    for column in ("subject_id", "study_id", "dicom_id", "normalized_path", "split"):
        if column not in output_columns:
            output_columns.append(column)
    all_rows = pd.DataFrame.from_records(kept_records, columns=output_columns)
    train_rows = all_rows.loc[all_rows["split"] == "train"].reset_index(drop=True)
    valid_rows = all_rows.loc[all_rows["split"] == "valid"].reset_index(drop=True)
    all_rows = all_rows.reset_index(drop=True)

    train_subjects = set(train_rows["subject_id"])
    valid_subjects = set(valid_rows["subject_id"])
    if train_subjects & valid_subjects:
        raise ValueError("Train and valid subject sets overlap")
    for _, row in all_rows.iterrows():
        identity = MIMICIdentity(row["subject_id"], row["study_id"], row["dicom_id"])
        if _matches_exclusion(identity, row["normalized_path"], exclusions):
            raise ValueError("MS-CXR identity remained after exclusion")

    return ProtocolSplitResult(
        all_rows=all_rows,
        train_rows=train_rows,
        valid_rows=valid_rows,
        num_rows_before_exclusion=len(rows),
        num_excluded_mscxr_rows=len(rows) - len(all_rows),
        valid_fraction=valid_fraction,
        seed=seed,
    )


def _sha256_values(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _exclusion_hash(exclusions: MSCXRExclusionSet) -> str:
    values = []
    values.extend(f"subject:{value}" for value in exclusions.subjects)
    values.extend(f"study:{value}" for value in exclusions.studies)
    values.extend(f"dicom:{value}" for value in exclusions.dicoms)
    values.extend(f"path:{value}" for value in exclusions.paths)
    return _sha256_values(values)


def write_protocol_manifest(
    path: Union[Path, str],
    result: ProtocolSplitResult,
    exclusions: MSCXRExclusionSet,
) -> Mapping[str, Any]:
    """Validate protocol invariants and write a deterministic audit manifest."""

    train_subjects = set(result.train_rows["subject_id"])
    valid_subjects = set(result.valid_rows["subject_id"])
    train_valid_overlap = train_subjects & valid_subjects
    mscxr_overlap = 0
    for _, row in result.all_rows.iterrows():
        identity = MIMICIdentity(row["subject_id"], row["study_id"], row["dicom_id"])
        if _matches_exclusion(identity, normalize_mimic_path(row["normalized_path"]), exclusions):
            mscxr_overlap += 1

    if train_valid_overlap:
        raise ValueError("Train and valid subject sets overlap")
    if mscxr_overlap:
        raise ValueError("MS-CXR rows remain in the MRSG protocol")

    manifest = {
        "uses_spatial_annotations": False,
        "uses_dcem": False,
        "num_rows_before_exclusion": result.num_rows_before_exclusion,
        "num_rows_after_exclusion": result.num_rows_after_exclusion,
        "num_excluded_mscxr_rows": result.num_excluded_mscxr_rows,
        "valid_fraction": result.valid_fraction,
        "seed": result.seed,
        "train_subjects_sha256": _sha256_values(train_subjects),
        "valid_subjects_sha256": _sha256_values(valid_subjects),
        "exclusion_ids_sha256": _exclusion_hash(exclusions),
        "sanity": {
            "train_valid_subject_overlap": len(train_valid_overlap),
            "mscxr_overlap": mscxr_overlap,
        },
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "MIMICIdentity",
    "MSCXRExclusionSet",
    "ProtocolSplitResult",
    "build_mscxr_exclusion_set",
    "filter_and_split_mimic_rows",
    "identity_from_path",
    "normalize_mimic_path",
    "patient_split",
    "write_protocol_manifest",
]
