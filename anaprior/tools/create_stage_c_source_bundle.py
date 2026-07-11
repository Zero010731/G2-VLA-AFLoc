"""Create a small source-only bundle for moving Stage C code to a server."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_INCLUDE_PATHS = (
    "README_NEXT_STAGE.md",
    "afloc/hf_utils.py",
    "afloc/models/afloc_model.py",
    "afloc/models/text_model.py",
    "anaprior",
    "docs",
    "scripts/run_stage_c_learned_repair_server.sh",
    "scripts/run_stage_d_oracle_recoverability_server.sh",
    "scripts/run_stage_e_8class_gap_diagnosis_server.sh",
    "tests",
)

EXCLUDED_DIR_NAMES = {".git", ".pytest_cache", "__pycache__", "outputs"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
MANIFEST_NAME = "ANAPRIOR_STAGE_C_BUNDLE_MANIFEST.json"


def _as_archive_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_excluded(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    if any(part in EXCLUDED_DIR_NAMES for part in rel_parts):
        return True
    return path.suffix in EXCLUDED_SUFFIXES


def iter_bundle_files(root: Path, include_paths: Iterable[str] = DEFAULT_INCLUDE_PATHS) -> list[Path]:
    """Return deterministic source files that belong in the Stage C bundle."""
    root = root.resolve()
    files: list[Path] = []

    for include_path in include_paths:
        candidate = (root / include_path).resolve()
        if not candidate.exists():
            continue
        if candidate.is_file():
            if not _is_excluded(candidate, root):
                files.append(candidate)
            continue

        for path in candidate.rglob("*"):
            if path.is_file() and not _is_excluded(path, root):
                files.append(path)

    return sorted(files, key=lambda path: _as_archive_name(path, root))


def create_bundle(root: Path | str, output: Path | str) -> dict[str, object]:
    """Write a zip bundle and return its manifest."""
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = iter_bundle_files(root_path)
    manifest = {
        "bundle": output_path.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root_path),
        "file_count": len(files),
        "files": [_as_archive_name(path, root_path) for path in files],
        "excluded": {
            "directories": sorted(EXCLUDED_DIR_NAMES),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
        },
    }

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zf:
        for path, archive_name in zip(files, manifest["files"]):
            zf.write(path, archive_name)
        zf.writestr(
            MANIFEST_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        type=Path,
        help="Project root to bundle. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        default=Path("outputs") / "anaprior_stage_c_source_bundle.zip",
        type=Path,
        help="Output zip path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = create_bundle(args.root, args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
