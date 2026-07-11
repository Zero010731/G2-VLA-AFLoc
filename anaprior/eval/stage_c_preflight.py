"""Preflight checks for the Stage C learned repair server run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageCPreflightConfig:
    ckpt: Path
    mimic_image_root: Path
    chest_imagenome_root: Path
    prior_table: Path
    base_hmaps_npy: Path
    predictor_ckpt: Path
    outroot: Path
    report_json: Path | None = None


def _path_record(name: str, path: Path, kind: str) -> dict[str, str]:
    return {"name": name, "path": str(path), "kind": kind}


def _check_path(name: str, path: Path, kind: str, missing: list[dict[str, str]], present: list[dict[str, str]]) -> None:
    exists = path.is_dir() if kind == "dir" else path.is_file()
    record = _path_record(name, path, kind)
    if exists:
        present.append(record)
    else:
        missing.append(record)


def run_stage_c_preflight(config: StageCPreflightConfig) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    present: list[dict[str, str]] = []
    _check_path("ckpt", config.ckpt, "file", missing, present)
    _check_path("mimic_image_root", config.mimic_image_root, "dir", missing, present)
    _check_path("chest_imagenome_root", config.chest_imagenome_root, "dir", missing, present)
    _check_path(
        "chest_imagenome_scene_graph_zip",
        config.chest_imagenome_root / "silver_dataset" / "scene_graph.zip",
        "file",
        missing,
        present,
    )
    _check_path("prior_table", config.prior_table, "file", missing, present)
    _check_path("base_hmaps_npy", config.base_hmaps_npy, "file", missing, present)
    _check_path("predictor_ckpt", config.predictor_ckpt, "file", missing, present)

    config.outroot.mkdir(parents=True, exist_ok=True)
    writable_probe = config.outroot / ".stage_c_preflight_write_test"
    writable = True
    try:
        writable_probe.write_text("ok", encoding="utf-8")
        writable_probe.unlink()
    except OSError:
        writable = False
        missing.append(_path_record("outroot_writable", config.outroot, "dir"))

    report = {
        "status": "ok" if not missing and writable else "error",
        "present": present,
        "missing": missing,
        "outroot": str(config.outroot),
        "outroot_writable": writable,
    }
    if config.report_json is not None:
        config.report_json.parent.mkdir(parents=True, exist_ok=True)
        config.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check required inputs before Stage C learned repair run.")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--mimic-image-root", required=True, type=Path)
    parser.add_argument("--chest-imagenome-root", required=True, type=Path)
    parser.add_argument("--prior-table", required=True, type=Path)
    parser.add_argument("--base-hmaps-npy", required=True, type=Path)
    parser.add_argument("--predictor-ckpt", required=True, type=Path)
    parser.add_argument("--outroot", required=True, type=Path)
    parser.add_argument("--report-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_stage_c_preflight(
        StageCPreflightConfig(
            ckpt=args.ckpt,
            mimic_image_root=args.mimic_image_root,
            chest_imagenome_root=args.chest_imagenome_root,
            prior_table=args.prior_table,
            base_hmaps_npy=args.base_hmaps_npy,
            predictor_ckpt=args.predictor_ckpt,
            outroot=args.outroot,
            report_json=args.report_json,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
