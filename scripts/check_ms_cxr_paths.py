"""Check whether MS-CXR annotation paths resolve under a MIMIC-CXR-JPG root.

MS-CXR stores image paths like:
    files/p10/p10233088/s54276838/xxx.jpg

AFLoc's loader strips the leading "files/" before joining with MIMIC_IMG_DIR.
Therefore:
    if your actual images are root/files/p10/...  -> use MIMIC_CXR_JPG_ROOT=root/files
    if your actual images are root/p10/...        -> use MIMIC_CXR_JPG_ROOT=root
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe MS-CXR and MIMIC-CXR-JPG path layout.")
    parser.add_argument(
        "--ms-cxr",
        default="data/ms-cxr/1.1.0/MS_CXR_Local_Alignment_v1.1.0.csv",
        help="MS-CXR CSV or COCO JSON annotation file.",
    )
    parser.add_argument(
        "--mimic-root",
        required=True,
        help="Candidate local MIMIC-CXR-JPG root, e.g. /mnt/mimic-cxr/jpg or D:/mimic-cxr/jpg.",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of annotation paths to test.")
    return parser.parse_args()


def load_paths(path: Path, n: int) -> list[str]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [img["path"] for img in data["images"][:n]]

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        paths = []
        for row in reader:
            paths.append(row["path"])
            if len(paths) >= n:
                break
        return paths


def score_layout(root: Path, rel_paths: list[str]) -> list[tuple[str, Path, int, Path | None]]:
    layouts = [
        ("root_plus_ms_cxr_path", root, lambda p: p),
        ("root_plus_path_without_files", root, lambda p: p.replace("files/", "", 1)),
        ("root_files_plus_path_without_files", root / "files", lambda p: p.replace("files/", "", 1)),
        ("root_jpg_files_plus_path_without_files", root / "jpg" / "files", lambda p: p.replace("files/", "", 1)),
        ("root_2_1_0_files_plus_path_without_files", root / "2.1.0" / "files", lambda p: p.replace("files/", "", 1)),
    ]

    results = []
    for name, base, transform in layouts:
        hits = 0
        first_hit = None
        for rel in rel_paths:
            candidate = base / transform(rel)
            if candidate.exists():
                hits += 1
                if first_hit is None:
                    first_hit = candidate
        results.append((name, base, hits, first_hit))
    return results


def main() -> int:
    args = parse_args()
    ms_cxr = Path(args.ms_cxr)
    root = Path(args.mimic_root)
    if not ms_cxr.exists():
        raise FileNotFoundError(f"MS-CXR annotation not found: {ms_cxr}")

    rel_paths = load_paths(ms_cxr, args.n)
    print(f"MS-CXR annotation: {ms_cxr}")
    print(f"Candidate MIMIC root: {root}")
    print(f"Testing {len(rel_paths)} paths")
    print()
    print("First MS-CXR relative path:")
    print(f"  {rel_paths[0]}")
    print()

    best = None
    for name, base, hits, first_hit in score_layout(root, rel_paths):
        print(f"{name}: hits={hits}/{len(rel_paths)}")
        print(f"  base: {base}")
        if first_hit:
            print(f"  first hit: {first_hit}")
        print()
        if best is None or hits > best[2]:
            best = (name, base, hits, first_hit)

    if best and best[2] > 0:
        print("Recommended setting for AFLoc localization:")
        print(f"  $env:MIMIC_CXR_JPG_ROOT = '{best[1]}'")
        print()
        print("Remember: AFLoc strips leading 'files/' internally for MS-CXR.")
    else:
        print("No tested layout found images. The supplied root may be one level too high/low,")
        print("or this Windows environment cannot access that path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
