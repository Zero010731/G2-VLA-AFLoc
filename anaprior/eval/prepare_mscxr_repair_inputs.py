"""Prepare MS-CXR inputs for learned repair evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from anaprior.eval.eval_mscxr_learned_repair import LearnedRepairInput


def normalize_bbox_name(value: Any) -> str:
    return str(value or "").strip().lower()


def dicom_id_from_path(path: str | Path) -> str:
    return Path(str(path)).stem


def mimic_ids_from_path(path: str | Path) -> tuple[str, str]:
    parts = Path(str(path)).parts
    subject_id = ""
    study_id = ""
    for part in parts:
        if part.startswith("p") and part[1:].isdigit() and len(part) > 3:
            subject_id = part[1:]
        if part.startswith("s") and part[1:].isdigit():
            study_id = part[1:]
    return subject_id, study_id


def load_prior_regions(path: Path) -> tuple[list[str], dict[str, str]]:
    prior = json.loads(path.read_text(encoding="utf-8"))
    regions = list(prior["regions"])
    bbox_to_region = {
        normalize_bbox_name(key): str(value)
        for key, value in prior["bbox_to_region_map"].items()
    }
    return regions, bbox_to_region


def parse_region_exclusions(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {item.strip() for item in str(raw).replace(";", ",").split(",") if item.strip()}


def filter_region_space(
    regions: list[str],
    bbox_to_region: dict[str, str],
    exclude_regions: set[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    excluded = {str(region).strip() for region in (exclude_regions or set()) if str(region).strip()}
    if not excluded:
        return list(regions), dict(bbox_to_region)
    filtered_regions = [region for region in regions if region not in excluded]
    filtered_mapping = {
        bbox: region
        for bbox, region in bbox_to_region.items()
        if region not in excluded
    }
    return filtered_regions, filtered_mapping


def fill_box(mask: np.ndarray, x1: float, y1: float, x2: float, y2: float, shape: tuple[int, int]) -> None:
    h, w = shape
    x1_i = int(np.floor(max(0, min(w, x1))))
    x2_i = int(np.ceil(max(0, min(w, x2))))
    y1_i = int(np.floor(max(0, min(h, y1))))
    y2_i = int(np.ceil(max(0, min(h, y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return
    mask[y1_i:y2_i, x1_i:x2_i] = 1.0


def build_region_probability_maps(
    objects: Iterable[dict[str, Any]],
    bbox_to_region: dict[str, str],
    regions: list[str],
    shape: tuple[int, int],
) -> np.ndarray:
    region_to_idx = {region: idx for idx, region in enumerate(regions)}
    maps = np.zeros((len(regions), shape[0], shape[1]), dtype=np.float32)
    for obj in objects or []:
        region = bbox_to_region.get(normalize_bbox_name(obj.get("bbox_name")))
        if region not in region_to_idx:
            continue
        fill_box(
            maps[region_to_idx[region]],
            float(obj.get("x1", 0)),
            float(obj.get("y1", 0)),
            float(obj.get("x2", 0)),
            float(obj.get("y2", 0)),
            shape,
        )
    denom = maps.sum(axis=0, keepdims=True)
    covered = denom > 0
    maps = np.divide(maps, denom, out=np.zeros_like(maps), where=covered)
    if np.any(~covered):
        maps[:, ~covered[0]] = 1.0 / max(len(regions), 1)
    return maps


def iter_scene_graphs(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".json"):
                continue
            with zf.open(info) as handle:
                yield json.load(handle)


def collect_scene_graphs(zip_path: Path, dicom_ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for graph in iter_scene_graphs(zip_path):
        image_id = str(graph.get("image_id") or "")
        if image_id in dicom_ids:
            found[image_id] = graph
            if len(found) == len(dicom_ids):
                break
    return found


def coerce_data_rows(data: Any, max_cases: int | None = None) -> list[dict[str, Any]]:
    """Normalize AFLoc localization data into a list of row dictionaries."""

    if hasattr(data, "iloc") and hasattr(data, "to_dict"):
        frame = data.iloc[:max_cases].copy() if max_cases is not None else data
        return frame.to_dict(orient="records")
    if isinstance(data, dict):
        keys = list(data.keys())
        if not keys:
            rows: list[dict[str, Any]] = []
        else:
            length = len(data[keys[0]])
            rows = [{key: data[key][idx] for key in keys} for idx in range(length)]
    else:
        rows = [dict(row) for row in data]
    if max_cases is not None:
        rows = rows[:max_cases]
    return rows


def load_mscxr_data_rows(max_cases: int | None = None, dataset: str = "MS_CXR_CLS") -> list[dict[str, Any]]:
    from localization.datasets import load_data

    data = load_data(dataset=dataset)
    return coerce_data_rows(data, max_cases=max_cases)


def build_prepared_inputs(
    data_rows: list[dict[str, Any]],
    base_hmaps: dict[str, dict[str, Any]],
    scene_graphs: dict[str, dict[str, Any]],
    regions: list[str],
    bbox_to_region: dict[str, str],
    exclude_regions: set[str] | None = None,
) -> tuple[list[LearnedRepairInput], dict[str, Any]]:
    inputs: list[LearnedRepairInput] = []
    regions, bbox_to_region = filter_region_space(regions, bbox_to_region, exclude_regions)
    excluded_regions = sorted(str(region) for region in (exclude_regions or set()) if str(region).strip())
    missing_hmap = 0
    missing_scene_graph = 0
    for row in data_rows:
        path = str(row["path"])
        label_text = str(row.get("label_text", ""))
        category = str(row["category"])
        case_id = path + label_text
        hmap_record = base_hmaps.get(case_id)
        if hmap_record is None:
            missing_hmap += 1
            continue
        hmap = np.asarray(hmap_record["hmap"], dtype=np.float32)
        dicom_id = dicom_id_from_path(path)
        graph = scene_graphs.get(dicom_id)
        if graph is None:
            missing_scene_graph += 1
            continue
        region_maps = build_region_probability_maps(
            objects=graph.get("objects", []),
            bbox_to_region=bbox_to_region,
            regions=regions,
            shape=tuple(hmap.shape),
        )
        inputs.append(
            LearnedRepairInput(
                case_id=case_id,
                dicom_id=dicom_id,
                category=category,
                finding=category,
                heatmap=hmap,
                region_maps=region_maps,
                regions=regions,
                phrase=label_text,
            )
        )
    report = {
        "num_rows": len(data_rows),
        "num_inputs": len(inputs),
        "missing_hmap": missing_hmap,
        "missing_scene_graph": missing_scene_graph,
        "num_regions": len(regions),
        "regions": regions,
        "excluded_regions": excluded_regions,
    }
    return inputs, report


def build_score_request_rows(
    data_rows: list[dict[str, Any]],
    scene_graphs: dict[str, dict[str, Any]],
    bbox_to_region: dict[str, str],
    findings: list[str],
    exclude_regions: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    excluded = {str(region).strip() for region in (exclude_regions or set()) if str(region).strip()}
    for data_row in data_rows:
        path = str(data_row["path"])
        dicom_id = dicom_id_from_path(path)
        graph = scene_graphs.get(dicom_id)
        if graph is None:
            continue
        subject_id, study_id = mimic_ids_from_path(path)
        for obj in graph.get("objects", []) or []:
            raw_region = normalize_bbox_name(obj.get("bbox_name"))
            mapped_region = bbox_to_region.get(raw_region)
            if not mapped_region:
                continue
            if mapped_region in excluded:
                continue
            for finding in findings:
                dedupe_key = (dicom_id, str(obj.get("object_id") or raw_region), mapped_region, finding)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                rows.append(
                    {
                        "split": "mscxr",
                        "subject_id": subject_id,
                        "study_id": study_id,
                        "dicom_id": dicom_id,
                        "image_id": graph.get("image_id", dicom_id),
                        "view": data_row.get("view", data_row.get("ViewPosition", "")),
                        "region": mapped_region,
                        "object_id": str(obj.get("object_id") or ""),
                        "x1": str(obj.get("x1") or ""),
                        "y1": str(obj.get("y1") or ""),
                        "x2": str(obj.get("x2") or ""),
                        "y2": str(obj.get("y2") or ""),
                        "width": str(obj.get("width") or ""),
                        "height": str(obj.get("height") or ""),
                        "finding": finding,
                        "label": 0,
                        "label_source": "score_request",
                    }
                )
    return rows


def write_score_request_table(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "split",
        "subject_id",
        "study_id",
        "dicom_id",
        "image_id",
        "view",
        "region",
        "object_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "finding",
        "label",
        "label_source",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_prepared_inputs(inputs: list[LearnedRepairInput], report: dict[str, Any], output_npz: Path) -> dict[str, Any]:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {
            "case_id": item.case_id,
            "dicom_id": item.dicom_id,
            "category": item.category,
            "finding": item.finding,
            "phrase": item.phrase,
            "heatmap": item.heatmap,
            "region_maps": item.region_maps,
            "regions": item.regions,
        }
        for item in inputs
    ]
    np.savez_compressed(output_npz, items=np.asarray(items, dtype=object))
    final_report = dict(report)
    final_report["output_npz"] = str(output_npz)
    report_path = output_npz.with_suffix(".report.json")
    report_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    return final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MS-CXR repair inputs from baseline heatmaps and scene graphs.")
    parser.add_argument("--base-hmaps-npy", required=True, type=Path)
    parser.add_argument("--prior-table", required=True, type=Path)
    parser.add_argument("--chest-imagenome-root", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--score-request-csv", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default="MS_CXR_CLS",
        help="Localization dataset protocol. Use MS_CXR for raw phrase hmaps and MS_CXR_CLS for class prompts.",
    )
    parser.add_argument("--findings", default="Pneumothorax,Pleural Effusion")
    parser.add_argument(
        "--exclude-regions",
        default="",
        help="Comma-separated coarse repair regions to remove from inputs and score requests.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_rows = load_mscxr_data_rows(max_cases=args.max_cases, dataset=args.dataset)
    base_hmaps = np.load(args.base_hmaps_npy, allow_pickle=True).item()
    regions, bbox_to_region = load_prior_regions(args.prior_table)
    excluded_regions = parse_region_exclusions(args.exclude_regions)
    dicom_ids = {dicom_id_from_path(row["path"]) for row in data_rows}
    scene_zip = args.chest_imagenome_root / "silver_dataset" / "scene_graph.zip"
    scene_graphs = collect_scene_graphs(scene_zip, dicom_ids)
    inputs, report = build_prepared_inputs(
        data_rows=data_rows,
        base_hmaps=base_hmaps,
        scene_graphs=scene_graphs,
        regions=regions,
        bbox_to_region=bbox_to_region,
        exclude_regions=excluded_regions,
    )
    if args.score_request_csv is not None:
        findings = [item.strip() for item in str(args.findings).replace(";", ",").split(",") if item.strip()]
        score_rows = build_score_request_rows(
            data_rows=data_rows,
            scene_graphs=scene_graphs,
            bbox_to_region=bbox_to_region,
            findings=findings,
            exclude_regions=excluded_regions,
        )
        write_score_request_table(score_rows, args.score_request_csv)
        report["score_request_csv"] = str(args.score_request_csv)
        report["score_request_rows"] = len(score_rows)
    report["dataset"] = args.dataset
    final_report = save_prepared_inputs(inputs, report, args.output_npz)
    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
