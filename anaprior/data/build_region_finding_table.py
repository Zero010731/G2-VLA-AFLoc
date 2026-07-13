"""Build Chest ImaGenome region-finding labels for AnaPrior-Loc.

The output is a weak supervision table for the Region Abnormality Predictor.
Each row is an image-region-finding tuple. Positive rows come from
`anatomicalfinding|yes|...` attributes in Chest ImaGenome scene graphs.
Explicit negations are marked separately from unmentioned negatives so later
training code can choose the conservative label policy it needs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable


DEFAULT_FINDINGS = ("Pneumothorax", "Pleural Effusion")

DEFAULT_FINDING_ATTRIBUTE_ALIASES = {
    "cardiomegaly": (
        ("anatomicalfinding", "enlarged cardiac silhouette"),
    ),
    "edema": (
        ("anatomicalfinding", "pulmonary edema/hazy opacity"),
        ("anatomicalfinding", "vascular congestion"),
    ),
    "pneumonia": (
        ("disease", "pneumonia"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build region-finding labels from clean Chest ImaGenome splits."
    )
    parser.add_argument("--split-csv", required=True, type=Path, help="Clean Chest ImaGenome split CSV.")
    parser.add_argument(
        "--chest-imagenome-root",
        required=True,
        type=Path,
        help="Root directory of chest-imagenome_1.0.0.",
    )
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--split-name", default="train", help="Name used in output filenames.")
    parser.add_argument(
        "--findings",
        default=",".join(DEFAULT_FINDINGS),
        help="Comma-separated finding names to extract.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional smoke-test limit on clean split rows.",
    )
    parser.add_argument(
        "--label-policy",
        choices=("all", "explicit"),
        default="all",
        help=(
            "all writes every region-finding tuple; explicit drops unmentioned "
            "negatives and keeps only explicit yes/no labels."
        ),
    )
    return parser.parse_args()


def normalize_finding(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").replace("-", " ").split())


def canonical_findings(values: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        name = " ".join(value.strip().split())
        if not name:
            continue
        mapping[normalize_finding(name)] = name
    if not mapping:
        raise ValueError("No findings were provided")
    return mapping


def finding_attribute_lookup(values: Iterable[str]) -> tuple[dict[tuple[str, str], str], dict[str, list[str]]]:
    findings = canonical_findings(values)
    lookup: dict[tuple[str, str], str] = {}
    sources: dict[str, list[str]] = {}

    for normalized_name, canonical in findings.items():
        raw_sources = [("anatomicalfinding", normalized_name)]
        raw_sources.extend(DEFAULT_FINDING_ATTRIBUTE_ALIASES.get(normalized_name, ()))
        for namespace, raw_finding in raw_sources:
            normalized_source = (namespace.strip(), normalize_finding(raw_finding))
            lookup[normalized_source] = canonical
            sources.setdefault(canonical, []).append(f"{normalized_source[0]}|{normalized_source[1]}")

    return lookup, {finding: sorted(set(values)) for finding, values in sources.items()}


def read_csv(path: Path, max_rows: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(row)
    if not fieldnames:
        raise ValueError(f"CSV has no header: {path}")
    for required in ("subject_id", "study_id", "dicom_id"):
        if required not in fieldnames:
            raise ValueError(f"CSV missing {required} column: {path}")
    return fieldnames, rows


def scene_graph_path(chest_imagenome_root: Path, dicom_id: str) -> Path:
    return (
        chest_imagenome_root
        / "silver_dataset"
        / "scene_graph"
        / "scene_graph"
        / f"{dicom_id}_SceneGraph.json"
    )


def flatten_attribute_groups(attribute_groups: object) -> list[str]:
    if not isinstance(attribute_groups, list):
        return []
    flattened: list[str] = []
    for group in attribute_groups:
        if isinstance(group, list):
            flattened.extend(str(item) for item in group)
        elif isinstance(group, str):
            flattened.append(group)
    return flattened


def parse_finding_attribute(attribute: str) -> tuple[str, str, str] | None:
    parts = attribute.split("|", 2)
    if len(parts) != 3:
        return None
    namespace, polarity, finding = (part.strip() for part in parts)
    if namespace not in {"anatomicalfinding", "disease"}:
        return None
    if polarity not in {"yes", "no"}:
        return None
    return namespace, polarity, normalize_finding(finding)


def build_region_index(scene_graph: dict) -> dict[str, dict[str, str]]:
    regions: dict[str, dict[str, str]] = {}
    for obj in scene_graph.get("objects", []):
        if not isinstance(obj, dict):
            continue
        region = str(obj.get("bbox_name") or obj.get("name") or "").strip()
        if not region:
            continue
        regions[region] = {
            "region": region,
            "object_id": str(obj.get("object_id") or ""),
            "x1": str(obj.get("x1") or ""),
            "y1": str(obj.get("y1") or ""),
            "x2": str(obj.get("x2") or ""),
            "y2": str(obj.get("y2") or ""),
            "width": str(obj.get("width") or ""),
            "height": str(obj.get("height") or ""),
        }

    for attr in scene_graph.get("attributes", []):
        if not isinstance(attr, dict):
            continue
        region = str(attr.get("bbox_name") or attr.get("name") or "").strip()
        if not region:
            continue
        if region not in regions:
            regions[region] = {
                "region": region,
                "object_id": str(attr.get("object_id") or ""),
                "x1": "",
                "y1": "",
                "x2": "",
                "y2": "",
                "width": "",
                "height": "",
            }
        elif not regions[region].get("object_id"):
            regions[region]["object_id"] = str(attr.get("object_id") or "")
    return regions


def collect_region_labels(
    scene_graph: dict,
    finding_lookup: dict[tuple[str, str], str],
) -> dict[str, dict[str, dict[str, object]]]:
    labels: dict[str, dict[str, dict[str, object]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "yes": 0,
                "no": 0,
                "phrases": [],
                "attributes": [],
            }
        )
    )

    for attr in scene_graph.get("attributes", []):
        if not isinstance(attr, dict):
            continue
        region = str(attr.get("bbox_name") or attr.get("name") or "").strip()
        if not region:
            continue
        phrases = attr.get("phrases") if isinstance(attr.get("phrases"), list) else []
        raw_attributes = flatten_attribute_groups(attr.get("attributes"))
        for raw_attribute in raw_attributes:
            parsed = parse_finding_attribute(raw_attribute)
            if parsed is None:
                continue
            namespace, polarity, normalized_finding = parsed
            lookup_key = (namespace, normalized_finding)
            if lookup_key not in finding_lookup:
                continue
            canonical = finding_lookup[lookup_key]
            labels[region][canonical][polarity] += 1
            labels[region][canonical]["attributes"].append(raw_attribute)
            labels[region][canonical]["phrases"].extend(str(phrase) for phrase in phrases)
    return labels


def label_for_region_finding(
    labels: dict[str, dict[str, dict[str, object]]],
    region: str,
    finding: str,
) -> tuple[int, str, int, int, list[str], list[str]]:
    info = labels.get(region, {}).get(finding)
    if not info:
        return 0, "unmentioned_negative", 0, 0, [], []

    yes_count = int(info["yes"])
    no_count = int(info["no"])
    phrases = sorted(set(str(item) for item in info["phrases"]))
    attributes = sorted(set(str(item) for item in info["attributes"]))

    if yes_count > 0:
        return 1, "explicit_yes", yes_count, no_count, phrases, attributes
    if no_count > 0:
        return 0, "explicit_no", yes_count, no_count, phrases, attributes
    return 0, "unmentioned_negative", yes_count, no_count, phrases, attributes


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(grouped: dict[tuple[str, str], Counter]) -> list[dict[str, object]]:
    summary_rows = []
    for (finding, region), counts in sorted(grouped.items()):
        rows = int(counts["rows"])
        positives = int(counts["positive_rows"])
        summary_rows.append(
            {
                "finding": finding,
                "region": region,
                "rows": rows,
                "positive_rows": positives,
                "prevalence": f"{positives / rows:.6f}" if rows else "0.000000",
                "explicit_yes_rows": int(counts["explicit_yes"]),
                "explicit_no_rows": int(counts["explicit_no"]),
                "unmentioned_negative_rows": int(counts["unmentioned_negative"]),
            }
        )
    return summary_rows


def counter_report(values: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {
        finding: {key: int(count) for key, count in sorted(counter.items())}
        for finding, counter in sorted(values.items())
    }


def build_pair_conflict_report(
    pair_counts: Counter[tuple[str, str]],
    finding_summary: dict[str, Counter],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (finding_a, finding_b), count in sorted(pair_counts.items()):
        positives_a = int(finding_summary[finding_a]["positive_rows"])
        positives_b = int(finding_summary[finding_b]["positive_rows"])
        rows.append(
            {
                "finding_a": finding_a,
                "finding_b": finding_b,
                "positive_region_pairs": int(count),
                "fraction_of_a_positive_rows": count / positives_a if positives_a else 0.0,
                "fraction_of_b_positive_rows": count / positives_b if positives_b else 0.0,
            }
        )
    return rows


def build_region_finding_table(
    split_csv: Path,
    chest_imagenome_root: Path,
    outdir: Path,
    split_name: str,
    findings: Iterable[str],
    max_rows: int | None = None,
    label_policy: str = "all",
) -> dict[str, object]:
    if label_policy not in {"all", "explicit"}:
        raise ValueError("label_policy must be 'all' or 'explicit'")

    finding_lookup, finding_attribute_sources = finding_attribute_lookup(findings)
    canonical_finding_names = sorted(set(finding_lookup.values()))
    _, split_rows = read_csv(split_csv, max_rows=max_rows)
    outdir.mkdir(parents=True, exist_ok=True)

    table_path = outdir / f"region_finding_{split_name}.csv"
    summary_path = outdir / f"region_finding_summary_{split_name}.csv"
    missing_path = outdir / f"missing_scene_graphs_{split_name}.txt"
    report_path = outdir / f"region_finding_report_{split_name}.json"

    table_fields = [
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
        "positive_attribute_count",
        "negative_attribute_count",
        "phrases",
        "raw_attributes",
    ]

    rows_written = 0
    grouped_summary: dict[tuple[str, str], Counter] = defaultdict(Counter)
    finding_summary: dict[str, Counter] = {finding: Counter() for finding in canonical_finding_names}
    positive_attribute_source_counts: dict[str, Counter] = {
        finding: Counter() for finding in canonical_finding_names
    }
    positive_region_counts: dict[str, Counter] = {finding: Counter() for finding in canonical_finding_names}
    positive_pair_counts: Counter[tuple[str, str]] = Counter()
    missing_dicoms: list[str] = []
    scene_graphs_found = 0
    regions_seen: set[str] = set()

    with table_path.open("w", encoding="utf-8", newline="") as table_handle:
        writer = csv.DictWriter(table_handle, fieldnames=table_fields)
        writer.writeheader()

        for split_row in split_rows:
            dicom_id = str(split_row["dicom_id"]).strip()
            path = scene_graph_path(chest_imagenome_root, dicom_id)
            if not path.exists():
                missing_dicoms.append(dicom_id)
                continue

            scene_graphs_found += 1
            with path.open("r", encoding="utf-8") as handle:
                scene_graph = json.load(handle)

            regions = build_region_index(scene_graph)
            labels = collect_region_labels(scene_graph, finding_lookup)
            for region, region_info in sorted(regions.items()):
                regions_seen.add(region)
                region_rows: dict[str, tuple[int, str, int, int, list[str], list[str]]] = {}
                positive_findings: list[str] = []
                for finding in canonical_finding_names:
                    label, source, yes_count, no_count, phrases, attributes = label_for_region_finding(
                        labels,
                        region,
                        finding,
                    )
                    region_rows[finding] = (label, source, yes_count, no_count, phrases, attributes)
                    if label == 1:
                        positive_findings.append(finding)
                        positive_region_counts[finding][region] += 1
                        for raw_attribute in attributes:
                            parsed = parse_finding_attribute(raw_attribute)
                            if parsed is None:
                                continue
                            namespace, polarity, normalized_finding = parsed
                            if polarity == "yes" and finding_lookup.get((namespace, normalized_finding)) == finding:
                                positive_attribute_source_counts[finding][raw_attribute] += 1

                for pair in combinations(sorted(positive_findings), 2):
                    positive_pair_counts[pair] += 1

                for finding in canonical_finding_names:
                    label, source, yes_count, no_count, phrases, attributes = region_rows[finding]
                    if label_policy == "explicit" and source == "unmentioned_negative":
                        continue

                    row = {
                            "split": split_name,
                            "subject_id": split_row.get("subject_id", ""),
                            "study_id": split_row.get("study_id", ""),
                            "dicom_id": dicom_id,
                            "image_id": scene_graph.get("image_id", dicom_id),
                            "view": split_row.get("ViewPosition", ""),
                            "region": region,
                            "object_id": region_info.get("object_id", ""),
                            "x1": region_info.get("x1", ""),
                            "y1": region_info.get("y1", ""),
                            "x2": region_info.get("x2", ""),
                            "y2": region_info.get("y2", ""),
                            "width": region_info.get("width", ""),
                            "height": region_info.get("height", ""),
                            "finding": finding,
                            "label": label,
                            "label_source": source,
                            "positive_attribute_count": yes_count,
                            "negative_attribute_count": no_count,
                            "phrases": " || ".join(phrases),
                            "raw_attributes": " || ".join(attributes),
                        }
                    writer.writerow(row)
                    rows_written += 1

                    key = (finding, region)
                    grouped_summary[key]["rows"] += 1
                    grouped_summary[key]["positive_rows"] += int(label)
                    grouped_summary[key][source] += 1
                    finding_summary[finding]["rows"] += 1
                    finding_summary[finding]["positive_rows"] += int(label)

    summary_rows = build_summary(grouped_summary)
    summary_fields = [
        "finding",
        "region",
        "rows",
        "positive_rows",
        "prevalence",
        "explicit_yes_rows",
        "explicit_no_rows",
        "unmentioned_negative_rows",
    ]
    write_rows(summary_path, summary_fields, summary_rows)
    missing_path.write_text("\n".join(missing_dicoms) + ("\n" if missing_dicoms else ""), encoding="utf-8")

    finding_report: dict[str, dict[str, object]] = {}
    for finding, counts in finding_summary.items():
        total = int(counts["rows"])
        positives = int(counts["positive_rows"])
        finding_report[finding] = {
            "rows": total,
            "positive_rows": positives,
            "prevalence": positives / total if total else 0.0,
        }

    report: dict[str, object] = {
        "status": "ok",
        "split": split_name,
        "split_csv": str(split_csv),
        "chest_imagenome_root": str(chest_imagenome_root),
        "outdir": str(outdir),
        "label_policy": label_policy,
        "input_rows": len(split_rows),
        "scene_graphs_found": scene_graphs_found,
        "scene_graphs_missing": len(missing_dicoms),
        "regions": len(regions_seen),
        "rows": rows_written,
        "findings": finding_report,
        "finding_attribute_sources": finding_attribute_sources,
        "positive_attribute_source_counts": counter_report(positive_attribute_source_counts),
        "positive_region_counts": counter_report(positive_region_counts),
        "positive_pair_conflicts": build_pair_conflict_report(positive_pair_counts, finding_summary),
        "outputs": {
            "table": str(table_path),
            "summary": str(summary_path),
            "missing_scene_graphs": str(missing_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    findings = [item.strip() for item in args.findings.split(",") if item.strip()]
    report = build_region_finding_table(
        split_csv=args.split_csv,
        chest_imagenome_root=args.chest_imagenome_root,
        outdir=args.outdir,
        split_name=args.split_name,
        findings=findings,
        max_rows=args.max_rows,
        label_policy=args.label_policy,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
