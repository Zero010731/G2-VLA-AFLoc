import csv
import json
from pathlib import Path

from anaprior.data.build_region_finding_table import build_region_finding_table


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "study_id", "dicom_id", "path", "ViewPosition"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_scene_graph(root: Path, dicom_id: str, payload: dict) -> None:
    scene_dir = root / "silver_dataset" / "scene_graph" / "scene_graph"
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / f"{dicom_id}_SceneGraph.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_builds_candidate_region_rows_and_summary(tmp_path: Path) -> None:
    root = tmp_path / "chest-imagenome"
    split_csv = tmp_path / "imagenome_train_clean.csv"
    outdir = tmp_path / "out"
    write_csv(
        split_csv,
        [
            {
                "subject_id": "10000001",
                "study_id": "50000001",
                "dicom_id": "dicom-a",
                "path": "files/p10/p10000001/s50000001/dicom-a.dcm",
                "ViewPosition": "PA",
            }
        ],
    )
    write_scene_graph(
        root,
        "dicom-a",
        {
            "image_id": "dicom-a",
            "patient_id": 10000001,
            "study_id": 50000001,
            "attributes": [
                {
                    "bbox_name": "right lung",
                    "object_id": "dicom-a_right lung",
                    "attributes": [
                        [
                            "anatomicalfinding|yes|pneumothorax",
                            "anatomicalfinding|no|pleural effusion",
                        ]
                    ],
                },
                {
                    "bbox_name": "left costophrenic angle",
                    "object_id": "dicom-a_left costophrenic angle",
                    "attributes": [
                        [
                            "anatomicalfinding|yes|pleural effusion",
                            "nlp|yes|abnormal",
                        ]
                    ],
                },
            ],
        },
    )

    report = build_region_finding_table(
        split_csv=split_csv,
        chest_imagenome_root=root,
        outdir=outdir,
        split_name="train",
        findings=["Pneumothorax", "Pleural Effusion"],
    )

    assert report["status"] == "ok"
    assert report["rows"] == 4
    assert report["scene_graphs_found"] == 1
    assert report["scene_graphs_missing"] == 0
    assert report["findings"]["Pneumothorax"]["positive_rows"] == 1
    assert report["findings"]["Pleural Effusion"]["positive_rows"] == 1

    rows = list(csv.DictReader((outdir / "region_finding_train.csv").open(encoding="utf-8")))
    assert {
        (row["region"], row["finding"], row["label"])
        for row in rows
    } == {
        ("right lung", "Pneumothorax", "1"),
        ("right lung", "Pleural Effusion", "0"),
        ("left costophrenic angle", "Pneumothorax", "0"),
        ("left costophrenic angle", "Pleural Effusion", "1"),
    }

    summary_rows = list(csv.DictReader((outdir / "region_finding_summary_train.csv").open(encoding="utf-8")))
    assert any(
        row["finding"] == "Pneumothorax"
        and row["region"] == "right lung"
        and row["positive_rows"] == "1"
        for row in summary_rows
    )


def test_counts_missing_scene_graph_without_failing(tmp_path: Path) -> None:
    root = tmp_path / "chest-imagenome"
    split_csv = tmp_path / "imagenome_valid_clean.csv"
    outdir = tmp_path / "out"
    write_csv(
        split_csv,
        [
            {
                "subject_id": "10000002",
                "study_id": "50000002",
                "dicom_id": "missing-dicom",
                "path": "files/p10/p10000002/s50000002/missing-dicom.dcm",
                "ViewPosition": "AP",
            }
        ],
    )

    report = build_region_finding_table(
        split_csv=split_csv,
        chest_imagenome_root=root,
        outdir=outdir,
        split_name="valid",
        findings=["Pneumothorax"],
    )

    assert report["status"] == "ok"
    assert report["rows"] == 0
    assert report["scene_graphs_found"] == 0
    assert report["scene_graphs_missing"] == 1
    assert (outdir / "missing_scene_graphs_valid.txt").read_text(encoding="utf-8").strip() == "missing-dicom"


def test_explicit_label_policy_drops_unmentioned_negatives(tmp_path: Path) -> None:
    root = tmp_path / "chest-imagenome"
    split_csv = tmp_path / "imagenome_train_clean.csv"
    outdir = tmp_path / "out"
    write_csv(
        split_csv,
        [
            {
                "subject_id": "10000003",
                "study_id": "50000003",
                "dicom_id": "dicom-b",
                "path": "files/p10/p10000003/s50000003/dicom-b.dcm",
                "ViewPosition": "PA",
            }
        ],
    )
    write_scene_graph(
        root,
        "dicom-b",
        {
            "image_id": "dicom-b",
            "patient_id": 10000003,
            "study_id": 50000003,
            "objects": [
                {"bbox_name": "right lung", "object_id": "dicom-b_right lung"},
                {"bbox_name": "left lung", "object_id": "dicom-b_left lung"},
            ],
            "attributes": [
                {
                    "bbox_name": "right lung",
                    "object_id": "dicom-b_right lung",
                    "attributes": [["anatomicalfinding|yes|pneumothorax"]],
                }
            ],
        },
    )

    report = build_region_finding_table(
        split_csv=split_csv,
        chest_imagenome_root=root,
        outdir=outdir,
        split_name="train",
        findings=["Pneumothorax"],
        label_policy="explicit",
    )

    assert report["rows"] == 1
    rows = list(csv.DictReader((outdir / "region_finding_train.csv").open(encoding="utf-8")))
    assert rows[0]["region"] == "right lung"
    assert rows[0]["label"] == "1"
    assert rows[0]["label_source"] == "explicit_yes"


def test_builds_alias_and_namespace_labels_for_mscxr_eight_class_names(tmp_path: Path) -> None:
    root = tmp_path / "chest-imagenome"
    split_csv = tmp_path / "imagenome_train_clean.csv"
    outdir = tmp_path / "out"
    write_csv(
        split_csv,
        [
            {
                "subject_id": "10000004",
                "study_id": "50000004",
                "dicom_id": "dicom-c",
                "path": "files/p10/p10000004/s50000004/dicom-c.dcm",
                "ViewPosition": "PA",
            }
        ],
    )
    write_scene_graph(
        root,
        "dicom-c",
        {
            "image_id": "dicom-c",
            "patient_id": 10000004,
            "study_id": 50000004,
            "objects": [
                {"bbox_name": "cardiac silhouette", "object_id": "dicom-c_cardiac silhouette"},
                {"bbox_name": "left hilar structures", "object_id": "dicom-c_left hilar structures"},
                {"bbox_name": "right lower lung zone", "object_id": "dicom-c_right lower lung zone"},
            ],
            "attributes": [
                {
                    "bbox_name": "cardiac silhouette",
                    "object_id": "dicom-c_cardiac silhouette",
                    "attributes": [["anatomicalfinding|yes|enlarged cardiac silhouette"]],
                },
                {
                    "bbox_name": "left hilar structures",
                    "object_id": "dicom-c_left hilar structures",
                    "attributes": [["anatomicalfinding|yes|vascular congestion"]],
                },
                {
                    "bbox_name": "right lower lung zone",
                    "object_id": "dicom-c_right lower lung zone",
                    "attributes": [["disease|yes|pneumonia"]],
                },
            ],
        },
    )

    report = build_region_finding_table(
        split_csv=split_csv,
        chest_imagenome_root=root,
        outdir=outdir,
        split_name="train",
        findings=["Cardiomegaly", "Edema", "Pneumonia"],
        label_policy="explicit",
    )

    assert report["findings"]["Cardiomegaly"]["positive_rows"] == 1
    assert report["findings"]["Edema"]["positive_rows"] == 1
    assert report["findings"]["Pneumonia"]["positive_rows"] == 1
    assert "anatomicalfinding|enlarged cardiac silhouette" in report["finding_attribute_sources"]["Cardiomegaly"]
    assert "anatomicalfinding|vascular congestion" in report["finding_attribute_sources"]["Edema"]
    assert "disease|pneumonia" in report["finding_attribute_sources"]["Pneumonia"]

    rows = list(csv.DictReader((outdir / "region_finding_train.csv").open(encoding="utf-8")))
    positives = {
        (row["region"], row["finding"], row["raw_attributes"])
        for row in rows
        if row["label"] == "1"
    }
    assert positives == {
        (
            "cardiac silhouette",
            "Cardiomegaly",
            "anatomicalfinding|yes|enlarged cardiac silhouette",
        ),
        ("left hilar structures", "Edema", "anatomicalfinding|yes|vascular congestion"),
        ("right lower lung zone", "Pneumonia", "disease|yes|pneumonia"),
    }


def test_reports_alias_sources_regions_and_pair_conflicts(tmp_path: Path) -> None:
    root = tmp_path / "chest-imagenome"
    split_csv = tmp_path / "imagenome_train_clean.csv"
    outdir = tmp_path / "out"
    write_csv(
        split_csv,
        [
            {
                "subject_id": "10000005",
                "study_id": "50000005",
                "dicom_id": "dicom-d",
                "path": "files/p10/p10000005/s50000005/dicom-d.dcm",
                "ViewPosition": "PA",
            }
        ],
    )
    write_scene_graph(
        root,
        "dicom-d",
        {
            "image_id": "dicom-d",
            "patient_id": 10000005,
            "study_id": 50000005,
            "objects": [
                {"bbox_name": "cardiac silhouette", "object_id": "dicom-d_cardiac silhouette"},
                {"bbox_name": "right lower lung zone", "object_id": "dicom-d_right lower lung zone"},
                {"bbox_name": "left lower lung zone", "object_id": "dicom-d_left lower lung zone"},
            ],
            "attributes": [
                {
                    "bbox_name": "cardiac silhouette",
                    "object_id": "dicom-d_cardiac silhouette",
                    "attributes": [["anatomicalfinding|yes|enlarged cardiac silhouette"]],
                },
                {
                    "bbox_name": "right lower lung zone",
                    "object_id": "dicom-d_right lower lung zone",
                    "attributes": [
                        [
                            "anatomicalfinding|yes|pulmonary edema/hazy opacity",
                            "anatomicalfinding|yes|lung opacity",
                        ]
                    ],
                },
                {
                    "bbox_name": "left lower lung zone",
                    "object_id": "dicom-d_left lower lung zone",
                    "attributes": [
                        [
                            "disease|yes|pneumonia",
                            "anatomicalfinding|yes|consolidation",
                        ]
                    ],
                },
            ],
        },
    )

    report = build_region_finding_table(
        split_csv=split_csv,
        chest_imagenome_root=root,
        outdir=outdir,
        split_name="train",
        findings=["Cardiomegaly", "Edema", "Pneumonia", "Lung Opacity", "Consolidation"],
        label_policy="explicit",
    )

    assert report["positive_attribute_source_counts"]["Cardiomegaly"] == {
        "anatomicalfinding|yes|enlarged cardiac silhouette": 1
    }
    assert report["positive_attribute_source_counts"]["Edema"] == {
        "anatomicalfinding|yes|pulmonary edema/hazy opacity": 1
    }
    assert report["positive_attribute_source_counts"]["Pneumonia"] == {"disease|yes|pneumonia": 1}
    assert report["positive_region_counts"]["Cardiomegaly"] == {"cardiac silhouette": 1}
    assert report["positive_region_counts"]["Edema"] == {"right lower lung zone": 1}
    assert report["positive_region_counts"]["Pneumonia"] == {"left lower lung zone": 1}

    conflicts = {
        (entry["finding_a"], entry["finding_b"]): entry
        for entry in report["positive_pair_conflicts"]
    }
    edema_lung_opacity = conflicts[("Edema", "Lung Opacity")]
    assert edema_lung_opacity["positive_region_pairs"] == 1
    assert edema_lung_opacity["fraction_of_a_positive_rows"] == 1.0
    assert edema_lung_opacity["fraction_of_b_positive_rows"] == 1.0
    pneumonia_consolidation = conflicts[("Consolidation", "Pneumonia")]
    assert pneumonia_consolidation["positive_region_pairs"] == 1
    assert pneumonia_consolidation["fraction_of_a_positive_rows"] == 1.0
    assert pneumonia_consolidation["fraction_of_b_positive_rows"] == 1.0
