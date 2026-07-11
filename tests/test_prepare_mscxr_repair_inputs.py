import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from anaprior.eval.prepare_mscxr_repair_inputs import (
    build_prepared_inputs,
    build_region_probability_maps,
    build_score_request_rows,
    coerce_data_rows,
    filter_region_space,
    load_mscxr_data_rows,
)


def test_build_region_probability_maps_normalizes_overlapping_regions() -> None:
    objects = [
        {"bbox_name": "upper box", "x1": 0, "y1": 0, "x2": 2, "y2": 2},
        {"bbox_name": "lower box", "x1": 0, "y1": 1, "x2": 2, "y2": 3},
    ]

    maps = build_region_probability_maps(
        objects=objects,
        bbox_to_region={"upper box": "upper", "lower box": "lower"},
        regions=["upper", "lower"],
        shape=(3, 2),
    )

    assert maps.shape == (2, 3, 2)
    assert np.allclose(maps[:, 1, 0], np.array([0.5, 0.5], dtype=np.float32))
    assert np.allclose(maps[:, 0, 0], np.array([1.0, 0.0], dtype=np.float32))


def test_build_prepared_inputs_joins_data_hmaps_and_scene_graphs() -> None:
    data_rows = [
        {
            "path": "/mimic/p10/p10000001/s50000001/dicom-a.jpg",
            "label_text": "prompt",
            "category": "Pneumothorax",
        }
    ]
    base_hmaps = {
        "/mimic/p10/p10000001/s50000001/dicom-a.jpgprompt": {
            "hmap": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        }
    }
    scene_graphs = {
        "dicom-a": {
            "objects": [
                {"bbox_name": "upper box", "x1": 0, "y1": 0, "x2": 2, "y2": 1},
                {"bbox_name": "lower box", "x1": 0, "y1": 1, "x2": 2, "y2": 2},
            ]
        }
    }

    inputs, report = build_prepared_inputs(
        data_rows=data_rows,
        base_hmaps=base_hmaps,
        scene_graphs=scene_graphs,
        regions=["upper", "lower"],
        bbox_to_region={"upper box": "upper", "lower box": "lower"},
    )

    assert report["num_inputs"] == 1
    assert inputs[0].case_id == "/mimic/p10/p10000001/s50000001/dicom-a.jpgprompt"
    assert inputs[0].dicom_id == "dicom-a"
    assert inputs[0].finding == "Pneumothorax"
    assert inputs[0].region_maps.shape == (2, 2, 2)


def test_build_prepared_inputs_excludes_regions_and_renormalizes_maps() -> None:
    data_rows = [
        {
            "path": "/mimic/p10/p10000001/s50000001/dicom-a.jpg",
            "label_text": "prompt",
            "category": "Pneumothorax",
        }
    ]
    base_hmaps = {
        "/mimic/p10/p10000001/s50000001/dicom-a.jpgprompt": {
            "hmap": np.ones((2, 2), dtype=np.float32)
        }
    }
    scene_graphs = {
        "dicom-a": {
            "objects": [
                {"bbox_name": "both lungs", "x1": 0, "y1": 0, "x2": 2, "y2": 2},
                {"bbox_name": "upper box", "x1": 0, "y1": 0, "x2": 2, "y2": 1},
                {"bbox_name": "lower box", "x1": 0, "y1": 1, "x2": 2, "y2": 2},
            ]
        }
    }

    inputs, report = build_prepared_inputs(
        data_rows=data_rows,
        base_hmaps=base_hmaps,
        scene_graphs=scene_graphs,
        regions=["bilateral_lungs", "upper", "lower"],
        bbox_to_region={"both lungs": "bilateral_lungs", "upper box": "upper", "lower box": "lower"},
        exclude_regions={"bilateral_lungs"},
    )

    assert report["excluded_regions"] == ["bilateral_lungs"]
    assert inputs[0].regions == ["upper", "lower"]
    assert inputs[0].region_maps.shape == (2, 2, 2)
    assert np.allclose(inputs[0].region_maps.sum(axis=0), np.ones((2, 2), dtype=np.float32))


def test_build_score_request_rows_maps_bbox_regions_to_repair_regions() -> None:
    data_rows = [
        {
            "path": "/mimic/p10/p10000001/s50000001/dicom-a.jpg",
            "label_text": "prompt",
            "category": "Pneumothorax",
        }
    ]
    scene_graphs = {
        "dicom-a": {
            "objects": [
                {
                    "bbox_name": "right upper lung zone",
                    "object_id": "obj-1",
                    "x1": 0,
                    "y1": 0,
                    "x2": 2,
                    "y2": 1,
                    "width": 2,
                    "height": 1,
                }
            ]
        }
    }

    rows = build_score_request_rows(
        data_rows=data_rows,
        scene_graphs=scene_graphs,
        bbox_to_region={"right upper lung zone": "upper"},
        findings=["Pneumothorax", "Pleural Effusion"],
    )

    assert len(rows) == 2
    assert rows[0]["subject_id"] == "10000001"
    assert rows[0]["study_id"] == "50000001"
    assert rows[0]["dicom_id"] == "dicom-a"
    assert rows[0]["region"] == "upper"
    assert rows[0]["finding"] == "Pneumothorax"
    assert rows[0]["label_source"] == "score_request"


def test_build_score_request_rows_excludes_regions() -> None:
    data_rows = [
        {
            "path": "/mimic/p10/p10000001/s50000001/dicom-a.jpg",
            "label_text": "prompt",
            "category": "Pneumothorax",
        }
    ]
    scene_graphs = {
        "dicom-a": {
            "objects": [
                {"bbox_name": "both lungs", "object_id": "obj-0", "x1": 0, "y1": 0, "x2": 2, "y2": 2},
                {"bbox_name": "right upper lung zone", "object_id": "obj-1", "x1": 0, "y1": 0, "x2": 2, "y2": 1},
            ]
        }
    }

    rows = build_score_request_rows(
        data_rows=data_rows,
        scene_graphs=scene_graphs,
        bbox_to_region={"both lungs": "bilateral_lungs", "right upper lung zone": "upper"},
        findings=["Pneumothorax"],
        exclude_regions={"bilateral_lungs"},
    )

    assert len(rows) == 1
    assert rows[0]["region"] == "upper"


def test_filter_region_space_removes_requested_regions() -> None:
    regions, mapping = filter_region_space(
        regions=["bilateral_lungs", "upper", "lower"],
        bbox_to_region={"both lungs": "bilateral_lungs", "upper box": "upper"},
        exclude_regions={"bilateral_lungs"},
    )

    assert regions == ["upper", "lower"]
    assert mapping == {"upper box": "upper"}


def test_coerce_data_rows_accepts_afloc_dict_output_and_applies_max_cases() -> None:
    data = {
        "path": ["a.jpg", "b.jpg", "c.jpg"],
        "label_text": ["pa", "pb", "pc"],
        "category": ["Pneumothorax", "Pleural Effusion", "Edema"],
    }

    rows = coerce_data_rows(data, max_cases=2)

    assert rows == [
        {"path": "a.jpg", "label_text": "pa", "category": "Pneumothorax"},
        {"path": "b.jpg", "label_text": "pb", "category": "Pleural Effusion"},
    ]


def test_load_mscxr_data_rows_uses_requested_dataset(monkeypatch) -> None:
    called = {}

    def fake_load_data(dataset):
        called["dataset"] = dataset
        return {
            "path": ["a.jpg"],
            "label_text": ["large right pneumothorax"],
            "category": ["Pneumothorax"],
        }

    fake_package = types.ModuleType("localization")
    fake_datasets = types.ModuleType("localization.datasets")
    fake_datasets.load_data = fake_load_data
    fake_package.datasets = fake_datasets
    monkeypatch.setitem(sys.modules, "localization", fake_package)
    monkeypatch.setitem(sys.modules, "localization.datasets", fake_datasets)

    rows = load_mscxr_data_rows(dataset="MS_CXR")

    assert called["dataset"] == "MS_CXR"
    assert rows == [
        {
            "path": "a.jpg",
            "label_text": "large right pneumothorax",
            "category": "Pneumothorax",
        }
    ]
