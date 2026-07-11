import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from anaprior.eval.eval_mscxr_learned_repair import (
    LearnedRepairInput,
    _load_prepared_inputs,
    build_method_hmaps,
    load_region_score_table,
    region_scores_for_case,
)


def write_score_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "dicom_id": "dicom-a",
            "region": "upper",
            "finding": "Pneumothorax",
            "score_probability": "0.9",
        },
        {
            "dicom_id": "dicom-a",
            "region": "lower",
            "finding": "Pneumothorax",
            "score_probability": "0.1",
        },
        {
            "dicom_id": "dicom-a",
            "region": "lower",
            "finding": "Pneumothorax",
            "score_probability": "0.4",
        },
        {
            "dicom_id": "dicom-a",
            "region": "right_upper_lung",
            "finding": "Pneumothorax",
            "score_probability": "0.35",
        },
        {
            "dicom_id": "dicom-a",
            "region": "cardiac_silhouette",
            "finding": "Pneumothorax",
            "score_probability": "0.9",
        },
        {
            "dicom_id": "dicom-b",
            "region": "upper",
            "finding": "Edema",
            "score_probability": "0.8",
        },
        {
            "dicom_id": "dicom-b",
            "region": "lower",
            "finding": "Edema",
            "score_probability": "0.2",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_region_score_table_and_case_vector(tmp_path: Path) -> None:
    score_csv = tmp_path / "scores.csv"
    write_score_csv(score_csv)

    table = load_region_score_table(score_csv)
    scores = region_scores_for_case(
        table,
        dicom_id="dicom-a",
        finding="Pneumothorax",
        regions=["upper", "lower", "missing"],
    )

    assert np.allclose(scores, np.array([0.9, 0.4, 0.0], dtype=np.float32))


def test_build_method_hmaps_runs_pre_registered_comparisons(tmp_path: Path) -> None:
    score_csv = tmp_path / "scores.csv"
    write_score_csv(score_csv)
    table = load_region_score_table(score_csv)
    region_maps = np.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    inputs = [
        LearnedRepairInput(
            case_id="case-a",
            dicom_id="dicom-a",
            category="Pneumothorax",
            finding="Pneumothorax",
            phrase="right apical pneumothorax",
            heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            region_maps=region_maps,
            regions=["right_upper_lung", "cardiac_silhouette"],
        ),
        LearnedRepairInput(
            case_id="case-b",
            dicom_id="dicom-b",
            category="Edema",
            finding="Edema",
            phrase="bilateral diffuse edema",
            heatmap=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            region_maps=region_maps,
            regions=["upper", "lower"],
        ),
    ]

    hmaps, stats = build_method_hmaps(
        inputs=inputs,
        region_score_table=table,
        candidate_categories={"Pneumothorax"},
        alpha=1.0,
        seed=3,
    )

    assert set(hmaps) == {
        "baseline",
        "learned_selective",
        "disease_gated_learned",
        "disease_pooled_learned",
        "phrase_anatomy_dcem",
        "all_class_learned",
        "candidate_shuffled",
        "candidate_uniform",
    }
    assert np.allclose(hmaps["baseline"]["case-a"]["hmap"], inputs[0].heatmap)
    assert hmaps["learned_selective"]["case-a"]["hmap"][1, 0] > hmaps["learned_selective"]["case-a"]["hmap"][0, 0]
    assert np.allclose(hmaps["disease_gated_learned"]["case-a"]["hmap"], inputs[0].heatmap)
    assert hmaps["phrase_anatomy_dcem"]["case-a"]["hmap"][0, 0] > hmaps["phrase_anatomy_dcem"]["case-a"]["hmap"][1, 0]
    assert np.allclose(hmaps["learned_selective"]["case-b"]["hmap"], inputs[1].heatmap)
    assert hmaps["all_class_learned"]["case-b"]["hmap"][0, 0] > hmaps["all_class_learned"]["case-b"]["hmap"][1, 0]
    assert stats["learned_selective"]["repaired_categories"]["Pneumothorax"] == 1
    assert stats["learned_selective"]["skipped_non_candidate"]["Edema"] == 1
    assert stats["disease_gated_learned"]["gate_bypassed_categories"]["Pneumothorax"] == 1
    assert stats["disease_pooled_learned"]["gate_passed_categories"]["Pneumothorax"] == 1
    assert stats["phrase_anatomy_dcem"]["gate_passed_categories"]["Pneumothorax"] == 1


def test_load_prepared_inputs_accepts_legacy_payload_without_phrase(tmp_path: Path) -> None:
    path = tmp_path / "legacy_inputs.npz"
    np.savez_compressed(
        path,
        items=np.asarray(
            [
                {
                    "case_id": "case-legacy",
                    "dicom_id": "dicom-legacy",
                    "category": "Pneumonia",
                    "finding": "Pneumonia",
                    "heatmap": np.zeros((2, 2), dtype=np.float32),
                    "region_maps": np.zeros((1, 2, 2), dtype=np.float32),
                    "regions": ["right_lower_lung"],
                }
            ],
            dtype=object,
        ),
    )

    inputs = _load_prepared_inputs(path)

    assert len(inputs) == 1
    assert inputs[0].phrase == "Pneumonia"
