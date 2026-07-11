from pathlib import Path

import pandas as pd
import pytest

from anaprior.eval.debias_region_scores import debias_scores, load_reference_region_stats


def test_load_reference_region_stats_uses_finding_agnostic_region_mean(tmp_path: Path) -> None:
    reference = tmp_path / "agnostic_region_summary.csv"
    pd.DataFrame(
        [
            {"dataset": "imagenome_valid", "region": "hilar_mediastinal", "mean_score_agnostic": 0.7},
            {"dataset": "imagenome_valid", "region": "pleural_space_costophrenic", "mean_score_agnostic": 0.2},
        ]
    ).to_csv(reference, index=False)

    stats = load_reference_region_stats(reference)

    assert stats["hilar_mediastinal"] == 0.7
    assert stats["pleural_space_costophrenic"] == 0.2


def test_load_reference_region_stats_filters_reference_dataset(tmp_path: Path) -> None:
    reference = tmp_path / "agnostic_region_summary.csv"
    pd.DataFrame(
        [
            {"dataset": "mscxr", "region": "hilar_mediastinal", "mean_score_agnostic": 0.9},
            {"dataset": "imagenome_valid", "region": "hilar_mediastinal", "mean_score_agnostic": 0.7},
            {"dataset": "imagenome_valid", "region": "pleural_space_costophrenic", "mean_score_agnostic": 0.2},
        ]
    ).to_csv(reference, index=False)

    stats = load_reference_region_stats(reference, dataset="imagenome_valid")

    assert stats == {
        "hilar_mediastinal": 0.7,
        "pleural_space_costophrenic": 0.2,
    }


def test_load_reference_region_stats_rejects_duplicate_regions_without_dataset_filter(tmp_path: Path) -> None:
    reference = tmp_path / "agnostic_region_summary.csv"
    pd.DataFrame(
        [
            {"dataset": "mscxr", "region": "hilar_mediastinal", "mean_score_agnostic": 0.9},
            {"dataset": "imagenome_valid", "region": "hilar_mediastinal", "mean_score_agnostic": 0.7},
        ]
    ).to_csv(reference, index=False)

    with pytest.raises(ValueError, match="Duplicate region rows"):
        load_reference_region_stats(reference)


def test_debias_scores_subtracts_region_agnostic_mean_and_rescales_by_finding() -> None:
    scores = pd.DataFrame(
        [
            {
                "dicom_id": "d1",
                "finding": "Pleural Effusion",
                "region": "hilar_mediastinal",
                "score_probability": 0.9,
            },
            {
                "dicom_id": "d1",
                "finding": "Pleural Effusion",
                "region": "pleural_space_costophrenic",
                "score_probability": 0.4,
            },
            {
                "dicom_id": "d1",
                "finding": "Pneumothorax",
                "region": "hilar_mediastinal",
                "score_probability": 0.6,
            },
            {
                "dicom_id": "d1",
                "finding": "Pneumothorax",
                "region": "pleural_space_costophrenic",
                "score_probability": 0.3,
            },
        ]
    )
    reference = {
        "hilar_mediastinal": 0.7,
        "pleural_space_costophrenic": 0.2,
    }

    out = debias_scores(scores, reference, mode="subtract", rescale="finding_minmax")

    effusion = out[out["finding"] == "Pleural Effusion"].set_index("region")
    assert effusion.loc["hilar_mediastinal", "score_debiased_raw"] == pytest.approx(0.2)
    assert effusion.loc["pleural_space_costophrenic", "score_debiased_raw"] == pytest.approx(0.2)
    assert effusion.loc["hilar_mediastinal", "score_probability"] == 0.5
    assert effusion.loc["pleural_space_costophrenic", "score_probability"] == 0.5

    ptx = out[out["finding"] == "Pneumothorax"].set_index("region")
    assert ptx.loc["hilar_mediastinal", "score_debiased_raw"] < ptx.loc[
        "pleural_space_costophrenic", "score_debiased_raw"
    ]
    assert ptx.loc["pleural_space_costophrenic", "score_probability"] == 1.0
