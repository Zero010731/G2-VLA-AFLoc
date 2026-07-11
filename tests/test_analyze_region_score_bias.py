from pathlib import Path

import pandas as pd

from anaprior.eval.analyze_region_score_bias import (
    load_score_csvs,
    summarize_region_score_bias,
)


def test_summarize_region_score_bias_separates_finding_specific_and_agnostic_bias() -> None:
    scores = pd.DataFrame(
        [
            {"dataset": "mscxr", "finding": "Pleural Effusion", "region": "hilar_mediastinal", "score_probability": 0.9},
            {"dataset": "mscxr", "finding": "Pleural Effusion", "region": "pleural_space_costophrenic", "score_probability": 0.4},
            {"dataset": "mscxr", "finding": "Pneumothorax", "region": "hilar_mediastinal", "score_probability": 0.8},
            {"dataset": "mscxr", "finding": "Pneumothorax", "region": "pleural_space_costophrenic", "score_probability": 0.2},
            {"dataset": "imagenome_valid", "finding": "Pleural Effusion", "region": "hilar_mediastinal", "score_probability": 0.7},
            {"dataset": "imagenome_valid", "finding": "Pleural Effusion", "region": "pleural_space_costophrenic", "score_probability": 0.6},
        ]
    )

    tables = summarize_region_score_bias(scores, top_k=1, bias_threshold=0.05)
    finding_summary = tables["finding_region_summary"]
    agnostic = tables["agnostic_region_summary"]
    bias = tables["finding_specific_bias"]
    top_routes = tables["top_region_bias_routes"]

    effusion_hilar = finding_summary[
        (finding_summary["dataset"] == "mscxr")
        & (finding_summary["finding"] == "Pleural Effusion")
        & (finding_summary["region"] == "hilar_mediastinal")
    ].iloc[0]
    assert effusion_hilar["rank_within_finding"] == 1

    mscxr_hilar = agnostic[(agnostic["dataset"] == "mscxr") & (agnostic["region"] == "hilar_mediastinal")].iloc[0]
    assert mscxr_hilar["rank_agnostic"] == 1

    effusion_costophrenic_bias = bias[
        (bias["dataset"] == "mscxr")
        & (bias["finding"] == "Pleural Effusion")
        & (bias["region"] == "pleural_space_costophrenic")
    ].iloc[0]
    assert effusion_costophrenic_bias["bias_index"] > 0

    top_effusion_hilar = top_routes[
        (top_routes["dataset"] == "mscxr")
        & (top_routes["finding"] == "Pleural Effusion")
        & (top_routes["region"] == "hilar_mediastinal")
    ].iloc[0]
    assert top_effusion_hilar["mechanism_hint"] == "finding_agnostic_feature_or_pooling_bias"


def test_load_score_csvs_accepts_named_specs(tmp_path: Path) -> None:
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {
                "dicom_id": "d1",
                "finding": "Pneumothorax",
                "region": "hilar_mediastinal",
                "score_probability": 0.9,
            }
        ]
    ).to_csv(csv_path, index=False)

    loaded = load_score_csvs([f"mscxr={csv_path}"])

    assert loaded["dataset"].tolist() == ["mscxr"]
    assert loaded["region"].tolist() == ["hilar_mediastinal"]
