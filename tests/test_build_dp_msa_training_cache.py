import csv
import json
from pathlib import Path

import numpy as np
import torch

from anaprior.eval.disease_properties import DISEASE_PROPERTY_NAMES, disease_property_vector
from anaprior.eval.phrase_subtype import PHRASE_SUBTYPE_TO_ID
from anaprior.train.build_dp_msa_training_cache import build_dp_msa_training_cache, main
from anaprior.train.train_dp_msa_adapter import REQUIRED_CACHE_FIELDS


def _write_prepared_inputs(path: Path) -> None:
    items = []
    for idx, category in enumerate(["Pneumonia", "Lung Opacity", "Pneumonia", "Lung Opacity"]):
        phrase = "right basilar opacity compatible with pneumonia" if category == "Pneumonia" else "patchy lung opacity"
        items.append(
            {
                "case_id": f"case-{idx}",
                "dicom_id": f"dicom-{idx}",
                "category": category,
                "finding": category,
                "phrase": phrase,
                "heatmap": np.array([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32) + idx * 0.01,
                "region_maps": np.array(
                    [
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 1.0], [0.0, 1.0]],
                    ],
                    dtype=np.float32,
                ),
                "regions": ["left_lower_lung", "right_lower_lung"],
            }
        )
    np.savez(path, items=np.array(items, dtype=object))


def _write_scores(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dicom_id", "region", "finding", "score_probability"],
        )
        writer.writeheader()
        for idx, category in enumerate(["Pneumonia", "Lung Opacity", "Pneumonia", "Lung Opacity"]):
            writer.writerow(
                {
                    "dicom_id": f"dicom-{idx}",
                    "region": "left_lower_lung",
                    "finding": category,
                    "score_probability": "0.2",
                }
            )
            writer.writerow(
                {
                    "dicom_id": f"dicom-{idx}",
                    "region": "right_lower_lung",
                    "finding": category,
                    "score_probability": "0.8",
                }
            )


def _write_base_hmaps(path: Path) -> None:
    hmaps = {}
    for idx in range(4):
        hmaps[f"case-{idx}"] = {
            "hmap": np.array([[0.9, 0.1], [0.7, 0.2]], dtype=np.float32) - idx * 0.01,
            "learned_repair": "phrase_anatomy_dcem",
        }
    np.save(path, hmaps)


def _write_spatial_feature_cache(path: Path) -> None:
    torch.save(
        {
            "case_ids": [f"case-{idx}" for idx in range(4)],
            "spatial_features": torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).view(4, 3, 2, 2),
        },
        path,
    )


def test_build_dp_msa_training_cache_writes_train_and_valid_pt(tmp_path: Path) -> None:
    prepared = tmp_path / "inputs.npz"
    scores = tmp_path / "scores.csv"
    outdir = tmp_path / "cache"
    _write_prepared_inputs(prepared)
    _write_scores(scores)

    report = build_dp_msa_training_cache(
        prepared_inputs_npz=prepared,
        region_score_csv=scores,
        outdir=outdir,
        findings=["Pneumonia", "Lung Opacity"],
        valid_fraction=0.5,
        seed=11,
        min_score_sum=0.0,
    )

    assert report["status"] == "ok"
    assert report["target_mode"] == "region_score_weighted"
    assert report["uses_mscxr_boxes"] is False
    train = torch.load(report["train_cache"], map_location="cpu", weights_only=False)
    valid = torch.load(report["valid_cache"], map_location="cpu", weights_only=False)
    assert REQUIRED_CACHE_FIELDS <= set(train)
    assert REQUIRED_CACHE_FIELDS <= set(valid)
    assert train["finding_vocab"] == {"Pneumonia": 0, "Lung Opacity": 1}
    assert train["subtype_vocab"] == dict(PHRASE_SUBTYPE_TO_ID)
    assert train["disease_property_names"] == DISEASE_PROPERTY_NAMES
    assert train["disease_properties"].shape[1] == len(DISEASE_PROPERTY_NAMES)
    assert train["region_names"] == ["left_lower_lung", "right_lower_lung"]
    assert train["base_hmaps"].shape[1:] == (1, 2, 2)
    assert valid["target_hmaps"].shape[1:] == (1, 2, 2)
    assert int(train["base_hmaps"].shape[0] + valid["base_hmaps"].shape[0]) == 4
    assert float(train["target_hmaps"].min()) >= 0.0
    assert float(valid["target_hmaps"].max()) <= 1.0


def test_build_dp_msa_training_cache_uses_v3_base_and_mixed_target(tmp_path: Path) -> None:
    prepared = tmp_path / "inputs.npz"
    scores = tmp_path / "scores.csv"
    base_hmaps = tmp_path / "phrase_anatomy_hmaps.npy"
    outdir = tmp_path / "cache"
    _write_prepared_inputs(prepared)
    _write_scores(scores)
    _write_base_hmaps(base_hmaps)

    report = build_dp_msa_training_cache(
        prepared_inputs_npz=prepared,
        region_score_csv=scores,
        outdir=outdir,
        findings=["Pneumonia", "Lung Opacity"],
        valid_fraction=0.5,
        seed=11,
        min_score_sum=0.0,
        base_hmaps_npy=base_hmaps,
        base_method_name="phrase_anatomy_dcem",
        target_mix_beta=0.10,
        disease_betas={"Pneumonia": 0.05},
    )

    assert report["base_method_name"] == "phrase_anatomy_dcem"
    assert report["target_mode"] == "mixed_base_region_score"
    assert report["target_mix_beta"] == 0.10
    assert report["disease_betas"] == {"Pneumonia": 0.05}
    train = torch.load(report["train_cache"], map_location="cpu", weights_only=False)
    valid = torch.load(report["valid_cache"], map_location="cpu", weights_only=False)
    payload = train if "case-0" in train["case_ids"] else valid
    row = payload["case_ids"].index("case-0")
    expected_base = torch.tensor([[[0.9, 0.1], [0.7, 0.2]]], dtype=torch.float32)
    assert torch.allclose(payload["base_hmaps"][row], expected_base)
    assert payload["target_mode"] == "mixed_base_region_score"
    assert payload["base_method_name"] == "phrase_anatomy_dcem"
    assert torch.allclose(
        payload["disease_properties"][row],
        torch.tensor(disease_property_vector("Pneumonia"), dtype=torch.float32),
    )
    assert payload["target_betas"][row].item() == torch.tensor(0.05).item()
    assert not torch.allclose(payload["target_hmaps"][row], payload["base_hmaps"][row])


def test_build_dp_msa_training_cache_attaches_spatial_features(tmp_path: Path) -> None:
    prepared = tmp_path / "inputs.npz"
    scores = tmp_path / "scores.csv"
    spatial = tmp_path / "spatial.pt"
    outdir = tmp_path / "cache"
    _write_prepared_inputs(prepared)
    _write_scores(scores)
    _write_spatial_feature_cache(spatial)

    report = build_dp_msa_training_cache(
        prepared_inputs_npz=prepared,
        region_score_csv=scores,
        outdir=outdir,
        findings=["Pneumonia", "Lung Opacity"],
        valid_fraction=0.5,
        seed=11,
        min_score_sum=0.0,
        spatial_feature_cache=spatial,
    )

    train = torch.load(report["train_cache"], map_location="cpu", weights_only=False)
    valid = torch.load(report["valid_cache"], map_location="cpu", weights_only=False)
    assert "spatial_features" in train
    assert "spatial_features" in valid
    assert train["spatial_features"].shape[1:] == (3, 2, 2)
    assert report["spatial_feature_cache"] == str(spatial)


def test_main_writes_dp_msa_cache_report(tmp_path: Path) -> None:
    prepared = tmp_path / "inputs.npz"
    scores = tmp_path / "scores.csv"
    outdir = tmp_path / "cache"
    _write_prepared_inputs(prepared)
    _write_scores(scores)

    status = main(
        [
            "--prepared-inputs-npz",
            str(prepared),
            "--region-score-csv",
            str(scores),
            "--outdir",
            str(outdir),
            "--findings",
            "Pneumonia,Lung Opacity",
            "--valid-fraction",
            "0.5",
            "--seed",
            "11",
            "--min-score-sum",
            "0.0",
        ]
    )

    assert status == 0
    report = json.loads((outdir / "dp_msa_cache_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert Path(report["train_cache"]).exists()
    assert Path(report["valid_cache"]).exists()
