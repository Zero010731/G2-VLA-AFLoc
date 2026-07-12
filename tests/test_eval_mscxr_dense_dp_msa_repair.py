import csv
from pathlib import Path

import numpy as np
import torch

from anaprior.eval.disease_properties import DISEASE_PROPERTY_NAMES
from anaprior.eval.eval_mscxr_dense_dp_msa_repair import build_dense_dp_msa_hmaps
from anaprior.eval.phrase_subtype import PHRASE_SUBTYPE_TO_ID
from anaprior.models.dense_dp_msa_adapter import DenseDPMultiScaleSpatialAdapter


def _checkpoint(path: Path) -> None:
    model_config = {
        "num_subtypes": len(PHRASE_SUBTYPE_TO_ID),
        "num_regions": 2,
        "num_disease_properties": len(DISEASE_PROPERTY_NAMES),
        "spatial_channels": 3,
        "hidden_channels": 4,
        "condition_dim": 4,
        "lambda_weight": 0.0,
        "residual_scale": 0.20,
    }
    model = DenseDPMultiScaleSpatialAdapter(**model_config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "subtype_vocab": dict(PHRASE_SUBTYPE_TO_ID),
            "disease_property_names": DISEASE_PROPERTY_NAMES,
            "region_names": ["left_lower_lung", "right_lower_lung"],
        },
        path,
    )


def _prepared_inputs(path: Path) -> None:
    item = {
        "case_id": "case-pna",
        "dicom_id": "dicom-pna",
        "category": "Pneumonia",
        "finding": "Pneumonia",
        "phrase": "right basilar opacity compatible with pneumonia",
        "heatmap": np.array([[0.0, 1.0], [0.3, 0.8]], dtype=np.float32),
        "region_maps": np.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        "regions": ["left_lower_lung", "right_lower_lung"],
    }
    np.savez(path, items=np.array([item], dtype=object))


def _scores(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dicom_id", "region", "finding", "score_probability"])
        writer.writeheader()
        writer.writerow({"dicom_id": "dicom-pna", "region": "left_lower_lung", "finding": "Pneumonia", "score_probability": "0.2"})
        writer.writerow({"dicom_id": "dicom-pna", "region": "right_lower_lung", "finding": "Pneumonia", "score_probability": "0.9"})


def _spatial(path: Path) -> None:
    torch.save({"case_ids": ["case-pna"], "spatial_features": torch.ones(1, 3, 2, 2)}, path)


def test_build_dense_dp_msa_hmaps_returns_repaired_method(tmp_path: Path) -> None:
    ckpt = tmp_path / "dense.pt"
    prepared = tmp_path / "inputs.npz"
    scores = tmp_path / "scores.csv"
    spatial = tmp_path / "spatial.pt"
    _checkpoint(ckpt)
    _prepared_inputs(prepared)
    _scores(scores)
    _spatial(spatial)

    hmaps, stats = build_dense_dp_msa_hmaps(
        prepared_inputs_npz=prepared,
        region_score_csv=scores,
        spatial_feature_cache=spatial,
        checkpoint=ckpt,
        device="cpu",
        method_name="dense_dp_msa",
    )

    assert set(hmaps) == {"baseline", "dense_dp_msa"}
    assert hmaps["dense_dp_msa"]["case-pna"]["hmap"].shape == (2, 2)
    assert stats["dense_dp_msa"]["repaired_categories"] == {"Pneumonia": 1}
