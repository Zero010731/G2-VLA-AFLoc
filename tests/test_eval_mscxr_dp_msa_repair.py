import csv
import json
from pathlib import Path

import numpy as np
import torch

from anaprior.eval.eval_mscxr_dp_msa_repair import build_dp_msa_hmaps, main
from anaprior.eval.phrase_subtype import PHRASE_SUBTYPE_TO_ID
from anaprior.models.dp_msa_adapter import DPMultiScaleSpatialAdapter


def _checkpoint(path: Path) -> None:
    model_config = {
        "num_diseases": 2,
        "num_subtypes": len(PHRASE_SUBTYPE_TO_ID),
        "num_regions": 2,
        "hidden_channels": 4,
        "embedding_dim": 4,
        "lambda_weight": 0.1,
        "residual_scale": 0.25,
    }
    model = DPMultiScaleSpatialAdapter(**model_config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "finding_vocab": {"Pneumonia": 0, "Lung Opacity": 1},
            "subtype_vocab": dict(PHRASE_SUBTYPE_TO_ID),
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
        writer = csv.DictWriter(
            handle,
            fieldnames=["dicom_id", "region", "finding", "score_probability"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dicom_id": "dicom-pna",
                "region": "left_lower_lung",
                "finding": "Pneumonia",
                "score_probability": "0.2",
            }
        )
        writer.writerow(
            {
                "dicom_id": "dicom-pna",
                "region": "right_lower_lung",
                "finding": "Pneumonia",
                "score_probability": "0.9",
            }
        )


def _base_hmaps(path: Path) -> None:
    np.save(
        path,
        {
            "case-pna": {
                "hmap": np.array([[0.9, 0.2], [0.7, 0.1]], dtype=np.float32),
                "learned_repair": "phrase_anatomy_dcem",
            }
        },
    )


def test_build_dp_msa_hmaps_returns_baseline_and_repaired_method(tmp_path: Path) -> None:
    ckpt = tmp_path / "dp_msa_adapter.pt"
    prepared = tmp_path / "inputs.npz"
    score_csv = tmp_path / "scores.csv"
    _checkpoint(ckpt)
    _prepared_inputs(prepared)
    _scores(score_csv)

    hmaps, stats = build_dp_msa_hmaps(
        prepared_inputs_npz=prepared,
        region_score_csv=score_csv,
        checkpoint=ckpt,
        device="cpu",
    )

    assert set(hmaps) == {"baseline", "dp_msa"}
    assert "case-pna" in hmaps["dp_msa"]
    assert hmaps["dp_msa"]["case-pna"]["hmap"].shape == (2, 2)
    assert stats["dp_msa"]["repaired_categories"] == {"Pneumonia": 1}


def test_build_dp_msa_hmaps_supports_lambda_override_and_method_name(tmp_path: Path) -> None:
    ckpt = tmp_path / "dp_msa_adapter.pt"
    prepared = tmp_path / "inputs.npz"
    score_csv = tmp_path / "scores.csv"
    _checkpoint(ckpt)
    _prepared_inputs(prepared)
    _scores(score_csv)

    hmaps, stats = build_dp_msa_hmaps(
        prepared_inputs_npz=prepared,
        region_score_csv=score_csv,
        checkpoint=ckpt,
        device="cpu",
        method_name="dp_msa_lambda0p02",
        lambda_override=0.02,
    )

    assert set(hmaps) == {"baseline", "dp_msa_lambda0p02"}
    assert hmaps["dp_msa_lambda0p02"]["case-pna"]["learned_repair"] == "dp_msa_lambda0p02_repaired"
    assert stats["dp_msa_lambda0p02"]["repaired_categories"] == {"Pneumonia": 1}


def test_main_writes_dp_msa_hmap_outputs(tmp_path: Path) -> None:
    ckpt = tmp_path / "dp_msa_adapter.pt"
    prepared = tmp_path / "inputs.npz"
    score_csv = tmp_path / "scores.csv"
    outdir = tmp_path / "out"
    _checkpoint(ckpt)
    _prepared_inputs(prepared)
    _scores(score_csv)

    status = main(
        [
            "--prepared-inputs-npz",
            str(prepared),
            "--region-score-csv",
            str(score_csv),
            "--checkpoint",
            str(ckpt),
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
        ]
    )

    assert status == 0
    assert (outdir / "baseline" / "hmaps.npy").exists()
    assert (outdir / "dp_msa" / "hmaps.npy").exists()
    summary = json.loads((outdir / "dp_msa_repair_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert summary["num_cases"] == 1


def test_main_writes_named_lambda_sweep_method(tmp_path: Path) -> None:
    ckpt = tmp_path / "dp_msa_adapter.pt"
    prepared = tmp_path / "inputs.npz"
    score_csv = tmp_path / "scores.csv"
    outdir = tmp_path / "out"
    _checkpoint(ckpt)
    _prepared_inputs(prepared)
    _scores(score_csv)

    status = main(
        [
            "--prepared-inputs-npz",
            str(prepared),
            "--region-score-csv",
            str(score_csv),
            "--checkpoint",
            str(ckpt),
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
            "--method-name",
            "dp_msa_lambda0p05",
            "--lambda-override",
            "0.05",
        ]
    )

    assert status == 0
    assert (outdir / "dp_msa_lambda0p05" / "hmaps.npy").exists()
    summary = json.loads((outdir / "dp_msa_repair_summary.json").read_text(encoding="utf-8"))
    assert summary["method_name"] == "dp_msa_lambda0p05"
    assert summary["lambda_override"] == 0.05


def test_build_dp_msa_hmaps_uses_v3_base_for_refinement_without_overwriting_baseline(tmp_path: Path) -> None:
    ckpt = tmp_path / "dp_msa_adapter.pt"
    prepared = tmp_path / "inputs.npz"
    score_csv = tmp_path / "scores.csv"
    base_hmaps = tmp_path / "phrase_anatomy_hmaps.npy"
    _checkpoint(ckpt)
    _prepared_inputs(prepared)
    _scores(score_csv)
    _base_hmaps(base_hmaps)

    hmaps, stats = build_dp_msa_hmaps(
        prepared_inputs_npz=prepared,
        region_score_csv=score_csv,
        checkpoint=ckpt,
        device="cpu",
        method_name="dp_msa_v2",
        lambda_override=0.0,
        base_hmaps_npy=base_hmaps,
        base_method_name="phrase_anatomy_dcem",
    )

    assert set(hmaps) == {"baseline", "dp_msa_v2"}
    assert np.allclose(hmaps["baseline"]["case-pna"]["hmap"], np.array([[0.0, 1.0], [0.3, 0.8]]))
    assert np.allclose(
        hmaps["dp_msa_v2"]["case-pna"]["hmap"],
        np.array([[1.0, 0.125], [0.75, 0.0]], dtype=np.float32),
    )
    assert stats["dp_msa_v2"]["base_method_name"] == {"phrase_anatomy_dcem": 1}
