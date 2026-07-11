import csv
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from anaprior.models.region_abnormality_predictor import RegionAbnormalityPredictor
from anaprior.eval.predict_region_scores import export_region_scores


def write_checkpoint(path: Path) -> None:
    model = RegionAbnormalityPredictor(
        feature_dim=2,
        num_findings=2,
        finding_embedding_dim=2,
        hidden_dim=4,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias.fill_(2.0)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "feature_dim": 2,
                "num_findings": 2,
                "finding_embedding_dim": 2,
                "hidden_dim": 4,
                "dropout": 0.0,
            },
            "finding_vocab": {"Pleural Effusion": 0, "Pneumothorax": 1},
        },
        path,
    )


def write_cache(path: Path) -> None:
    torch.save(
        {
            "region_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            "labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "finding_ids": torch.tensor([1, 0], dtype=torch.long),
            "valid_mask": torch.tensor([True, False]),
            "finding_vocab": {"Pleural Effusion": 0, "Pneumothorax": 1},
            "metadata": [
                {
                    "dicom_id": "dicom-a",
                    "subject_id": "10000001",
                    "study_id": "50000001",
                    "region": "right upper lung zone",
                    "finding": "Pneumothorax",
                },
                {
                    "dicom_id": "dicom-a",
                    "subject_id": "10000001",
                    "study_id": "50000001",
                    "region": "left costophrenic angle",
                    "finding": "Pleural Effusion",
                },
            ],
        },
        path,
    )


def test_export_region_scores_writes_valid_rows_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "region_predictor.pt"
    cache = tmp_path / "features.pt"
    out_csv = tmp_path / "scores.csv"
    write_checkpoint(checkpoint)
    write_cache(cache)

    report = export_region_scores(
        checkpoint_path=checkpoint,
        cache_path=cache,
        output_csv=out_csv,
        device="cpu",
        valid_only=True,
    )

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert report["rows_written"] == 1
    assert rows[0]["dicom_id"] == "dicom-a"
    assert rows[0]["finding"] == "Pneumothorax"
    assert rows[0]["region"] == "right upper lung zone"
    assert float(rows[0]["score_probability"]) > 0.8
    assert (tmp_path / "scores.report.json").exists()


def test_export_region_scores_remaps_cache_vocab_and_skips_unsupported_findings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "region_predictor.pt"
    cache = tmp_path / "features.pt"
    out_csv = tmp_path / "scores.csv"
    write_checkpoint(checkpoint)
    torch.save(
        {
            "region_features": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
            "finding_ids": torch.tensor([1, 2, 0], dtype=torch.long),
            "valid_mask": torch.tensor([True, True, True]),
            "finding_vocab": {
                "Atelectasis": 0,
                "Pleural Effusion": 1,
                "Pneumothorax": 2,
            },
            "metadata": [
                {"dicom_id": "dicom-a", "region": "left costophrenic angle", "finding": "Pleural Effusion"},
                {"dicom_id": "dicom-b", "region": "right upper lung zone", "finding": "Pneumothorax"},
                {"dicom_id": "dicom-c", "region": "left lower lung zone", "finding": "Atelectasis"},
            ],
        },
        cache,
    )

    report = export_region_scores(
        checkpoint_path=checkpoint,
        cache_path=cache,
        output_csv=out_csv,
        device="cpu",
        valid_only=True,
    )

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert [row["finding"] for row in rows] == ["Pleural Effusion", "Pneumothorax"]
    assert [row["finding_id"] for row in rows] == ["0", "1"]
    assert report["rows_written"] == 2
    assert report["skipped_unsupported"] == 1
    assert report["unsupported_findings"] == {"Atelectasis": 1}
