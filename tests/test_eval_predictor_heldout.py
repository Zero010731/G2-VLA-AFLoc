import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from anaprior.eval.eval_predictor_heldout import (
    _score_pairs_in_batches,
    binary_average_precision,
    binary_roc_auc,
    best_binary_f1,
    evaluate_logits_by_finding,
    evaluate_region_predictor,
)


def test_binary_metrics_reward_perfect_ranking() -> None:
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    assert binary_roc_auc(scores, labels) == 1.0
    assert binary_average_precision(scores, labels) == 1.0
    assert best_binary_f1(scores, labels)["f1"] == 1.0


def test_evaluate_logits_by_finding_reports_overall_and_per_finding(tmp_path: Path) -> None:
    logits = torch.tensor([3.0, 2.0, -2.0, -3.0, 2.0, -2.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    finding_ids = torch.tensor([0, 0, 0, 0, 1, 1])
    finding_vocab = {"Pleural Effusion": 0, "Pneumothorax": 1}

    report = evaluate_logits_by_finding(
        logits=logits,
        labels=labels,
        finding_ids=finding_ids,
        finding_vocab=finding_vocab,
        output_dir=tmp_path,
    )

    assert report["overall"]["auroc"] == 1.0
    assert report["overall"]["average_precision"] == 1.0
    assert {row["finding"] for row in report["per_finding"]} == {
        "Pleural Effusion",
        "Pneumothorax",
    }
    assert (tmp_path / "predictor_heldout_metrics.json").exists()
    assert (tmp_path / "predictor_heldout_metrics.csv").exists()


def test_evaluate_logits_by_finding_exports_probability_distribution(tmp_path: Path) -> None:
    logits = torch.tensor([4.0, 1.0, -2.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    finding_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    finding_vocab = {"Pleural Effusion": 0, "Pneumothorax": 1}

    report = evaluate_logits_by_finding(
        logits=logits,
        labels=labels,
        finding_ids=finding_ids,
        finding_vocab=finding_vocab,
        output_dir=tmp_path,
    )

    rows = report["probability_summary"]
    pneumothorax_positive = [
        row for row in rows if row["finding"] == "Pneumothorax" and row["label_group"] == "positive"
    ][0]
    assert pneumothorax_positive["n"] == 1
    assert pneumothorax_positive["probability_q50"] < 0.1
    assert pneumothorax_positive["fraction_below_0_2"] == 1.0
    assert (tmp_path / "predictor_heldout_probability_summary.csv").exists()


def test_score_pairs_in_batches_chunks_model_calls() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def score_pairs(self, features: torch.Tensor, finding_ids: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(int(features.shape[0]))
            return features[:, 0] + finding_ids.float()

    model = RecordingModel()
    features = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    finding_ids = torch.arange(7, dtype=torch.long)

    logits = _score_pairs_in_batches(model, features, finding_ids, batch_size=3)

    assert model.batch_sizes == [3, 3, 1]
    assert torch.equal(logits, features[:, 0] + finding_ids.float())


def test_evaluate_region_predictor_loads_cache_on_cpu(monkeypatch, tmp_path: Path) -> None:
    class FakeModel:
        def score_pairs(self, features: torch.Tensor, finding_ids: torch.Tensor) -> torch.Tensor:
            return torch.ones(features.shape[0], dtype=torch.float32)

    def fake_load_predictor_from_checkpoint(path: Path, device: str):
        return FakeModel(), {"finding_vocab": {"Atelectasis": 0}}

    calls = []

    def fake_load(path: Path, map_location: str, weights_only: bool):
        calls.append({"path": path, "map_location": map_location, "weights_only": weights_only})
        return {
            "region_features": torch.zeros(2, 3),
            "finding_ids": torch.zeros(2, dtype=torch.long),
            "labels": torch.tensor([0.0, 1.0]),
            "valid_mask": torch.ones(2, dtype=torch.bool),
        }

    monkeypatch.setattr(
        "anaprior.eval.eval_predictor_heldout.load_predictor_from_checkpoint",
        fake_load_predictor_from_checkpoint,
    )
    monkeypatch.setattr(torch, "load", fake_load)

    evaluate_region_predictor(
        checkpoint_path=tmp_path / "region_predictor.pt",
        cache_path=tmp_path / "valid.pt",
        output_dir=tmp_path / "metrics",
        device="cuda",
        batch_size=1,
    )

    assert calls == [
        {"path": tmp_path / "valid.pt", "map_location": "cpu", "weights_only": False}
    ]
