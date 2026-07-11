import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from anaprior.train.train_region_predictor import (
    TrainConfig,
    _load_cache,
    _loss_on_tensors,
    build_hard_negative_pairs,
    hard_negative_ranking_loss,
    train_region_predictor,
)


def write_cache(path: Path, *, offset: float = 0.0) -> None:
    features = []
    labels = []
    finding_ids = []
    for finding_id in [0, 1]:
        for idx in range(12):
            label = float(idx % 2)
            sign = 1.0 if label else -1.0
            features.append(torch.tensor([sign + offset, float(finding_id), sign * 0.5]))
            labels.append(label)
            finding_ids.append(finding_id)
    torch.save(
        {
            "region_features": torch.stack(features),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "finding_ids": torch.tensor(finding_ids, dtype=torch.long),
            "valid_mask": torch.ones(len(labels), dtype=torch.bool),
            "finding_vocab": {"Pleural Effusion": 0, "Pneumothorax": 1},
            "metadata": [{"dicom_id": f"case-{idx}"} for idx in range(len(labels))],
        },
        path,
    )


def test_train_region_predictor_writes_checkpoint_and_loss_report(tmp_path: Path) -> None:
    train_cache = tmp_path / "train.pt"
    valid_cache = tmp_path / "valid.pt"
    outdir = tmp_path / "trained"
    write_cache(train_cache)
    write_cache(valid_cache, offset=0.1)

    report = train_region_predictor(
        TrainConfig(
            train_cache=train_cache,
            valid_cache=valid_cache,
            output_dir=outdir,
            epochs=12,
            batch_size=8,
            learning_rate=0.05,
            hidden_dim=8,
            finding_embedding_dim=4,
            seed=7,
        )
    )

    checkpoint_path = outdir / "region_predictor.pt"
    report_path = outdir / "train_report.json"
    assert checkpoint_path.exists()
    assert report_path.exists()
    assert report["final_train_loss"] < report["initial_train_loss"]
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["finding_vocab"] == {"Pleural Effusion": 0, "Pneumothorax": 1}
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_config"]["feature_dim"] == 3
    assert checkpoint["model_config"]["num_findings"] == 2


def test_train_region_predictor_prints_epoch_progress(tmp_path: Path, capsys) -> None:
    train_cache = tmp_path / "train.pt"
    valid_cache = tmp_path / "valid.pt"
    outdir = tmp_path / "trained"
    write_cache(train_cache)
    write_cache(valid_cache, offset=0.1)

    train_region_predictor(
        TrainConfig(
            train_cache=train_cache,
            valid_cache=valid_cache,
            output_dir=outdir,
            epochs=2,
            batch_size=8,
            learning_rate=0.05,
            hidden_dim=8,
            finding_embedding_dim=4,
            seed=7,
            log_every=1,
        )
    )

    out = capsys.readouterr().out
    assert "[train_region_predictor] epoch 1/2" in out
    assert "[train_region_predictor] epoch 2/2" in out


def test_loss_on_tensors_uses_eval_batches() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def eval(self) -> None:
            return None

        def score_pairs(self, features: torch.Tensor, finding_ids: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(int(features.shape[0]))
            return torch.zeros(features.shape[0], dtype=torch.float32)

        def loss(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            pos_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=pos_weight,
            )

    model = RecordingModel()
    features = torch.randn(10, 3)
    labels = torch.tensor([0, 1] * 5, dtype=torch.float32)
    finding_ids = torch.arange(10, dtype=torch.long) % 2

    loss = _loss_on_tensors(model, features, labels, finding_ids, eval_batch_size=4)

    assert round(loss, 6) == round(float(torch.log(torch.tensor(2.0))), 6)
    assert model.batch_sizes == [4, 4, 2]


def test_load_cache_keeps_feature_cache_on_cpu(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_load(path: Path, map_location: str, weights_only: bool) -> dict[str, torch.Tensor | dict[str, int]]:
        calls.append({"path": path, "map_location": map_location, "weights_only": weights_only})
        return {
            "region_features": torch.zeros(1, 2),
            "labels": torch.zeros(1),
            "finding_ids": torch.zeros(1, dtype=torch.long),
            "valid_mask": torch.ones(1, dtype=torch.bool),
            "finding_vocab": {"Atelectasis": 0},
        }

    monkeypatch.setattr(torch, "load", fake_load)

    _load_cache(tmp_path / "cache.pt", device="cuda")

    assert calls == [
        {"path": tmp_path / "cache.pt", "map_location": "cpu", "weights_only": False}
    ]


def test_build_hard_negative_pairs_uses_positive_prior_and_hard_negative_regions() -> None:
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    finding_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    region_names = [
        "left_upper_lung",
        "cardiac_silhouette",
        "cardiac_silhouette",
        "left_lower_lung",
    ]
    finding_vocab = {"Pneumothorax": 0, "Cardiomegaly": 1}

    pairs = build_hard_negative_pairs(
        labels=labels,
        finding_ids=finding_ids,
        region_names=region_names,
        finding_vocab=finding_vocab,
        max_pairs_per_finding=8,
    )

    assert pairs == [(0, 1), (2, 3)]


def test_build_hard_negative_pairs_accepts_chest_imagenome_raw_region_names() -> None:
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    finding_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    region_names = [
        "right upper lung zone",
        "cardiac silhouette",
        "cardiac silhouette",
        "right lung",
    ]
    finding_vocab = {"Pneumothorax": 0, "Cardiomegaly": 1}

    pairs = build_hard_negative_pairs(
        labels=labels,
        finding_ids=finding_ids,
        region_names=region_names,
        finding_vocab=finding_vocab,
        max_pairs_per_finding=8,
    )

    assert pairs == [(0, 1), (2, 3)]


def test_hard_negative_ranking_loss_penalizes_when_negative_scores_above_positive() -> None:
    class PairScoreModel:
        def score_pairs(self, features: torch.Tensor, finding_ids: torch.Tensor) -> torch.Tensor:
            return features[:, 0]

    features = torch.tensor([[0.1], [0.9]], dtype=torch.float32)
    finding_ids = torch.tensor([0, 0], dtype=torch.long)

    loss = hard_negative_ranking_loss(
        model=PairScoreModel(),
        features=features,
        finding_ids=finding_ids,
        pairs=[(0, 1)],
        margin=0.2,
    )

    assert torch.isclose(loss, torch.tensor(1.0))


def test_train_region_predictor_reports_rank_pairs_when_enabled(tmp_path: Path) -> None:
    train_cache = tmp_path / "train.pt"
    valid_cache = tmp_path / "valid.pt"
    outdir = tmp_path / "trained"
    write_cache(train_cache)
    write_cache(valid_cache, offset=0.1)

    payload = torch.load(train_cache, weights_only=False)
    payload["finding_vocab"] = {"Pneumothorax": 0, "Cardiomegaly": 1}
    payload["region_names"] = [
        "left_upper_lung",
        "cardiac_silhouette",
    ] * 12
    torch.save(payload, train_cache)
    payload = torch.load(valid_cache, weights_only=False)
    payload["finding_vocab"] = {"Pneumothorax": 0, "Cardiomegaly": 1}
    payload["region_names"] = [
        "left_upper_lung",
        "cardiac_silhouette",
    ] * 12
    torch.save(payload, valid_cache)

    report = train_region_predictor(
        TrainConfig(
            train_cache=train_cache,
            valid_cache=valid_cache,
            output_dir=outdir,
            epochs=1,
            batch_size=8,
            learning_rate=0.01,
            hidden_dim=8,
            finding_embedding_dim=4,
            rank_loss_weight=0.1,
            max_rank_pairs_per_finding=4,
        )
    )

    assert report["rank_loss_weight"] == 0.1
    assert report["num_rank_pairs"] > 0
    checkpoint = torch.load(outdir / "region_predictor.pt", weights_only=False)
    assert checkpoint["train_config"]["rank_loss_weight"] == 0.1
