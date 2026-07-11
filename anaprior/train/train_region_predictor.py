"""Train the learned region abnormality predictor."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from anaprior.eval.eval_predictor_heldout import evaluate_region_predictor
from anaprior.models.region_abnormality_predictor import RegionAbnormalityPredictor
from anaprior.train.dcem_v2_hard_negatives import (
    hard_negative_regions_for,
    normalize_region_name,
    positive_prior_regions_for,
)


@dataclass(frozen=True)
class TrainConfig:
    train_cache: Path
    valid_cache: Path
    output_dir: Path
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    finding_embedding_dim: int = 64
    dropout: float = 0.0
    seed: int = 13
    device: str = "cpu"
    use_pos_weight: bool = True
    log_every: int = 1
    eval_batch_size: int | None = None
    rank_loss_weight: float = 0.0
    rank_margin: float = 0.2
    max_rank_pairs_per_finding: int = 4096


def _load_cache(path: Path, device: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"region_features", "labels", "finding_ids", "valid_mask", "finding_vocab"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} is missing cache fields: {missing}")
    return payload


def _valid_payload_tensors(payload: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = payload["region_features"].cpu().float()
    labels = payload["labels"].cpu().float()
    finding_ids = payload["finding_ids"].cpu().long()
    valid_mask = payload["valid_mask"].cpu().bool()
    if features.ndim != 2:
        raise ValueError("region_features must have shape [N,C]")
    if labels.ndim != 1 or finding_ids.ndim != 1 or valid_mask.ndim != 1:
        raise ValueError("labels, finding_ids, and valid_mask must be flat tensors")
    if not (features.shape[0] == labels.shape[0] == finding_ids.shape[0] == valid_mask.shape[0]):
        raise ValueError("cache tensors must have matching row counts")
    return features[valid_mask], labels[valid_mask], finding_ids[valid_mask]


def _valid_region_names(payload: dict[str, Any]) -> list[str]:
    valid_mask = payload["valid_mask"].cpu().bool()
    region_names = payload.get("region_names")
    if region_names is None:
        return []
    if len(region_names) != int(valid_mask.numel()):
        raise ValueError("region_names must match cache row count")
    return [str(region) for region, keep in zip(region_names, valid_mask.tolist()) if keep]


def _make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    finding_ids: torch.Tensor,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dataset = TensorDataset(features.cpu(), labels.cpu(), finding_ids.cpu())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def _loss_on_tensors(
    model: RegionAbnormalityPredictor,
    features: torch.Tensor,
    labels: torch.Tensor,
    finding_ids: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    eval_batch_size: int | None = None,
) -> float:
    if eval_batch_size is not None and eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    model.eval()
    batch_size = eval_batch_size or int(labels.numel())
    weighted_loss = 0.0
    total = 0
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = features.device
    with torch.no_grad():
        for start in range(0, int(labels.numel()), batch_size):
            end = min(start + batch_size, int(labels.numel()))
            batch_labels = labels[start:end].to(device)
            logits = model.score_pairs(features[start:end].to(device), finding_ids[start:end].to(device))
            loss = model.loss(logits, batch_labels, pos_weight=pos_weight)
            weighted_loss += float(loss.item()) * int(batch_labels.numel())
            total += int(batch_labels.numel())
    return weighted_loss / max(total, 1)


def _positive_weight(labels: torch.Tensor, device: str) -> torch.Tensor | None:
    positives = float((labels == 1).sum().item())
    negatives = float((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return None
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def _normalized_region_names(region_names: list[str]) -> list[str]:
    return [normalize_region_name(region) for region in region_names]


def build_hard_negative_pairs(
    labels: torch.Tensor,
    finding_ids: torch.Tensor,
    region_names: list[str],
    finding_vocab: dict[str, int],
    max_pairs_per_finding: int = 4096,
) -> list[tuple[int, int]]:
    """Build positive-vs-hard-negative training pairs for DCEM-v2-B.

    Each pair is `(positive_index, hard_negative_index)`.  Pairs are formed only
    within the same finding id and only from conservative anatomy priors.
    """

    if max_pairs_per_finding <= 0:
        return []
    if labels.ndim != 1 or finding_ids.ndim != 1:
        raise ValueError("labels and finding_ids must be flat tensors")
    if labels.shape[0] != finding_ids.shape[0]:
        raise ValueError("labels and finding_ids must have matching row counts")
    if len(region_names) != int(labels.shape[0]):
        raise ValueError("region_names must match labels length")

    normalized_regions = _normalized_region_names(region_names)
    id_to_finding = {int(idx): str(finding) for finding, idx in finding_vocab.items()}
    pairs: list[tuple[int, int]] = []
    labels_cpu = labels.cpu().float()
    finding_ids_cpu = finding_ids.cpu().long()
    for finding_id in sorted(set(int(value) for value in finding_ids_cpu.tolist())):
        finding = id_to_finding.get(finding_id)
        if finding is None:
            continue
        positive_priors = positive_prior_regions_for(finding)
        hard_negatives = hard_negative_regions_for(finding)
        if not positive_priors or not hard_negatives:
            continue
        positive_indices = [
            idx
            for idx, (label, fid, region) in enumerate(
                zip(labels_cpu.tolist(), finding_ids_cpu.tolist(), normalized_regions)
            )
            if int(fid) == finding_id and float(label) > 0.5 and region in positive_priors
        ]
        negative_indices = [
            idx
            for idx, (label, fid, region) in enumerate(
                zip(labels_cpu.tolist(), finding_ids_cpu.tolist(), normalized_regions)
            )
            if int(fid) == finding_id and float(label) <= 0.5 and region in hard_negatives
        ]
        if not positive_indices or not negative_indices:
            continue
        count = 0
        for pos_idx in positive_indices:
            for neg_idx in negative_indices:
                pairs.append((int(pos_idx), int(neg_idx)))
                count += 1
                if count >= int(max_pairs_per_finding):
                    break
            if count >= int(max_pairs_per_finding):
                break
    return pairs


def hard_negative_ranking_loss(
    model: RegionAbnormalityPredictor,
    features: torch.Tensor,
    finding_ids: torch.Tensor,
    pairs: list[tuple[int, int]] | torch.Tensor,
    margin: float,
    batch_size: int | None = None,
) -> torch.Tensor:
    """Margin loss that enforces positive prior regions above hard negatives."""

    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = features.device
    if isinstance(pairs, torch.Tensor):
        pair_tensor = pairs.cpu().long()
    else:
        pair_tensor = torch.tensor(pairs, dtype=torch.long)
    if pair_tensor.numel() == 0:
        return features.to(device).sum() * 0.0
    if pair_tensor.ndim != 2 or pair_tensor.shape[1] != 2:
        raise ValueError("pairs must have shape [P,2]")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")

    losses = []
    step = int(batch_size or pair_tensor.shape[0])
    for start in range(0, int(pair_tensor.shape[0]), step):
        chunk = pair_tensor[start : start + step]
        pos_idx = chunk[:, 0]
        neg_idx = chunk[:, 1]
        pos_logits = model.score_pairs(features[pos_idx].to(device), finding_ids[pos_idx].to(device))
        neg_logits = model.score_pairs(features[neg_idx].to(device), finding_ids[neg_idx].to(device))
        losses.append(torch.relu(float(margin) - pos_logits + neg_logits))
    return torch.cat(losses).mean()


def train_region_predictor(config: TrainConfig) -> dict[str, Any]:
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.rank_loss_weight < 0:
        raise ValueError("rank_loss_weight must be non-negative")
    if config.rank_margin < 0:
        raise ValueError("rank_margin must be non-negative")
    if config.max_rank_pairs_per_finding < 0:
        raise ValueError("max_rank_pairs_per_finding must be non-negative")
    eval_batch_size = config.eval_batch_size or config.batch_size
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = config.device
    train_payload = _load_cache(config.train_cache, device)
    valid_payload = _load_cache(config.valid_cache, device)
    train_features, train_labels, train_finding_ids = _valid_payload_tensors(train_payload, device)
    valid_features, valid_labels, valid_finding_ids = _valid_payload_tensors(valid_payload, device)
    train_region_names = _valid_region_names(train_payload)

    finding_vocab = dict(train_payload["finding_vocab"])
    if finding_vocab != dict(valid_payload["finding_vocab"]):
        raise ValueError("train and valid caches use different finding vocabularies")

    model = RegionAbnormalityPredictor(
        feature_dim=int(train_features.shape[1]),
        num_findings=len(finding_vocab),
        finding_embedding_dim=config.finding_embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    pos_weight = _positive_weight(train_labels, device) if config.use_pos_weight else None
    rank_pairs = (
        build_hard_negative_pairs(
            labels=train_labels,
            finding_ids=train_finding_ids,
            region_names=train_region_names,
            finding_vocab=finding_vocab,
            max_pairs_per_finding=config.max_rank_pairs_per_finding,
        )
        if config.rank_loss_weight > 0
        else []
    )
    if config.rank_loss_weight > 0 and not rank_pairs:
        raise ValueError(
            "rank_loss_weight > 0 but no hard-negative pairs were built; "
            "check region_names, finding_vocab, labels, and disease priors"
        )
    initial_train_loss = _loss_on_tensors(
        model,
        train_features,
        train_labels,
        train_finding_ids,
        pos_weight,
        eval_batch_size=eval_batch_size,
    )
    train_loader = _make_loader(train_features, train_labels, train_finding_ids, config.batch_size, config.seed)
    epoch_losses = []
    epoch_bce_losses = []
    epoch_rank_losses = []
    rank_pair_tensor = torch.tensor(rank_pairs, dtype=torch.long) if rank_pairs else torch.empty((0, 2), dtype=torch.long)
    rank_cursor = 0

    for epoch_idx in range(config.epochs):
        model.train()
        running_loss = 0.0
        running_bce_loss = 0.0
        running_rank_loss = 0.0
        running_count = 0
        for batch_features, batch_labels, batch_finding_ids in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_finding_ids = batch_finding_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.score_pairs(batch_features, batch_finding_ids)
            bce_loss = model.loss(logits, batch_labels, pos_weight=pos_weight)
            rank_loss = bce_loss * 0.0
            if config.rank_loss_weight > 0 and int(rank_pair_tensor.shape[0]) > 0:
                pair_batch_size = min(int(config.batch_size), int(rank_pair_tensor.shape[0]))
                end = rank_cursor + pair_batch_size
                if end <= int(rank_pair_tensor.shape[0]):
                    pair_batch = rank_pair_tensor[rank_cursor:end]
                else:
                    pair_batch = torch.cat(
                        [
                            rank_pair_tensor[rank_cursor:],
                            rank_pair_tensor[: end % int(rank_pair_tensor.shape[0])],
                        ],
                        dim=0,
                    )
                rank_cursor = end % int(rank_pair_tensor.shape[0])
                rank_loss = hard_negative_ranking_loss(
                    model=model,
                    features=train_features,
                    finding_ids=train_finding_ids,
                    pairs=pair_batch,
                    margin=config.rank_margin,
                    batch_size=pair_batch_size,
                )
            loss = bce_loss + float(config.rank_loss_weight) * rank_loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(batch_labels.numel())
            running_bce_loss += float(bce_loss.item()) * int(batch_labels.numel())
            running_rank_loss += float(rank_loss.item()) * int(batch_labels.numel())
            running_count += int(batch_labels.numel())
        epoch_loss = running_loss / max(running_count, 1)
        epoch_losses.append(epoch_loss)
        epoch_bce_losses.append(running_bce_loss / max(running_count, 1))
        epoch_rank_losses.append(running_rank_loss / max(running_count, 1))
        epoch_number = epoch_idx + 1
        if config.log_every > 0 and (
            epoch_number == 1 or epoch_number == config.epochs or epoch_number % config.log_every == 0
        ):
            print(
                f"[train_region_predictor] epoch {epoch_number}/{config.epochs} "
                f"loss={epoch_loss:.6f}",
                flush=True,
            )

    final_train_loss = _loss_on_tensors(
        model,
        train_features,
        train_labels,
        train_finding_ids,
        pos_weight,
        eval_batch_size=eval_batch_size,
    )
    final_valid_loss = _loss_on_tensors(
        model,
        valid_features,
        valid_labels,
        valid_finding_ids,
        pos_weight,
        eval_batch_size=eval_batch_size,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "region_predictor.pt"
    report_path = config.output_dir / "train_report.json"
    model_config = {
        "feature_dim": int(train_features.shape[1]),
        "num_findings": len(finding_vocab),
        "finding_embedding_dim": config.finding_embedding_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
    }
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "finding_vocab": finding_vocab,
        "train_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "status": "ok",
        "checkpoint": str(checkpoint_path),
        "train_cache": str(config.train_cache),
        "valid_cache": str(config.valid_cache),
        "finding_vocab": finding_vocab,
        "train_rows": int(train_labels.numel()),
        "valid_rows": int(valid_labels.numel()),
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "final_valid_loss": final_valid_loss,
        "epoch_losses": epoch_losses,
        "epoch_bce_losses": epoch_bce_losses,
        "epoch_rank_losses": epoch_rank_losses,
        "rank_loss_weight": float(config.rank_loss_weight),
        "rank_margin": float(config.rank_margin),
        "max_rank_pairs_per_finding": int(config.max_rank_pairs_per_finding),
        "num_rank_pairs": int(len(rank_pairs)),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    evaluate_region_predictor(
        checkpoint_path=checkpoint_path,
        cache_path=config.valid_cache,
        output_dir=config.output_dir / "valid_metrics",
        device=device,
        batch_size=eval_batch_size,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AnaPrior region abnormality predictor.")
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--valid-cache", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--finding-embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pos-weight", action="store_true")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for initial/final loss evaluation. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print training loss every N epochs. Use 0 to disable progress logs.",
    )
    parser.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.0,
        help="DCEM-v2-B hard-negative ranking loss weight. Default 0 keeps DCEM-v1 training.",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.2,
        help="Margin for DCEM-v2-B positive-vs-hard-negative region ranking.",
    )
    parser.add_argument(
        "--max-rank-pairs-per-finding",
        type=int,
        default=4096,
        help="Maximum hard-negative ranking pairs sampled per finding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train_region_predictor(
        TrainConfig(
            train_cache=args.train_cache,
            valid_cache=args.valid_cache,
            output_dir=args.outdir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            finding_embedding_dim=args.finding_embedding_dim,
            dropout=args.dropout,
            seed=args.seed,
            device=args.device,
            use_pos_weight=not args.no_pos_weight,
            log_every=args.log_every,
            eval_batch_size=args.eval_batch_size,
            rank_loss_weight=args.rank_loss_weight,
            rank_margin=args.rank_margin,
            max_rank_pairs_per_finding=args.max_rank_pairs_per_finding,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
