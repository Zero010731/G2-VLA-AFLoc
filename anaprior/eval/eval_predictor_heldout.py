"""Held-out evaluation for the region abnormality predictor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from anaprior.models.region_abnormality_predictor import RegionAbnormalityPredictor


def _as_float_tensor(values: torch.Tensor) -> torch.Tensor:
    return values.detach().cpu().float().flatten()


def binary_roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Compute binary AUROC with average ranks for tied scores."""

    scores = _as_float_tensor(scores)
    labels = _as_float_tensor(labels)
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        return None

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    rank_sum_pos = 0.0
    start = 0
    total = int(scores.numel())
    while start < total:
        end = start + 1
        while end < total and sorted_scores[end].item() == sorted_scores[start].item():
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum_pos += average_rank * float((sorted_labels[start:end] == 1).sum().item())
        start = end

    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def binary_average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Compute average precision from descending score order."""

    scores = _as_float_tensor(scores)
    labels = _as_float_tensor(labels)
    positives = int((labels == 1).sum().item())
    if positives == 0:
        return None

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order]
    cumulative_positive = torch.cumsum((sorted_labels == 1).float(), dim=0)
    ranks = torch.arange(1, sorted_labels.numel() + 1, dtype=torch.float32)
    precision_at_positive = cumulative_positive[sorted_labels == 1] / ranks[sorted_labels == 1]
    return float(precision_at_positive.mean().item())


def best_binary_f1(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float | None]:
    """Sweep unique score thresholds and return the best binary F1."""

    scores = _as_float_tensor(scores)
    labels = _as_float_tensor(labels)
    if scores.numel() == 0:
        return {"f1": None, "precision": None, "recall": None, "threshold": None}

    thresholds = torch.unique(scores, sorted=True).flip(0)
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": float(thresholds[0].item())}
    for threshold in thresholds:
        predictions = scores >= threshold
        tp = float(((predictions == 1) & (labels == 1)).sum().item())
        fp = float(((predictions == 1) & (labels == 0)).sum().item())
        fn = float(((predictions == 0) & (labels == 1)).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > float(best["f1"]):
            best = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "threshold": float(threshold.item()),
            }
    return best


def _metric_row(finding: str, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    f1 = best_binary_f1(scores, labels)
    return {
        "finding": finding,
        "n": int(labels.numel()),
        "positives": int((labels == 1).sum().item()),
        "negatives": int((labels == 0).sum().item()),
        "auroc": binary_roc_auc(scores, labels),
        "average_precision": binary_average_precision(scores, labels),
        "best_f1": f1["f1"],
        "best_precision": f1["precision"],
        "best_recall": f1["recall"],
        "best_threshold": f1["threshold"],
    }


def _probability_summary_row(finding: str, label_group: str, probabilities: torch.Tensor) -> dict[str, Any]:
    values = _as_float_tensor(probabilities)
    if values.numel() == 0:
        return {
            "finding": finding,
            "label_group": label_group,
            "n": 0,
            "probability_mean": None,
            "probability_std": None,
            "probability_min": None,
            "probability_q05": None,
            "probability_q25": None,
            "probability_q50": None,
            "probability_q75": None,
            "probability_q95": None,
            "probability_max": None,
            "fraction_below_0_2": None,
            "fraction_above_0_5": None,
        }
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float32))
    return {
        "finding": finding,
        "label_group": label_group,
        "n": int(values.numel()),
        "probability_mean": float(values.mean().item()),
        "probability_std": float(values.std(unbiased=False).item()),
        "probability_min": float(values.min().item()),
        "probability_q05": float(quantiles[0].item()),
        "probability_q25": float(quantiles[1].item()),
        "probability_q50": float(quantiles[2].item()),
        "probability_q75": float(quantiles[3].item()),
        "probability_q95": float(quantiles[4].item()),
        "probability_max": float(values.max().item()),
        "fraction_below_0_2": float((values < 0.2).float().mean().item()),
        "fraction_above_0_5": float((values >= 0.5).float().mean().item()),
    }


def probability_distribution_summary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    finding_ids: torch.Tensor,
    finding_vocab: dict[str, int],
) -> list[dict[str, Any]]:
    logits = _as_float_tensor(logits)
    labels = _as_float_tensor(labels)
    finding_ids = finding_ids.detach().cpu().long().flatten()
    probabilities = torch.sigmoid(logits)
    inverse_vocab = {idx: finding for finding, idx in finding_vocab.items()}

    rows: list[dict[str, Any]] = []
    for finding_id in ["overall", *sorted(inverse_vocab)]:
        if finding_id == "overall":
            finding = "overall"
            mask = torch.ones_like(labels, dtype=torch.bool)
        else:
            finding = inverse_vocab[int(finding_id)]
            mask = finding_ids == int(finding_id)
        if int(mask.sum().item()) == 0:
            continue
        finding_probs = probabilities[mask]
        finding_labels = labels[mask]
        rows.append(_probability_summary_row(finding, "all", finding_probs))
        rows.append(_probability_summary_row(finding, "positive", finding_probs[finding_labels == 1]))
        rows.append(_probability_summary_row(finding, "negative", finding_probs[finding_labels == 0]))
    return rows


def evaluate_logits_by_finding(
    logits: torch.Tensor,
    labels: torch.Tensor,
    finding_ids: torch.Tensor,
    finding_vocab: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate logits overall and separately for each finding id."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logits = _as_float_tensor(logits)
    labels = _as_float_tensor(labels)
    finding_ids = finding_ids.detach().cpu().long().flatten()
    if logits.shape != labels.shape or logits.shape != finding_ids.shape:
        raise ValueError("logits, labels, and finding_ids must have the same flat shape")

    inverse_vocab = {idx: finding for finding, idx in finding_vocab.items()}
    per_finding = []
    for finding_id in sorted(inverse_vocab):
        mask = finding_ids == finding_id
        if int(mask.sum().item()) == 0:
            continue
        per_finding.append(_metric_row(inverse_vocab[finding_id], logits[mask], labels[mask]))

    report = {
        "overall": _metric_row("overall", logits, labels),
        "per_finding": per_finding,
        "probability_summary": probability_distribution_summary(logits, labels, finding_ids, finding_vocab),
        "finding_vocab": finding_vocab,
    }
    json_path = output_dir / "predictor_heldout_metrics.json"
    csv_path = output_dir / "predictor_heldout_metrics.csv"
    probability_csv_path = output_dir / "predictor_heldout_probability_summary.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["overall"].keys()))
        writer.writeheader()
        writer.writerow(report["overall"])
        writer.writerows(per_finding)
    with probability_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(report["probability_summary"][0].keys()) if report["probability_summary"] else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(report["probability_summary"])
    return report


def load_predictor_from_checkpoint(path: Path, device: str) -> tuple[RegionAbnormalityPredictor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model = RegionAbnormalityPredictor(**model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _score_pairs_in_batches(
    model: RegionAbnormalityPredictor,
    region_features: torch.Tensor,
    finding_ids: torch.Tensor,
    batch_size: int | None = None,
) -> torch.Tensor:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    effective_batch_size = batch_size or int(finding_ids.numel())
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = region_features.device
    logits = []
    with torch.no_grad():
        for start in range(0, int(finding_ids.numel()), effective_batch_size):
            end = min(start + effective_batch_size, int(finding_ids.numel()))
            logits.append(
                model.score_pairs(
                    region_features[start:end].to(device),
                    finding_ids[start:end].to(device),
                )
                .detach()
                .cpu()
            )
    if not logits:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(logits, dim=0)


def evaluate_region_predictor(
    checkpoint_path: Path,
    cache_path: Path,
    output_dir: Path,
    device: str = "cpu",
    batch_size: int | None = None,
) -> dict[str, Any]:
    model, checkpoint = load_predictor_from_checkpoint(checkpoint_path, device=device)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    region_features = payload["region_features"].cpu().float()
    finding_ids = payload["finding_ids"].cpu().long()
    labels = payload["labels"].cpu().float()
    valid_mask = payload.get("valid_mask", torch.ones_like(labels, dtype=torch.bool)).cpu().bool()
    logits = _score_pairs_in_batches(model, region_features, finding_ids, batch_size=batch_size)
    valid_mask_cpu = valid_mask.detach().cpu()
    return evaluate_logits_by_finding(
        logits=logits[valid_mask_cpu],
        labels=labels.detach().cpu()[valid_mask_cpu],
        finding_ids=finding_ids.detach().cpu()[valid_mask_cpu],
        finding_vocab=checkpoint["finding_vocab"],
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained region abnormality predictor.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional scoring batch size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_region_predictor(
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        output_dir=args.outdir,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
