"""Train DP-MSA-v0 from weak anatomy/region supervision caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from anaprior.models.dp_msa_adapter import DPMultiScaleSpatialAdapter, normalize_heatmaps


REQUIRED_CACHE_FIELDS = {
    "base_hmaps",
    "region_maps",
    "region_scores",
    "target_hmaps",
    "disease_ids",
    "subtype_ids",
    "finding_vocab",
    "subtype_vocab",
    "region_names",
}


def load_dp_msa_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing = sorted(REQUIRED_CACHE_FIELDS - set(payload))
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    base_hmaps = payload["base_hmaps"].float()
    region_maps = payload["region_maps"].float()
    region_scores = payload["region_scores"].float()
    target_hmaps = payload["target_hmaps"].float()
    disease_ids = payload["disease_ids"].long()
    subtype_ids = payload["subtype_ids"].long()
    if base_hmaps.ndim != 4 or base_hmaps.shape[1] != 1:
        raise ValueError("base_hmaps must have shape [N,1,H,W]")
    if region_maps.ndim != 4:
        raise ValueError("region_maps must have shape [N,R,H,W]")
    if region_scores.shape != region_maps.shape[:2]:
        raise ValueError("region_scores must have shape [N,R]")
    if target_hmaps.shape != base_hmaps.shape:
        raise ValueError("target_hmaps must match base_hmaps shape")
    n = int(base_hmaps.shape[0])
    if disease_ids.shape[0] != n or subtype_ids.shape[0] != n:
        raise ValueError("disease_ids and subtype_ids must match cache row count")
    return {
        **payload,
        "base_hmaps": base_hmaps,
        "region_maps": region_maps,
        "region_scores": region_scores,
        "target_hmaps": normalize_heatmaps(target_hmaps),
        "disease_ids": disease_ids,
        "subtype_ids": subtype_ids,
    }


def _make_loader(payload: dict[str, Any], batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dataset = TensorDataset(
        payload["base_hmaps"],
        payload["region_maps"],
        payload["region_scores"],
        payload["target_hmaps"],
        payload["disease_ids"],
        payload["subtype_ids"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def _loss_on_loader(
    model: DPMultiScaleSpatialAdapter,
    loader: DataLoader,
    device: torch.device,
    residual_l1_weight: float,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for base, maps, scores, target, disease_ids, subtype_ids in loader:
            out = model(
                base.to(device),
                maps.to(device),
                scores.to(device),
                disease_ids.to(device),
                subtype_ids.to(device),
            )
            loss = F.mse_loss(out.final_heatmap, target.to(device))
            loss = loss + float(residual_l1_weight) * out.residual_map.abs().mean()
            total += float(loss.item()) * int(base.shape[0])
            count += int(base.shape[0])
    return total / max(count, 1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dp_msa_adapter(
    train_cache: Path,
    valid_cache: Path,
    outdir: Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    hidden_channels: int = 16,
    embedding_dim: int = 32,
    lambda_weight: float = 0.1,
    residual_l1_weight: float = 0.01,
    seed: int = 13,
    device: str = "cpu",
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    _set_seed(seed)
    torch_device = torch.device(device)
    train_payload = load_dp_msa_cache(train_cache)
    valid_payload = load_dp_msa_cache(valid_cache)
    finding_vocab = {str(k): int(v) for k, v in train_payload["finding_vocab"].items()}
    subtype_vocab = {str(k): int(v) for k, v in train_payload["subtype_vocab"].items()}
    region_names = [str(region) for region in train_payload["region_names"]]
    model_config = {
        "num_diseases": len(finding_vocab),
        "num_subtypes": len(subtype_vocab),
        "num_regions": int(train_payload["region_maps"].shape[1]),
        "hidden_channels": int(hidden_channels),
        "embedding_dim": int(embedding_dim),
        "lambda_weight": float(lambda_weight),
        "residual_scale": 0.25,
    }
    model = DPMultiScaleSpatialAdapter(**model_config).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    train_loader = _make_loader(train_payload, batch_size=batch_size, seed=seed)
    valid_loader = _make_loader(valid_payload, batch_size=batch_size, seed=seed + 1)
    epoch_losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        weighted_loss = 0.0
        total = 0
        for base, maps, scores, target, disease_ids, subtype_ids in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                base.to(torch_device),
                maps.to(torch_device),
                scores.to(torch_device),
                disease_ids.to(torch_device),
                subtype_ids.to(torch_device),
            )
            loss = F.mse_loss(out.final_heatmap, target.to(torch_device))
            loss = loss + float(residual_l1_weight) * out.residual_map.abs().mean()
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * int(base.shape[0])
            total += int(base.shape[0])
        epoch_losses.append(weighted_loss / max(total, 1))
    final_valid_loss = _loss_on_loader(model, valid_loader, torch_device, residual_l1_weight=residual_l1_weight)
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "dp_msa_adapter.pt"
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_config": model_config,
            "finding_vocab": finding_vocab,
            "subtype_vocab": subtype_vocab,
            "region_names": region_names,
        },
        checkpoint,
    )
    report_path = outdir / "train_report.json"
    report: dict[str, Any] = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "report": str(report_path),
        "train_cache": str(train_cache),
        "valid_cache": str(valid_cache),
        "finding_vocab": finding_vocab,
        "subtype_vocab": subtype_vocab,
        "region_names": region_names,
        "num_train_rows": int(train_payload["base_hmaps"].shape[0]),
        "num_valid_rows": int(valid_payload["base_hmaps"].shape[0]),
        "num_regions": int(train_payload["region_maps"].shape[1]),
        "epoch_losses": epoch_losses,
        "final_train_loss": float(epoch_losses[-1]),
        "final_valid_loss": float(final_valid_loss),
        "model_config": model_config,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DP-MSA-v0 weak residual spatial adapter.")
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--valid-cache", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--lambda-weight", type=float, default=0.1)
    parser.add_argument("--residual-l1-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = train_dp_msa_adapter(
        train_cache=args.train_cache,
        valid_cache=args.valid_cache,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        embedding_dim=args.embedding_dim,
        lambda_weight=args.lambda_weight,
        residual_l1_weight=args.residual_l1_weight,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
