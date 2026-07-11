"""Export learned region abnormality scores from a cached feature table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from anaprior.eval.eval_predictor_heldout import load_predictor_from_checkpoint


FIELDNAMES = [
    "row_index",
    "dicom_id",
    "subject_id",
    "study_id",
    "region",
    "finding",
    "finding_id",
    "valid",
    "score_logit",
    "score_probability",
]


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, "")
    return "" if value is None else str(value)


def _inverse_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {int(idx): str(finding) for finding, idx in vocab.items()}


def export_region_scores(
    checkpoint_path: Path,
    cache_path: Path,
    output_csv: Path,
    device: str = "cpu",
    valid_only: bool = True,
) -> dict[str, Any]:
    model, checkpoint = load_predictor_from_checkpoint(checkpoint_path, device=device)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    features = payload["region_features"].float()
    finding_ids = payload["finding_ids"].long()
    valid_mask = payload.get("valid_mask", torch.ones(finding_ids.shape, dtype=torch.bool)).bool()
    metadata = list(payload.get("metadata", [{} for _ in range(int(finding_ids.numel()))]))
    if features.shape[0] != finding_ids.shape[0] or len(metadata) != int(finding_ids.numel()):
        raise ValueError("cache feature rows, finding ids, and metadata must have matching lengths")

    checkpoint_vocab = {str(finding): int(idx) for finding, idx in checkpoint["finding_vocab"].items()}
    checkpoint_inverse_vocab = _inverse_vocab(checkpoint_vocab)
    cache_vocab = {str(finding): int(idx) for finding, idx in payload.get("finding_vocab", {}).items()}
    cache_inverse_vocab = _inverse_vocab(cache_vocab)

    selected_indices: list[int] = []
    selected_finding_ids: list[int] = []
    selected_findings: list[str] = []
    skipped_invalid = 0
    skipped_unsupported = 0
    unsupported_findings: dict[str, int] = {}
    for idx in range(int(finding_ids.numel())):
        valid = bool(valid_mask[idx].item())
        if valid_only and not valid:
            skipped_invalid += 1
            continue
        source_finding_id = int(finding_ids[idx].item())
        meta = metadata[idx]
        finding = (
            cache_inverse_vocab.get(source_finding_id)
            or _metadata_value(meta, "finding")
            or checkpoint_inverse_vocab.get(source_finding_id, str(source_finding_id))
        )
        if finding not in checkpoint_vocab:
            skipped_unsupported += 1
            unsupported_findings[finding] = unsupported_findings.get(finding, 0) + 1
            continue
        selected_indices.append(idx)
        selected_finding_ids.append(checkpoint_vocab[finding])
        selected_findings.append(finding)

    with torch.no_grad():
        if selected_indices:
            selected_features = features[selected_indices].to(device)
            mapped_finding_ids = torch.tensor(selected_finding_ids, dtype=torch.long, device=device)
            logits = model.score_pairs(selected_features, mapped_finding_ids).detach().cpu()
        else:
            logits = torch.empty((0,), dtype=torch.float32)
    probabilities = torch.sigmoid(logits)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for out_idx, idx in enumerate(selected_indices):
            valid = bool(valid_mask[idx].item())
            meta = metadata[idx]
            finding_id = selected_finding_ids[out_idx]
            writer.writerow(
                {
                    "row_index": idx,
                    "dicom_id": _metadata_value(meta, "dicom_id"),
                    "subject_id": _metadata_value(meta, "subject_id"),
                    "study_id": _metadata_value(meta, "study_id"),
                    "region": _metadata_value(meta, "region"),
                    "finding": selected_findings[out_idx],
                    "finding_id": finding_id,
                    "valid": int(valid),
                    "score_logit": float(logits[out_idx].item()),
                    "score_probability": float(probabilities[out_idx].item()),
                }
            )
            rows_written += 1

    report = {
        "status": "ok",
        "checkpoint": str(checkpoint_path),
        "cache": str(cache_path),
        "output_csv": str(output_csv),
        "rows_total": int(finding_ids.numel()),
        "rows_written": rows_written,
        "skipped_invalid": skipped_invalid,
        "skipped_unsupported": skipped_unsupported,
        "unsupported_findings": unsupported_findings,
        "valid_only": bool(valid_only),
        "finding_vocab": checkpoint_vocab,
        "cache_finding_vocab": cache_vocab,
    }
    report_path = output_csv.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export learned region abnormality scores.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-invalid", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = export_region_scores(
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        output_csv=args.output_csv,
        device=args.device,
        valid_only=not args.include_invalid,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
