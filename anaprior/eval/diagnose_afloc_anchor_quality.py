"""Diagnostic-only evaluation of frozen AFLoc phrase-patch anchors.

MS-CXR annotations are used here only to estimate whether an AFLoc anchor is
worth using as weak spatial evidence. The output must not be used for training
or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder
from anaprior.models.afloc_mrsg.anchor import compute_afloc_phrase_anchor
from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch, PhraseFeatureBatch


def normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    low = values.amin()
    high = values.amax()
    if not torch.isfinite(values).all() or float(high - low) <= 1.0e-8:
        return torch.zeros_like(values)
    return (values - low) / (high - low)


def phrase_patch_anchor(
    image_feature_maps: Mapping[str, torch.Tensor],
    word_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a multi-scale max-token cosine anchor on the finest grid."""
    if word_embeddings.ndim != 3 or word_embeddings.shape[0] != 1:
        raise ValueError("word_embeddings must have shape [1,T,C]")
    valid = attention_mask.to(dtype=torch.bool)
    if valid.ndim != 2 or not bool(valid.any(dim=1).all()):
        raise ValueError("attention_mask must contain at least one valid token")
    image_batch = AFLocFeatureBatch(
        img_emb_l2=image_feature_maps["l2"],
        img_emb_l=image_feature_maps["l"],
        img_emb_lf=image_feature_maps["lf"],
        image_gray=torch.empty(
            word_embeddings.shape[0],
            1,
            0,
            0,
            device=word_embeddings.device,
        ),
    )
    pooled_words = (word_embeddings * valid[:, :, None]).sum(dim=1)
    pooled_words = pooled_words / valid.sum(dim=1, keepdim=True).clamp_min(1)
    phrase_batch = PhraseFeatureBatch(
        word_embeddings=word_embeddings,
        sentence_embedding=pooled_words,
        disease_description_embedding=pooled_words,
        attention_mask=valid,
    )
    anchor, _, scale_maps = compute_afloc_phrase_anchor(image_batch, phrase_batch)
    return anchor[0, 0], {name: value[0, 0] for name, value in scale_maps.items()}


def anchor_metrics(
    anchor: torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    topk_fraction: float = 0.15,
) -> dict[str, float]:
    if anchor.ndim != 2:
        raise ValueError("anchor must have shape [H,W]")
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0,1]")
    gt = torch.as_tensor(gt_mask, dtype=torch.float32, device=anchor.device)
    if gt.ndim != 2:
        raise ValueError("gt_mask must have shape [H,W]")
    gt = F.interpolate(gt[None, None], size=anchor.shape, mode="nearest")[0, 0] > 0.0
    anchor = normalize_map(anchor)
    count = max(1, round(anchor.numel() * topk_fraction))
    topk = torch.zeros_like(anchor, dtype=torch.bool).flatten()
    topk[anchor.flatten().topk(count).indices] = True
    topk = topk.view_as(anchor)
    intersection = float((topk & gt).sum())
    union = float((topk | gt).sum())
    gt_count = float(gt.sum())
    pred_count = float(topk.sum())
    pointing = bool(gt.flatten()[int(anchor.argmax())]) if gt_count > 0 else False
    return {
        "topk_iou": intersection / union if union else 0.0,
        "topk_dice": 2.0 * intersection / (pred_count + gt_count) if pred_count + gt_count else 0.0,
        "pointing_hit": 1.0 if pointing else 0.0,
        "anchor_area_ratio": float((anchor >= 0.5).float().mean()),
        "gt_area_ratio": gt_count / float(gt.numel()),
    }


def summarize_anchor_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"status": "no_rows", "n": 0}
    metric_names = (
        "topk_iou",
        "topk_dice",
        "pointing_hit",
        "anchor_area_ratio",
        "gt_area_ratio",
        "multi_scale_agreement",
    )
    summary: dict[str, Any] = {
        "status": "ok",
        "n": int(len(frame)),
        "diagnostic_only": True,
        "metrics": {name: float(frame[name].mean()) for name in metric_names},
        "per_class": {
            str(category): {
                "n": int(group.shape[0]),
                **{name: float(group[name].mean()) for name in metric_names},
            }
            for category, group in frame.groupby("category", sort=True)
        },
    }
    summary["anchor_decision"] = (
        "promising"
        if summary["metrics"]["topk_iou"] >= 0.15
        and summary["metrics"]["pointing_hit"] >= 0.30
        else "weak_anchor_requires_stronger_cross_view_supervision"
    )
    return summary


def run_anchor_diagnostic(
    *,
    afloc_checkpoint: Path,
    mscxr_json: Path,
    mimic_img_dir: Path,
    outdir: Path,
    device: str = "cpu",
    max_cases: int | None = None,
    topk_fraction: float = 0.15,
    bert_type: str | None = None,
    hf_local_files_only: bool = True,
) -> dict[str, Any]:
    from localization.datasets import load_data

    if bert_type:
        os.environ["AFLOC_BERT_TYPE"] = str(bert_type)
    if hf_local_files_only:
        os.environ["AFLOC_HF_LOCAL_FILES_ONLY"] = "1"

    data = load_data(
        dataset="MS_CXR",
        ms_cxr_json=mscxr_json,
        mimic_img_dir=mimic_img_dir,
    )
    frame = pd.DataFrame(data)
    if max_cases is not None:
        frame = frame.iloc[:max_cases].copy()
    encoder = FrozenAFLocMRSGEncoder.from_checkpoint(afloc_checkpoint, device=device)
    encoder.eval()
    rows: list[dict[str, Any]] = []
    for index, record in frame.iterrows():
        path = str(record["path"])
        phrase = str(record["label_text"])
        category = str(record["category"])
        image = encoder.afloc.process_img([path], device, flag=0)
        image_features = encoder.encode_images(image)
        phrase_features = encoder.encode_phrases([phrase], [category], device=device)
        anchor, scales = phrase_patch_anchor(
            {
                "l2": image_features.img_emb_l2,
                "l": image_features.img_emb_l,
                "lf": image_features.img_emb_lf,
            },
            phrase_features.word_embeddings,
            phrase_features.attention_mask,
        )
        metrics = anchor_metrics(anchor, np.asarray(record["gtmasks"]), topk_fraction)
        scale_vectors = torch.stack([normalize_map(scales[name]).flatten() for name in ("l2", "l", "lf")])
        agreement = float(F.cosine_similarity(scale_vectors[0:1], scale_vectors[1:2]).item())
        agreement = (agreement + float(F.cosine_similarity(scale_vectors[0:1], scale_vectors[2:3]).item())) / 2.0
        rows.append({
            "index": int(index),
            "path": path,
            "label_text": phrase,
            "category": category,
            "multi_scale_agreement": agreement,
            **metrics,
        })
    outdir.mkdir(parents=True, exist_ok=True)
    cases_path = outdir / "anchor_cases.csv"
    pd.DataFrame(rows).to_csv(cases_path, index=False)
    summary = summarize_anchor_metrics(rows)
    summary.update({
        "afloc_checkpoint": str(afloc_checkpoint),
        "mscxr_json": str(mscxr_json),
        "n_requested": int(len(frame)),
        "topk_fraction": float(topk_fraction),
        "output_cases": str(cases_path),
    })
    (outdir / "anchor_quality_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic-only frozen AFLoc anchor quality evaluation.")
    parser.add_argument("--afloc-checkpoint", required=True, type=Path)
    parser.add_argument("--mscxr-json", required=True, type=Path)
    parser.add_argument("--mimic-img-dir", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bert-type", required=True)
    parser.add_argument("--hf-local-files-only", action="store_true", default=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--topk-fraction", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_anchor_diagnostic(
        afloc_checkpoint=args.afloc_checkpoint,
        mscxr_json=args.mscxr_json,
        mimic_img_dir=args.mimic_img_dir,
        outdir=args.outdir,
        device=args.device,
        max_cases=args.max_cases,
        topk_fraction=args.topk_fraction,
        bert_type=args.bert_type,
        hf_local_files_only=args.hf_local_files_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
