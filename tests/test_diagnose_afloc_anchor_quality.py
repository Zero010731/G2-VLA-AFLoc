from __future__ import annotations

import numpy as np
import pytest
import torch

from anaprior.eval.diagnose_afloc_anchor_quality import (
    anchor_metrics,
    phrase_patch_anchor,
    summarize_anchor_metrics,
)


def test_anchor_metrics_detects_correct_topk_and_pointing() -> None:
    anchor = torch.tensor([[0.9, 0.8], [0.1, 0.0]])
    metrics = anchor_metrics(anchor, np.array([[1, 1], [0, 0]], dtype=np.float32), topk_fraction=0.5)

    assert metrics["topk_iou"] == pytest.approx(1.0)
    assert metrics["topk_dice"] == pytest.approx(1.0)
    assert metrics["pointing_hit"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_anchor_metrics_accepts_cpu_gt_mask_for_cuda_anchor() -> None:
    anchor = torch.tensor([[0.9, 0.8], [0.1, 0.0]], device="cuda")
    metrics = anchor_metrics(anchor, np.array([[1, 1], [0, 0]], dtype=np.float32), topk_fraction=0.5)

    assert metrics["topk_iou"] == pytest.approx(1.0)


def test_phrase_patch_anchor_fuses_three_scales_and_masks_tokens() -> None:
    image = {
        "l2": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]]),
        "l": torch.tensor([[[[1.0]], [[0.0]]]]),
        "lf": torch.tensor([[[[1.0]], [[0.0]]]]),
    }
    words = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]])
    anchor, scales = phrase_patch_anchor(
        image,
        words,
        torch.tensor([[True, False, False]]),
    )

    assert anchor.shape == (2, 2)
    assert set(scales) == {"l2", "l", "lf"}
    assert torch.isfinite(anchor).all()


def test_anchor_summary_marks_weak_anchor() -> None:
    summary = summarize_anchor_metrics(
        [
            {"category": "Pneumonia", "topk_iou": 0.01, "topk_dice": 0.02, "pointing_hit": 0.0, "anchor_area_ratio": 0.15, "gt_area_ratio": 0.1, "multi_scale_agreement": 0.5},
        ]
    )

    assert summary["diagnostic_only"] is True
    assert summary["anchor_decision"].startswith("weak_anchor")
