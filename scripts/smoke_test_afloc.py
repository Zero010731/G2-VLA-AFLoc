"""Minimal Windows smoke test for the AFLoc CXR checkpoint.

This script verifies that the environment can:
1. import AFLoc and PyTorch,
2. load the downloaded CXR checkpoint,
3. preprocess one image and one text prompt,
4. run a forward pass on CPU or CUDA.

It is not a clinical inference script. The default image is a repository
visualization asset, so the printed similarity score is only for environment
verification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import afloc


def default_repo_root() -> Path:
    return REPO_ROOT


def parse_args() -> argparse.Namespace:
    root = default_repo_root()
    parser = argparse.ArgumentParser(description="Run a minimal AFLoc smoke test.")
    parser.add_argument(
        "--ckpt",
        default=str(root / "weight" / "pretrained" / "Pretrained_CXR.ckpt"),
        help="Path to the AFLoc CXR checkpoint.",
    )
    parser.add_argument(
        "--image",
        default=str(root / "assets" / "results_cxr.jpg"),
        help="Path to a test image.",
    )
    parser.add_argument(
        "--text",
        default="There is pleural effusion.",
        help="English text prompt for the ClinicalBERT text encoder.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for the test.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


def main() -> int:
    args = parse_args()
    ckpt = Path(args.ckpt)
    image = Path(args.image)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    device = resolve_device(args.device)
    model = afloc.builder.load_model(ckpt_path=str(ckpt), device=device)
    model.eval()

    with torch.no_grad():
        imgs = model.process_img(
            str(image),
            device=device,
            flag=model.cfg.data.image.flag,
        )
        text = model.process_text(args.text, device=device)
        output = model(
            {
                "imgs": imgs,
                "caption_ids": text["caption_ids"],
                "attention_mask": text["attention_mask"],
                "token_type_ids": text["token_type_ids"],
            }
        )
        global_similarity = F.cosine_similarity(
            output["img_emb_g"],
            output["text_emb_r"],
            dim=-1,
        ).detach().cpu().tolist()

    report = {
        "status": "ok",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(next(model.parameters()).device),
        "checkpoint": str(ckpt),
        "image": str(image),
        "image_tensor_shape": list(imgs.shape),
        "embeddings": {
            key: list(value.shape)
            for key, value in output.items()
            if hasattr(value, "shape")
        },
        "global_similarity": global_similarity,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
