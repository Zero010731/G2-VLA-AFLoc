"""Single-image AFLoc CXR heatmap inference.

Example:
    conda run -n AFLoc python scripts/infer_cxr_heatmap.py \
        --image C:/path/to/a/single_cxr_image.png \
        --prompt "pleural effusion"

The script writes:
    - *_overlay.png: preprocessed CXR with heatmap overlay
    - *_heatmap.png: normalized heatmap only
    - *_meta.json: prompt, checkpoint, similarity and tensor metadata

Use a single frontal/lateral chest X-ray image. Do not use paper figure panels,
collages, screenshots with labels, or multi-image result composites.

This is a research/debug utility, not a clinical diagnostic tool.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import afloc


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", text.strip())
    slug = slug.strip("_")
    return (slug[:max_len] or "prompt").strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an AFLoc heatmap for one CXR image and one prompt.")
    parser.add_argument(
        "--ckpt",
        default=str(REPO_ROOT / "weight" / "pretrained" / "Pretrained_CXR.ckpt"),
        help="Path to the AFLoc CXR checkpoint.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to one chest X-ray image.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt, preferably English because the released model uses ClinicalBERT.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "outputs" / "cxr_heatmaps"),
        help="Directory for generated visualizations.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap opacity in the overlay image.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for the patch similarity map.",
    )
    parser.add_argument(
        "--equalize-hist",
        action="store_true",
        help="Apply histogram equalization, matching the optional preprocessing path in AFLoc.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return requested


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    valid = np.isfinite(heatmap)
    if not np.any(valid):
        return np.zeros_like(heatmap, dtype=np.float32)
    hmin = float(np.min(heatmap[valid]))
    hmax = float(np.max(heatmap[valid]))
    if abs(hmax - hmin) < 1e-8:
        out = np.zeros_like(heatmap, dtype=np.float32)
        out[valid] = 0.0
        return out
    out = (heatmap - hmin) / (hmax - hmin)
    out[~valid] = 0.0
    return out.astype(np.float32)


def patch_text_similarity_map(
    patch_embeddings: torch.Tensor,
    text_global_embedding: torch.Tensor,
    sigma: float,
) -> np.ndarray:
    """Compute a smoothed patch-level similarity map.

    patch_embeddings: [H, W, D]
    text_global_embedding: [1, D]
    """
    h, w, dim = patch_embeddings.shape
    if text_global_embedding.shape != (1, dim):
        raise ValueError(f"Expected text embedding shape (1, {dim}), got {tuple(text_global_embedding.shape)}")
    sim = patch_embeddings.reshape(-1, dim) @ text_global_embedding.t()
    sim = sim.reshape(h, w).detach().cpu().numpy()
    return ndimage.gaussian_filter(sim, sigma=(sigma, sigma), order=0).astype(np.float32)


def resize_to_square_map(sim_map: np.ndarray, size: int) -> np.ndarray:
    resized = cv2.resize(sim_map, (size, size), interpolation=cv2.INTER_CUBIC)
    return resized.astype(np.float32)


def make_reference_image(model, image_path: Path, flag: int, equalize_hist: bool) -> np.ndarray:
    """Create the same 224x224 padded image space used for AFLoc input."""
    arr = cv2.imread(str(image_path), flag)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if flag == 0:
        arr = model._resize_img(arr, model.cfg.data.image.imsize)
    else:
        arr = cv2.resize(arr, (model.cfg.data.image.imsize, model.cfg.data.image.imsize), interpolation=cv2.INTER_AREA)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if equalize_hist:
        arr = arr.astype(np.float32) / 255.0
        arr = cv2.equalizeHist((arr * 255).astype(np.uint8)) if arr.ndim == 2 else arr
        arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return arr.astype(np.uint8)


def save_visualizations(
    ref_rgb: np.ndarray,
    heatmap01: np.ndarray,
    out_prefix: Path,
    alpha: float,
) -> dict[str, str]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    heat_u8 = np.clip(heatmap01 * 255, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.clip((1.0 - alpha) * ref_rgb + alpha * colored_rgb, 0, 255).astype(np.uint8)

    heatmap_path = out_prefix.with_name(out_prefix.name + "_heatmap.png")
    overlay_path = out_prefix.with_name(out_prefix.name + "_overlay.png")
    ref_path = out_prefix.with_name(out_prefix.name + "_preprocessed.png")

    cv2.imwrite(str(heatmap_path), colored_bgr)
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(ref_path), cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR))

    return {
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
        "preprocessed_image": str(ref_path),
    }


def main() -> int:
    args = parse_args()
    ckpt = Path(args.ckpt)
    image = Path(args.image)
    out_dir = Path(args.out_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    device = resolve_device(args.device)
    model = afloc.builder.load_model(ckpt_path=str(ckpt), device=device)
    model.eval()
    model.cfg.data.image.imsize = 224
    flag = model.cfg.data.image.flag

    with torch.no_grad():
        imgs = model.process_img(
            str(image),
            device=device,
            equalize_hist=args.equalize_hist,
            flag=flag,
        )
        img_emb_l, _, _, img_emb_g = model.image_encoder_forward(imgs)
        text = model.process_text(args.prompt, device=device)
        text_res = model.text_encoder_forward(
            text["caption_ids"],
            text["attention_mask"],
            text["token_type_ids"],
        )
        text_emb_g = text_res["report_embeddings"]

        patch_emb = img_emb_l.view(-1, *img_emb_l.shape[2:]).permute(1, 2, 0)
        raw_map = patch_text_similarity_map(patch_emb, text_emb_g, sigma=args.sigma)
        heatmap224 = resize_to_square_map(raw_map, size=model.cfg.data.image.imsize)
        heatmap01 = normalize_heatmap(heatmap224)
        global_similarity = F.cosine_similarity(img_emb_g, text_emb_g, dim=-1).detach().cpu().tolist()

    ref_rgb = make_reference_image(model, image, flag=flag, equalize_hist=args.equalize_hist)
    out_prefix = out_dir / f"{image.stem}_{slugify(args.prompt)}"
    paths = save_visualizations(ref_rgb, heatmap01, out_prefix, alpha=args.alpha)

    meta = {
        "status": "ok",
        "checkpoint": str(ckpt),
        "image": str(image),
        "prompt": args.prompt,
        "device": str(next(model.parameters()).device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "global_similarity": global_similarity,
        "raw_similarity_min": float(np.min(raw_map)),
        "raw_similarity_max": float(np.max(raw_map)),
        "heatmap_min": float(np.min(heatmap01)),
        "heatmap_max": float(np.max(heatmap01)),
        "image_tensor_shape": list(imgs.shape),
        "local_embedding_shape": list(img_emb_l.shape),
        "text_global_embedding_shape": list(text_emb_g.shape),
        "outputs": paths,
    }
    meta_path = out_prefix.with_name(out_prefix.name + "_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["outputs"]["metadata"] = str(meta_path)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
