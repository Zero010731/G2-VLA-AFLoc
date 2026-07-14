from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class AFLocImagePreprocessing:
    imsize: int
    center_crop_size: tuple[int, int] | None
    norm: str | None
    flag: int = 0

    @property
    def output_size(self) -> tuple[int, int]:
        if self.center_crop_size is not None:
            return self.center_crop_size
        return (int(self.imsize), int(self.imsize))


def _cfg_get(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _normalize_crop_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Unsupported AFLoc crop size: {value!r}")


def extract_afloc_image_preprocessing(afloc_model: Any) -> AFLocImagePreprocessing:
    cfg = getattr(afloc_model, "cfg", None)
    if cfg is None:
        raise ValueError("AFLoc runtime does not expose cfg for preprocessing extraction")

    imsize = _cfg_get(cfg, "data", "image", "imsize")
    if imsize is None:
        raise ValueError("AFLoc cfg is missing data.image.imsize")

    crop_size = _normalize_crop_size(_cfg_get(cfg, "transforms", "random_crop", "crop_size"))
    norm = _cfg_get(cfg, "transforms", "norm")
    if norm not in (None, "half", "imagenet"):
        raise ValueError(f"Unsupported AFLoc normalization mode: {norm!r}")

    return AFLocImagePreprocessing(
        imsize=int(imsize),
        center_crop_size=crop_size,
        norm=norm,
        flag=0,
    )


def _resize_and_pad_grayscale(image: np.ndarray, scale: int) -> np.ndarray:
    size = image.shape
    max_dim = max(size)
    max_index = size.index(max_dim)

    if max_index == 0:
        width_percent = scale / float(size[0])
        resized_height = scale
        resized_width = int(float(size[1]) * width_percent)
    else:
        height_percent = scale / float(size[1])
        resized_height = int(float(size[0]) * height_percent)
        resized_width = scale

    resized = np.asarray(
        Image.fromarray(image).resize(
            (int(resized_width), int(resized_height)),
            Image.Resampling.BOX,
        ),
        dtype=image.dtype,
    )

    if max_index == 0:
        pad_size = scale - resized.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = 0
        bottom = 0
    else:
        pad_size = scale - resized.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = 0
        right = 0

    return np.pad(
        resized,
        [(top, bottom), (left, right)],
        mode="constant",
        constant_values=0,
    )


def _center_crop(image: Image.Image, crop_size: tuple[int, int] | None) -> Image.Image:
    if crop_size is None:
        return image
    crop_height, crop_width = (int(crop_size[0]), int(crop_size[1]))
    width, height = image.size
    left = max((width - crop_width) // 2, 0)
    top = max((height - crop_height) // 2, 0)
    right = left + min(crop_width, width)
    bottom = top + min(crop_height, height)
    return image.crop((left, top, right, bottom))


def _normalize_rgb(image: torch.Tensor, norm: str | None) -> torch.Tensor:
    if norm is None:
        return image
    if norm == "half":
        mean = torch.tensor((0.5, 0.5, 0.5), dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor((0.5, 0.5, 0.5), dtype=image.dtype, device=image.device).view(3, 1, 1)
        return (image - mean) / std
    if norm == "imagenet":
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype, device=image.device).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=image.dtype, device=image.device).view(3, 1, 1)
        return (image - mean) / std
    raise ValueError(f"Unsupported AFLoc normalization mode: {norm!r}")


def preprocess_afloc_image_array(
    image: np.ndarray,
    config: AFLocImagePreprocessing,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 2:
        raise ValueError(f"AFLoc grayscale preprocessing expects a 2D array, got shape {image.shape}")

    resized = (
        _resize_and_pad_grayscale(image, int(config.imsize))
        if int(config.flag) == 0
        else image
    )
    rgb = _center_crop(Image.fromarray(resized).convert("RGB"), config.center_crop_size)
    gray = rgb.convert("L")

    rgb_tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
    gray_tensor = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0).unsqueeze(0).contiguous()
    return _normalize_rgb(rgb_tensor, config.norm), gray_tensor


def preprocess_afloc_image_from_path(
    path: Path | str,
    config: AFLocImagePreprocessing,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"missing image: {resolved}")

    try:
        with Image.open(resolved) as handle:
            image = np.asarray(handle.convert("L"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - mirrors dataset image-loading failure handling
        raise ValueError(f"Failed to load image: {resolved}") from exc
    return preprocess_afloc_image_array(image, config)
