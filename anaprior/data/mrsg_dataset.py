"""MRSG dataset with geometry-tracked augmentation metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from anaprior.models.afloc_mrsg.teacher import GeometryTransform, transform_phrase


FORBIDDEN_SPATIAL_KEYS = frozenset(
    {"box", "bbox", "mask", "region", "coordinates", "oracle", "dcem"}
)
GEOMETRY_METADATA_KEYS = (
    "horizontal_flip",
    "crop_top",
    "crop_left",
    "crop_height",
    "crop_width",
)


def _normalize_key(key: object) -> str:
    return "".join(ch for ch in str(key).strip().lower() if ch.isalnum())


def _assert_no_forbidden_spatial_fields(value: Any, prefix: str = "") -> None:
    forbidden: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_path = f"{prefix}.{key}" if prefix else str(key)
            if _normalize_key(key) in FORBIDDEN_SPATIAL_KEYS:
                forbidden.append(key_path)
            _assert_no_forbidden_spatial_fields(nested, key_path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_forbidden_spatial_fields(nested, f"{prefix}[{index}]")
    if forbidden:
        unique = ", ".join(sorted(set(forbidden)))
        raise ValueError(f"Manifest contains forbidden spatial supervision fields: {unique}")


def _stable_fraction(seed: int, index: int, label: str, *, epoch: int = 0) -> float:
    digest = hashlib.md5(f"{seed}:{index}:{epoch}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(0xFFFFFFFF)


def _noise_like(
    image: torch.Tensor,
    seed: int,
    index: int,
    label: str,
    std: float,
    *,
    epoch: int = 0,
) -> torch.Tensor:
    if std <= 0.0:
        return image.clone()
    generator = torch.Generator()
    digest = hashlib.md5(f"{seed}:{index}:{epoch}:{label}".encode("utf-8")).hexdigest()
    generator.manual_seed(int(digest[:16], 16))
    noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
    return (image + noise * float(std)).clamp(0.0, 1.0)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        _assert_no_forbidden_spatial_fields(payload)
        rows.append(payload)
    return rows


def _scalar_value(value: object) -> int | bool:
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    return int(value)


def geometry_metadata(transform: GeometryTransform) -> dict[str, int | bool]:
    return {
        "horizontal_flip": bool(transform.horizontal_flip),
        "crop_top": int(transform.crop_top),
        "crop_left": int(transform.crop_left),
        "crop_height": int(transform.crop_height if transform.crop_height is not None else 0),
        "crop_width": int(transform.crop_width if transform.crop_width is not None else 0),
    }


def geometry_transform_from_metadata(metadata: Mapping[str, object]) -> GeometryTransform:
    missing = [key for key in GEOMETRY_METADATA_KEYS if key not in metadata]
    if missing:
        raise KeyError(f"Geometry metadata is missing keys: {', '.join(missing)}")
    return GeometryTransform(
        horizontal_flip=bool(_scalar_value(metadata["horizontal_flip"])),
        crop_top=int(_scalar_value(metadata["crop_top"])),
        crop_left=int(_scalar_value(metadata["crop_left"])),
        crop_height=int(_scalar_value(metadata["crop_height"])),
        crop_width=int(_scalar_value(metadata["crop_width"])),
    )


class MRSGDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        manifest_path: Path | str,
        image_root: Path | str | None = None,
        image_size: tuple[int, int] = (224, 224),
        crop_size: tuple[int, int] | None = None,
        seed: int = 13,
        geometry_prob: float = 1.0,
        horizontal_flip_prob: float = 0.5,
        equivariance_horizontal_flip_prob: float = 0.5,
        weak_noise_std: float = 0.02,
        strong_noise_std: float = 0.08,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_root = None if image_root is None else Path(image_root).resolve()
        self.rows = _load_manifest(self.manifest_path)
        self.image_size = tuple(int(value) for value in image_size)
        self.crop_size = (
            tuple(int(value) for value in crop_size)
            if crop_size is not None
            else self.image_size
        )
        self.seed = int(seed)
        self.geometry_prob = float(geometry_prob)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.equivariance_horizontal_flip_prob = float(equivariance_horizontal_flip_prob)
        self.weak_noise_std = float(weak_noise_std)
        self.strong_noise_std = float(strong_noise_std)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        base = self.manifest_path.parent.resolve() if self.image_root is None else self.image_root
        resolved = (base / path).resolve()
        if self.image_root is not None:
            try:
                resolved.relative_to(self.image_root)
            except ValueError as exc:
                raise ValueError(
                    f"Resolved image path outside image_root: {path}"
                ) from exc
        return resolved

    def _load_image(self, image_path: str) -> torch.Tensor:
        path = self._resolve_image_path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"missing image: {path}")
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB").resize(
                    (self.image_size[1], self.image_size[0]),
                    Image.Resampling.BILINEAR,
                )
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - exercised in tests
            raise ValueError(f"Failed to load image: {path}") from exc
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _geometry_transform(self, index: int) -> GeometryTransform:
        height, width = self.image_size
        crop_height, crop_width = self.crop_size
        max_top = max(height - crop_height, 0)
        max_left = max(width - crop_width, 0)
        use_crop = (
            crop_height < height or crop_width < width
        ) and _stable_fraction(self.seed, index, "geometry", epoch=self.epoch) < self.geometry_prob
        crop_top = 0
        crop_left = max_left if use_crop else 0
        return GeometryTransform(
            horizontal_flip=_stable_fraction(self.seed, index, "flip", epoch=self.epoch)
            < self.horizontal_flip_prob,
            crop_top=crop_top,
            crop_left=crop_left,
            crop_height=crop_height if use_crop else height,
            crop_width=crop_width if use_crop else width,
        )

    def _equivariance_transform(self, index: int, base: GeometryTransform) -> GeometryTransform:
        return GeometryTransform(
            horizontal_flip=_stable_fraction(
                self.seed,
                index,
                "equivariance_flip",
                epoch=self.epoch,
            )
            < self.equivariance_horizontal_flip_prob,
            crop_top=base.crop_top,
            crop_left=base.crop_left,
            crop_height=base.crop_height,
            crop_width=base.crop_width,
        )

    @staticmethod
    def _apply_geometry(image: torch.Tensor, transform: GeometryTransform) -> torch.Tensor:
        top = int(transform.crop_top)
        left = int(transform.crop_left)
        crop_height = image.shape[1] if transform.crop_height is None else int(transform.crop_height)
        crop_width = image.shape[2] if transform.crop_width is None else int(transform.crop_width)
        cropped = image[:, top : top + crop_height, left : left + crop_width]
        if transform.horizontal_flip:
            cropped = cropped.flip(-1)
        return cropped.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = self._load_image(str(row["image_path"]))
        geometry = self._geometry_transform(index)
        equivariance_transform = self._equivariance_transform(index, geometry)
        geometry_applied = self._apply_geometry(image, geometry)
        equivariance_image = self._apply_geometry(image, equivariance_transform)
        weak_image = _noise_like(
            geometry_applied,
            self.seed,
            index,
            "weak",
            self.weak_noise_std,
            epoch=self.epoch,
        )
        strong_image = _noise_like(
            geometry_applied,
            self.seed,
            index,
            "strong",
            self.strong_noise_std,
            epoch=self.epoch,
        )

        phrase = transform_phrase(str(row["phrase"]), geometry)
        negative_phrases = [
            transform_phrase(str(item), geometry) for item in list(row["negative_phrases"])
        ]
        equivariance_phrase = transform_phrase(str(row["phrase"]), equivariance_transform)
        equivariance_negative_phrases = [
            transform_phrase(str(item), equivariance_transform)
            for item in list(row["negative_phrases"])
        ]

        return {
            "original_image": image,
            "geometry_applied_image": geometry_applied,
            "weak_geometry_image": geometry_applied.clone(),
            "strong_geometry_image": geometry_applied.clone(),
            "weak_image": weak_image,
            "strong_image": strong_image,
            "equivariance_image": equivariance_image,
            "geometry": geometry_metadata(geometry),
            "equivariance_transform": geometry_metadata(equivariance_transform),
            "subject_id": str(row["subject_id"]),
            "study_id": str(row["study_id"]),
            "dicom_id": str(row["dicom_id"]),
            "finding": str(row["finding"]),
            "phrase": phrase,
            "negative_phrases": negative_phrases,
            "equivariance_phrase": equivariance_phrase,
            "equivariance_negative_phrases": equivariance_negative_phrases,
            "disease_description": str(row["disease_description"]),
}


__all__ = [
    "MRSGDataset",
    "geometry_metadata",
    "geometry_transform_from_metadata",
]
