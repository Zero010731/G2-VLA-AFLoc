"""Pool AFLoc local features over Chest ImaGenome anatomical regions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RegionBox:
    """A region box in image coordinates."""

    name: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class RegionPoolingResult:
    """Pooled region features and bookkeeping."""

    features: torch.Tensor
    valid_mask: torch.Tensor
    cell_counts: torch.Tensor
    region_names: list[str]


def _normalize_feature_map(local_features: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if local_features.ndim == 3:
        return local_features.unsqueeze(0), True
    if local_features.ndim == 4:
        return local_features, False
    raise ValueError("local_features must have shape [C,H,W] or [B,C,H,W]")


def _region_mask(
    box: RegionBox,
    feature_height: int,
    feature_width: int,
    image_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    image_height, image_width = image_size
    x1 = max(0.0, min(float(image_width), float(box.x1)))
    x2 = max(0.0, min(float(image_width), float(box.x2)))
    y1 = max(0.0, min(float(image_height), float(box.y1)))
    y2 = max(0.0, min(float(image_height), float(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return torch.zeros((feature_height, feature_width), dtype=torch.bool, device=device)

    y_centers = (torch.arange(feature_height, device=device, dtype=torch.float32) + 0.5) * (
        image_height / feature_height
    )
    x_centers = (torch.arange(feature_width, device=device, dtype=torch.float32) + 0.5) * (
        image_width / feature_width
    )
    y_mask = (y_centers >= y1) & (y_centers < y2)
    x_mask = (x_centers >= x1) & (x_centers < x2)
    return y_mask[:, None] & x_mask[None, :]


def pool_region_features(
    local_features: torch.Tensor,
    boxes: list[RegionBox],
    image_size: tuple[int, int] = (224, 224),
) -> RegionPoolingResult:
    """Mean-pool local features over region boxes.

    Args:
        local_features: AFLoc local embeddings with shape [C,H,W] or [B,C,H,W].
        boxes: Anatomical regions in image coordinates.
        image_size: Image size as (height, width), matching the coordinate frame.

    Returns:
        RegionPoolingResult. For batched input, features shape is [B,R,C]. For a
        single image input, features shape is [R,C].
    """

    features, squeezed = _normalize_feature_map(local_features)
    batch, channels, feature_height, feature_width = features.shape
    pooled = []
    valid = []
    counts = []

    flat_features = features.view(batch, channels, feature_height * feature_width)
    for box in boxes:
        mask = _region_mask(
            box,
            feature_height=feature_height,
            feature_width=feature_width,
            image_size=image_size,
            device=features.device,
        ).reshape(-1)
        count = int(mask.sum().item())
        counts.append(count)
        valid.append(count > 0)
        if count == 0:
            pooled.append(torch.zeros((batch, channels), dtype=features.dtype, device=features.device))
            continue
        pooled.append(flat_features[:, :, mask].mean(dim=-1))

    if pooled:
        pooled_tensor = torch.stack(pooled, dim=1)
    else:
        pooled_tensor = torch.empty((batch, 0, channels), dtype=features.dtype, device=features.device)

    result_features = pooled_tensor.squeeze(0) if squeezed else pooled_tensor
    return RegionPoolingResult(
        features=result_features,
        valid_mask=torch.tensor(valid, dtype=torch.bool, device=features.device),
        cell_counts=torch.tensor(counts, dtype=torch.long, device=features.device),
        region_names=[box.name for box in boxes],
    )

