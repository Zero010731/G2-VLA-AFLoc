from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch


@dataclass(frozen=True)
class PyramidOutput:
    fused: torch.Tensor
    source_targets: Dict[str, torch.Tensor]
    edge_features: torch.Tensor
    masked_prediction: Dict[str, torch.Tensor]


class LocalityAlignedFeaturePyramid(nn.Module):
    def __init__(
        self,
        source_channels: Sequence[int],
        feature_dim: int,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if len(source_channels) != 3:
            raise ValueError("source_channels must contain exactly three AFLoc scales")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")

        self.feature_dim = feature_dim
        self.source_names = ("l2", "l", "lf")
        self.expected_channels = dict(zip(self.source_names, source_channels))
        self.projections = nn.ModuleDict(
            {
                name: nn.Conv2d(channels, feature_dim, kernel_size=1)
                for name, channels in self.expected_channels.items()
            }
        )
        self.lateral_gates = nn.ModuleDict(
            {
                name: nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1)
                for name in self.source_names
            }
        )
        self.edge_adapter = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.neck = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))
        self.predictors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
                )
                for name in self.source_names
            }
        )

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
            ).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
            ).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor(
                [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]
            ).unsqueeze(0),
            persistent=False,
        )
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def forward(
        self,
        image_features: AFLocFeatureBatch,
        patch_mask: torch.Tensor,
    ) -> PyramidOutput:
        target_size = self._validate_inputs(image_features, patch_mask)
        projected = self._project_sources(image_features, target_size)
        edge_features = self._compute_edge_features(image_features.image_gray.detach(), target_size)
        fused_input = self._fuse_sources(projected, edge_features)
        fused = self._encode_grid(fused_input)
        masked_context = self._encode_grid(self._apply_patch_mask(fused_input, patch_mask))
        source_targets = {
            name: tensor.detach()
            for name, tensor in projected.items()
        }
        masked_prediction = {
            name: predictor(masked_context)
            for name, predictor in self.predictors.items()
        }
        return PyramidOutput(
            fused=fused,
            source_targets=source_targets,
            edge_features=edge_features,
            masked_prediction=masked_prediction,
        )

    def _validate_inputs(
        self,
        image_features: AFLocFeatureBatch,
        patch_mask: torch.Tensor,
    ) -> Tuple[int, int]:
        batch_size = None
        target_size = None
        for name in self.source_names:
            tensor = getattr(image_features, f"img_emb_{name}")
            if tensor.ndim != 4:
                raise ValueError(f"img_emb_{name} must have shape [B,C,H,W]")
            expected_channels = self.expected_channels[name]
            if tensor.shape[1] != expected_channels:
                raise ValueError(
                    f"img_emb_{name} expected {expected_channels} channels but received {tensor.shape[1]}"
                )
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError("All AFLoc scales must share the same batch size")
            if name == "l2":
                target_size = (tensor.shape[-2], tensor.shape[-1])

        if image_features.image_gray.ndim != 4:
            raise ValueError("image_gray must have shape [B,1,H,W]")
        if image_features.image_gray.shape[0] != batch_size:
            raise ValueError("image_gray must share the same batch size as AFLoc scales")
        if image_features.image_gray.shape[1] != 1:
            raise ValueError("image_gray must have a single grayscale channel")

        if patch_mask.ndim != 4:
            raise ValueError("patch_mask must have shape [B,1,H,W]")
        if patch_mask.shape[0] != batch_size or patch_mask.shape[1] != 1:
            raise ValueError("patch_mask must have shape [B,1,H,W]")
        if tuple(patch_mask.shape[-2:]) != target_size:
            raise ValueError("patch_mask must match the l2 patch grid")
        if patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be a boolean tensor")

        return target_size

    def _project_sources(
        self,
        image_features: AFLocFeatureBatch,
        target_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        projected = {}
        for name in self.source_names:
            tensor = getattr(image_features, f"img_emb_{name}").detach()
            projected_tensor = self.projections[name](tensor)
            if tuple(projected_tensor.shape[-2:]) != target_size:
                projected_tensor = F.interpolate(
                    projected_tensor,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected[name] = projected_tensor
        return projected

    def _compute_edge_features(
        self,
        image_gray: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        gx = F.conv2d(image_gray, self.sobel_x, padding=1)
        gy = F.conv2d(image_gray, self.sobel_y, padding=1)
        sobel_magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        laplacian = F.conv2d(image_gray, self.laplacian_kernel, padding=1).abs()
        local_contrast = (image_gray - F.avg_pool2d(image_gray, kernel_size=9, stride=1, padding=4)).abs()
        edge_features = torch.cat((sobel_magnitude, laplacian, local_contrast), dim=1)
        if tuple(edge_features.shape[-2:]) != target_size:
            edge_features = F.interpolate(
                edge_features,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        mean = edge_features.mean(dim=(-2, -1), keepdim=True)
        std = edge_features.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        return (edge_features - mean) / std

    def _fuse_sources(
        self,
        projected: Dict[str, torch.Tensor],
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        stacked = torch.cat([projected[name] for name in self.source_names], dim=1)
        gates = {
            name: torch.sigmoid(layer(stacked))
            for name, layer in self.lateral_gates.items()
        }
        weight_sum = torch.zeros_like(projected["l2"])
        fused = torch.zeros_like(projected["l2"])
        for name in self.source_names:
            weight_sum = weight_sum + gates[name]
            fused = fused + (gates[name] * projected[name])
        fused = fused / weight_sum.clamp_min(1e-6)
        return fused + self.edge_adapter(edge_features)

    def _apply_patch_mask(self, fused_input: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        mask = patch_mask.to(dtype=fused_input.dtype)
        return fused_input * (1.0 - mask) + (self.mask_token * mask)

    def _encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = grid.shape
        tokens = grid.flatten(2).transpose(1, 2)
        encoded = self.neck(tokens)
        return encoded.transpose(1, 2).reshape(batch_size, channels, height, width)


def masked_patch_distillation_loss(
    output: PyramidOutput,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    if patch_mask.ndim != 4 or patch_mask.shape[1] != 1:
        raise ValueError("patch_mask must have shape [B,1,H,W]")

    mask = patch_mask.to(dtype=output.fused.dtype)
    masked_positions = mask.sum()
    if masked_positions.item() == 0:
        return output.fused.new_zeros(())

    losses = []
    for name in ("l2", "l", "lf"):
        prediction = output.masked_prediction[name]
        target = output.source_targets[name]
        if prediction.shape != target.shape:
            raise ValueError(f"masked_prediction[{name}] must match source_targets[{name}]")
        losses.append(
            ((prediction - target).square() * mask).sum()
            / (masked_positions * target.shape[1])
        )
    return torch.stack(losses).mean()
