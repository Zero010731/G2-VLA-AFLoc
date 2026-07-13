from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from anaprior.models.afloc_mrsg.contracts import (
    AFLocFeatureBatch,
    MRSGConfig,
    PhraseFeatureBatch,
)


class FakeAFLoc(nn.Module):
    def __init__(self, text_dim: int = 24) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.text_dim = text_dim

    def image_encoder_forward(self, images: torch.Tensor):
        batch = images.shape[0]
        device = images.device
        return (
            torch.rand(batch, 64, 8, 8, device=device),
            torch.rand(batch, 32, 16, 16, device=device),
            torch.rand(batch, 128, 4, 4, device=device),
            torch.rand(batch, 128, device=device),
        )

    def process_text(self, texts, device):
        batch = len(texts)
        return {
            "caption_ids": torch.ones(batch, 7, dtype=torch.long, device=device),
            "attention_mask": torch.ones(batch, 7, dtype=torch.long, device=device),
            "token_type_ids": torch.zeros(batch, 7, dtype=torch.long, device=device),
        }

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        batch, tokens = caption_ids.shape
        device = caption_ids.device
        return {
            "word_embeddings": torch.rand(batch, self.text_dim, tokens, device=device),
            "sent_embeddings": torch.rand(batch, self.text_dim, 1, device=device),
            "report_embeddings": torch.rand(batch, self.text_dim, device=device),
            "report": [["finding"] for _ in range(batch)],
            "sent_units_report": [["finding"] for _ in range(batch)],
        }


def fake_image_features(batch: int = 2) -> AFLocFeatureBatch:
    return AFLocFeatureBatch(
        img_emb_l2=torch.rand(batch, 32, 16, 16),
        img_emb_l=torch.rand(batch, 64, 8, 8),
        img_emb_lf=torch.rand(batch, 128, 4, 4),
        image_gray=torch.rand(batch, 1, 224, 224),
    )


def fake_phrase_features(
    batch: int = 2,
    tokens: int = 7,
    text_dim: int = 24,
) -> PhraseFeatureBatch:
    return PhraseFeatureBatch(
        word_embeddings=torch.rand(batch, tokens, text_dim),
        sentence_embedding=torch.rand(batch, text_dim),
        disease_description_embedding=torch.rand(batch, text_dim),
        attention_mask=torch.ones(batch, tokens, dtype=torch.bool),
    )


def test_config() -> MRSGConfig:
    return MRSGConfig(feature_dim=16, text_dim=24, num_heads=4, focal_slots=3)


def uniform_routes(batch: int) -> torch.Tensor:
    return torch.full((batch, 4), 0.25)


def clone_parameters(module: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def add_to_parameters(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(value)


def parameters_changed(before: list[torch.Tensor], module: nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def write_tiny_manifest(tmp_path: Path, split: str) -> Path:
    path = tmp_path / f"{split}.jsonl"
    rows = [
        {
            "image_path": str(tmp_path / f"{split}-{index}.jpg"),
            "subject_id": f"{split}-{index}",
            "study_id": f"study-{index}",
            "dicom_id": f"dicom-{index}",
            "phrase": "small right apical pneumothorax",
            "finding": "Pneumothorax",
            "disease_description": "air in the pleural space",
            "negative_phrases": ["left basilar consolidation"],
        }
        for index in range(2)
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return path


def healthy_grounding_diagnostics() -> dict[str, float]:
    return {
        "heatmap_std": 0.08,
        "max_route_utilization": 0.45,
        "query_pairwise_cosine": 0.4,
        "positive_negative_margin": 0.2,
        "teacher_confident_coverage": 0.3,
    }


def write_fake_mimic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "mimic.csv"
    pd.DataFrame(
        [
            {
                "path": "files/p10/p10000001/s50000001/excluded.jpg",
                "report": "right pneumothorax",
            },
            {
                "path": "files/p11/p11000001/s51000001/train.jpg",
                "report": "left basilar opacity",
            },
            {
                "path": "files/p12/p12000001/s52000001/valid.jpg",
                "report": "cardiomegaly",
            },
        ]
    ).to_csv(path, index=False)
    return path


def write_fake_mscxr_json(tmp_path: Path) -> Path:
    path = tmp_path / "mscxr.json"
    path.write_text(
        json.dumps([{"path": "files/p10/p10000001/s50000001/excluded.jpg"}]),
        encoding="utf-8",
    )
    return path


def write_descriptions(tmp_path: Path) -> Path:
    path = tmp_path / "descriptions.json"
    path.write_text(
        json.dumps(
            {
                "Pneumothorax": "air in the pleural space",
                "Lung Opacity": "pulmonary opacity",
            }
        ),
        encoding="utf-8",
    )
    return path


def write_failed_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "failed-phase-b.pt"
    torch.save({"phase": "grounding", "phase_gate": {"passed": False}}, path)
    return path
