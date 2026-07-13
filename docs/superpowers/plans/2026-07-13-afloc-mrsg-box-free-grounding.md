# AFLoc-MRSG Box-Free Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build AFLoc-MRSG as a standalone box-free phrase-grounding model over frozen AFLoc image and text backbones, with four morphology-routed spatial operators and no dependency on DCEM-v3, region predictors, boxes, masks, or spatially supervised external weights.

**Architecture:** AFLoc-MRSG consumes frozen multi-scale AFLoc image features and real AFLoc phrase tokens, restores patch locality with a trainable feature pyramid, routes phrases through focal-set, diffuse, boundary, and structural query operators, and decodes a direct heatmap. Training proceeds through locality warm-up, sparse grounding, and EMA teacher consistency phases using leakage-free MIMIC-CXR image-report pairs.

**Tech Stack:** Python 3.9, PyTorch, torchvision, NumPy, pandas, Pillow/OpenCV, existing AFLoc checkpoint APIs, pytest, existing MS-CXR localization scoring utilities.

## Global Constraints

- Freeze the complete AFLoc image and text backbones; assert that no AFLoc parameter receives gradients.
- Do not read MS-CXR boxes, masks, oracle maps, metric deltas, or oracle-derived properties during training or checkpoint selection.
- Do not use Chest ImaGenome boxes, region maps, scene graphs, or region-predictor outputs.
- Do not load external weights trained with boxes or masks.
- Do not use DCEM-v3 heatmaps as model input, pseudo-label, teacher, fallback, fusion source, or gate.
- Route with AFLoc phrase tokens and disease-description text embeddings, never disease IDs or disease-name conditionals.
- Exclude all MS-CXR subjects, studies, DICOMs, and image paths from MIMIC-CXR train and validation manifests.
- Keep Python 3.9 compatibility: use `from __future__ import annotations`; do not rely on Python 3.10-only runtime behavior.
- Use only PyTorch-native operators in the first implementation; do not add custom CUDA deformable-attention dependencies.
- Expose exactly four top-level loss weights: grounding, teacher, mask, and query.
- Evaluate the raw model without a validation gate.

## Research Self-Audit Decisions

1. Implement focal sparse selection with a straight-through top-k gate, not a custom deformable-attention extension.
2. Derive teacher confidence from agreement signals; do not train a free confidence fallback head.
3. Keep four operator implementations genuinely different in data flow and regularization.
4. Use three implementation gates before final evaluation:
   - Gate A: locality reconstruction works and patch features retain variance;
   - Gate B: phrase grounding rejects disease/location/laterality counterfactuals without map collapse;
   - Gate C: EMA consistency improves stability without reducing phrase margins.
5. Do not run the final MS-CXR test command until all model and training settings are frozen in a manifest.

## File Structure

### New model package

- `anaprior/models/afloc_mrsg/contracts.py`: dataclasses and tensor-shape validation.
- `anaprior/models/afloc_mrsg/feature_pyramid.py`: multi-scale AFLoc fusion, raw-image operators, and masked predictor.
- `anaprior/models/afloc_mrsg/phrase_router.py`: phrase token projection, disease-description projection, and soft routing.
- `anaprior/models/afloc_mrsg/query_operators.py`: four spatial query implementations.
- `anaprior/models/afloc_mrsg/grounding_transformer.py`: sparse phrase-patch gates and bidirectional grounding.
- `anaprior/models/afloc_mrsg/dense_decoder.py`: direct heatmap decoder.
- `anaprior/models/afloc_mrsg/model.py`: integrated standalone model.
- `anaprior/models/afloc_mrsg/losses.py`: four grouped objectives and fixed internal ratios.
- `anaprior/models/afloc_mrsg/teacher.py`: EMA update, view alignment, and deterministic confidence.
- `anaprior/models/afloc_mrsg/diagnostics.py`: anti-collapse and per-query diagnostics.

### New data/training/evaluation files

- `anaprior/data/mrsg_protocol.py`: leakage exclusion and patient-disjoint manifests.
- `anaprior/data/mrsg_phrases.py`: phrase mining and validated counterfactual generation.
- `anaprior/data/mrsg_dataset.py`: image-report dataset and geometry metadata.
- `anaprior/features/afloc_mrsg_encoder.py`: frozen AFLoc multimodal wrapper.
- `anaprior/train/build_mrsg_image_report_cache.py`: manifest builder CLI.
- `anaprior/train/train_afloc_mrsg.py`: three-phase trainer and checkpoint selection.
- `anaprior/eval/eval_mscxr_afloc_mrsg.py`: raw heatmap generation.
- `scripts/run_afloc_mrsg_full_server.sh`: one-command phased runner.

### Modified shared files

- `anaprior/eval/score_mscxr_learned_repair_metrics.py`: recognize `afloc_mrsg*` as standalone learned methods.
- `tests/test_python39_compat.py`: include the new package.
- `tests/mrsg_test_utils.py`: deterministic fake AFLoc, tensor factories, manifest writers, and parameter helpers shared by MRSG tests.

---

### Task 1: Define MRSG Contracts and Configuration

**Files:**
- Create: `anaprior/models/afloc_mrsg/__init__.py`
- Create: `anaprior/models/afloc_mrsg/contracts.py`
- Create: `tests/mrsg_test_utils.py`
- Test: `tests/test_afloc_mrsg_contracts.py`

**Interfaces:**
- Produces: `MRSGConfig`, `AFLocFeatureBatch`, `PhraseFeatureBatch`, `MRSGOutput`.
- Consumed by every later model, loss, trainer, and evaluator task.

- [ ] **Step 1: Write failing contract tests**

```python
from dataclasses import replace

import pytest
import torch

from anaprior.models.afloc_mrsg.contracts import (
    AFLocFeatureBatch,
    MRSGConfig,
    MRSGOutput,
    PhraseFeatureBatch,
)


def test_mrsg_config_has_exactly_four_query_operators() -> None:
    config = MRSGConfig()
    assert config.query_names == ("focal", "diffuse", "boundary", "structural")
    with pytest.raises(ValueError, match="exactly four"):
        replace(config, query_names=("focal", "diffuse"))


def test_mrsg_output_validates_direct_heatmap_shapes() -> None:
    output = MRSGOutput(
        final_heatmap=torch.rand(2, 1, 16, 16),
        query_heatmaps=torch.rand(2, 4, 16, 16),
        query_route_weights=torch.softmax(torch.rand(2, 4), dim=-1),
        query_reliability=torch.rand(2, 4),
        phrase_patch_logits=torch.rand(2, 8, 16, 16),
    )
    output.validate()
    assert output.final_heatmap.shape == (2, 1, 16, 16)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_afloc_mrsg_contracts.py -v`

Expected: FAIL with `ModuleNotFoundError: anaprior.models.afloc_mrsg`.

- [ ] **Step 3: Implement contracts**

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MRSGConfig:
    feature_dim: int = 256
    text_dim: int = 768
    num_heads: int = 8
    focal_slots: int = 4
    topk_fraction: float = 0.15
    route_temperature: float = 1.0
    query_names: tuple[str, ...] = ("focal", "diffuse", "boundary", "structural")

    def __post_init__(self) -> None:
        if len(self.query_names) != 4:
            raise ValueError("MRSG requires exactly four query operators")
        if self.feature_dim % self.num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if not 0.0 < self.topk_fraction <= 1.0:
            raise ValueError("topk_fraction must be in (0,1]")


@dataclass(frozen=True)
class AFLocFeatureBatch:
    img_emb_l2: torch.Tensor
    img_emb_l: torch.Tensor
    img_emb_lf: torch.Tensor
    image_gray: torch.Tensor


@dataclass(frozen=True)
class PhraseFeatureBatch:
    word_embeddings: torch.Tensor
    sentence_embedding: torch.Tensor
    disease_description_embedding: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class MRSGOutput:
    final_heatmap: torch.Tensor
    query_heatmaps: torch.Tensor
    query_route_weights: torch.Tensor
    query_reliability: torch.Tensor
    phrase_patch_logits: torch.Tensor

    def validate(self) -> None:
        batch, channels, height, width = self.final_heatmap.shape
        if channels != 1:
            raise ValueError("final_heatmap must have one channel")
        if self.query_heatmaps.shape != (batch, 4, height, width):
            raise ValueError("query_heatmaps must have shape [B,4,H,W]")
        if self.query_route_weights.shape != (batch, 4):
            raise ValueError("query_route_weights must have shape [B,4]")
```

- [ ] **Step 4: Export contracts from package and run tests**

Create shared deterministic test helpers so later task tests do not rely on undefined fixtures:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from anaprior.models.afloc_mrsg.contracts import AFLocFeatureBatch, MRSGConfig, PhraseFeatureBatch


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


def fake_phrase_features(batch: int = 2, tokens: int = 7, text_dim: int = 24) -> PhraseFeatureBatch:
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
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, module.parameters()))


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
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
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
    pd.DataFrame([
        {"path": "files/p10/p10000001/s50000001/excluded.jpg", "report": "right pneumothorax"},
        {"path": "files/p11/p11000001/s51000001/train.jpg", "report": "left basilar opacity"},
        {"path": "files/p12/p12000001/s52000001/valid.jpg", "report": "cardiomegaly"},
    ]).to_csv(path, index=False)
    return path


def write_fake_mscxr_json(tmp_path: Path) -> Path:
    path = tmp_path / "mscxr.json"
    path.write_text(json.dumps([{"path": "files/p10/p10000001/s50000001/excluded.jpg"}]), encoding="utf-8")
    return path


def write_descriptions(tmp_path: Path) -> Path:
    path = tmp_path / "descriptions.json"
    path.write_text(json.dumps({"Pneumothorax": "air in the pleural space", "Lung Opacity": "pulmonary opacity"}), encoding="utf-8")
    return path


def write_failed_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "failed-phase-b.pt"
    torch.save({"phase": "grounding", "phase_gate": {"passed": False}}, path)
    return path


```

Run: `pytest tests/test_afloc_mrsg_contracts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add anaprior/models/afloc_mrsg tests/mrsg_test_utils.py tests/test_afloc_mrsg_contracts.py
git commit -m "feat: define AFLoc-MRSG model contracts"
```

### Task 2: Build a Leakage-Proof Image-Report Protocol

**Files:**
- Create: `anaprior/data/__init__.py`
- Create: `anaprior/data/mrsg_protocol.py`
- Test: `tests/test_mrsg_protocol.py`

**Interfaces:**
- Produces: `MIMICIdentity`, `build_mscxr_exclusion_set`, `filter_and_split_mimic_rows`, `write_protocol_manifest`.
- Consumes MIMIC CSV rows and MS-CXR JSON/path records.

- [ ] **Step 1: Write failing leakage tests**

```python
import pandas as pd

from anaprior.data.mrsg_protocol import filter_and_split_mimic_rows


def test_protocol_excludes_mscxr_subject_study_and_dicom() -> None:
    rows = pd.DataFrame([
        {"path": "files/p10/p10000001/s50000001/excluded.jpg", "report": "right pneumothorax"},
        {"path": "files/p11/p11000001/s51000001/train.jpg", "report": "left basilar opacity"},
        {"path": "files/p12/p12000001/s52000001/valid.jpg", "report": "cardiomegaly"},
    ])
    result = filter_and_split_mimic_rows(
        rows,
        excluded_subjects={"10000001"},
        excluded_studies={"50000001"},
        excluded_dicoms={"excluded"},
        valid_fraction=0.5,
        seed=13,
    )
    assert "10000001" not in set(result.all_rows["subject_id"])
    assert set(result.train_rows["subject_id"]).isdisjoint(result.valid_rows["subject_id"])
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_protocol.py -v`

Expected: FAIL because `mrsg_protocol` does not exist.

- [ ] **Step 3: Implement identity parsing and exclusion**

```python
@dataclass(frozen=True)
class MIMICIdentity:
    subject_id: str
    study_id: str
    dicom_id: str


def identity_from_path(path: str) -> MIMICIdentity:
    parts = Path(path).parts
    subject = next((part[1:] for part in parts if part.startswith("p") and part[1:].isdigit()), "")
    study = next((part[1:] for part in parts if part.startswith("s") and part[1:].isdigit()), "")
    return MIMICIdentity(subject, study, Path(path).stem)


def patient_split(subject_id: str, valid_fraction: float, seed: int) -> str:
    digest = hashlib.md5(f"{seed}:{subject_id}".encode("utf-8")).hexdigest()
    unit = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "valid" if unit < valid_fraction else "train"
```

Filter if any subject, study, DICOM, or normalized path matches the exclusion manifest. Raise `ValueError` when reports are missing or when train/valid subjects overlap.

- [ ] **Step 4: Add manifest integrity assertions**

The written JSON must include:

```json
{
  "uses_spatial_annotations": false,
  "uses_dcem": false,
  "num_rows_before_exclusion": 3,
  "num_rows_after_exclusion": 2,
  "num_excluded_mscxr_rows": 1,
  "train_subjects_sha256": "64-character lowercase SHA-256 hex digest",
  "valid_subjects_sha256": "64-character lowercase SHA-256 hex digest",
  "exclusion_ids_sha256": "64-character lowercase SHA-256 hex digest",
  "sanity": {
    "train_valid_subject_overlap": 0,
    "mscxr_overlap": 0
  }
}
```

- [ ] **Step 5: Run protocol tests and commit**

Run: `pytest tests/test_mrsg_protocol.py -v`

Expected: PASS.

```bash
git add anaprior/data tests/test_mrsg_protocol.py
git commit -m "feat: add leakage-proof MRSG data protocol"
```

### Task 3: Mine Positive Phrases and Validated Counterfactuals

**Files:**
- Create: `anaprior/data/mrsg_phrases.py`
- Create: `anaprior/configs/mrsg_disease_descriptions.json`
- Test: `tests/test_mrsg_phrases.py`

**Interfaces:**
- Produces: `MinedPhrase`, `mine_report_phrases`, `build_counterfactuals`, `load_disease_descriptions`.
- Output records contain text only; they never contain coordinates, regions, boxes, or masks.

- [ ] **Step 1: Write failing phrase tests**

```python
from anaprior.data.mrsg_phrases import build_counterfactuals, mine_report_phrases


def test_phrase_miner_preserves_location_laterality_and_uncertainty() -> None:
    phrases = mine_report_phrases(
        "There is a small right apical pneumothorax. No left pneumothorax."
    )
    positive = [item for item in phrases if not item.negated]
    assert positive[0].finding == "Pneumothorax"
    assert positive[0].laterality == ("right",)
    assert positive[0].location_terms == ("apical",)


def test_counterfactual_generator_rejects_false_negative_supported_by_report() -> None:
    report = "Small bilateral pleural effusions."
    phrase = mine_report_phrases(report)[0]
    negatives = build_counterfactuals(phrase, full_report=report, max_negatives=4)
    assert all("left pleural effusion" != item.text for item in negatives)
    assert all(item.text.lower() not in report.lower() for item in negatives)
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_phrases.py -v`

Expected: FAIL because phrase mining is missing.

- [ ] **Step 3: Implement a frozen lexical parser**

Use sentence splitting plus fixed term groups for the eight study findings, negation, uncertainty, laterality, and generic location terms. Return text attributes only:

```python
@dataclass(frozen=True)
class MinedPhrase:
    text: str
    finding: str
    laterality: tuple[str, ...]
    location_terms: tuple[str, ...]
    severity_terms: tuple[str, ...]
    uncertain: bool
    negated: bool
```

Counterfactual generation must check the complete report before accepting a disease, location, or laterality replacement.

- [ ] **Step 4: Add disease descriptions as text, not numeric properties**

The JSON maps canonical finding text to one concise external clinical description. It must not contain MS-CXR statistics or oracle-derived locations. Example:

```json
{
  "Pneumothorax": "air in the pleural space that may produce a thin peripheral pleural boundary pattern",
  "Edema": "diffuse or multifocal pulmonary interstitial or airspace fluid-related opacity"
}
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_mrsg_phrases.py -v`

Expected: PASS.

```bash
git add anaprior/data/mrsg_phrases.py anaprior/configs/mrsg_disease_descriptions.json tests/test_mrsg_phrases.py
git commit -m "feat: add box-free phrase and counterfactual mining"
```

### Task 4: Add the Frozen AFLoc Multimodal Encoder Wrapper

**Files:**
- Create: `anaprior/features/afloc_mrsg_encoder.py`
- Test: `tests/test_afloc_mrsg_encoder.py`

**Interfaces:**
- Produces: `FrozenAFLocMRSGEncoder.encode_images`, `encode_phrases`, `forward`.
- Uses `afloc.builder.load_model` and AFLoc `process_text`, `text_encoder_forward`, and `image_encoder_forward`.

- [ ] **Step 1: Write failing mock-model tests**

```python
import torch
from torch import nn

from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder


def test_encoder_freezes_afloc_and_returns_all_feature_levels() -> None:
    fake_afloc = FakeAFLoc()
    encoder = FrozenAFLocMRSGEncoder(fake_afloc)
    features = encoder.encode_images(torch.rand(2, 3, 224, 224))
    assert features.img_emb_l2.ndim == 4
    assert features.img_emb_l.ndim == 4
    assert features.img_emb_lf.ndim == 4
    assert all(not parameter.requires_grad for parameter in fake_afloc.parameters())


def test_encoder_returns_real_word_tokens_and_sentence_embedding() -> None:
    encoded = FrozenAFLocMRSGEncoder(FakeAFLoc()).encode_phrases(
        ["small right apical pneumothorax"],
        ["air in the pleural space"],
        device=torch.device("cpu"),
    )
    assert encoded.word_embeddings.ndim == 3
    assert encoded.sentence_embedding.shape[0] == 1
    assert encoded.attention_mask.dtype == torch.bool
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_afloc_mrsg_encoder.py -v`

Expected: FAIL because wrapper is missing.

- [ ] **Step 3: Implement no-gradient AFLoc encoding**

```python
class FrozenAFLocMRSGEncoder(nn.Module):
    def __init__(self, afloc_model: nn.Module) -> None:
        super().__init__()
        self.afloc = afloc_model.eval()
        for parameter in self.afloc.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> AFLocFeatureBatch:
        img_l, img_l2, img_lf, _ = self.afloc.image_encoder_forward(images)
        gray = images.mean(dim=1, keepdim=True)
        return AFLocFeatureBatch(img_l2, img_l, img_lf, gray)
```

Convert AFLoc word embeddings to `[B,T,768]`, sentence embeddings to `[B,768]`, and masks to boolean `[B,T]`. Assert `.grad is None` for all AFLoc parameters after a downstream backward pass.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/test_afloc_mrsg_encoder.py -v`

Expected: PASS.

```bash
git add anaprior/features/afloc_mrsg_encoder.py tests/test_afloc_mrsg_encoder.py
git commit -m "feat: expose frozen AFLoc features for MRSG"
```

### Task 5: Implement the Locality-Aligned Feature Pyramid

**Files:**
- Create: `anaprior/models/afloc_mrsg/feature_pyramid.py`
- Test: `tests/test_mrsg_feature_pyramid.py`

**Interfaces:**
- Produces: `PyramidOutput(fused, source_targets, edge_features, masked_prediction)`.
- Consumes `AFLocFeatureBatch` and a patch mask.

- [ ] **Step 1: Write failing pyramid tests**

```python
def test_feature_pyramid_fuses_three_scales_and_edge_channels() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16)
    output = module(
        AFLocFeatureBatch(
            img_emb_l2=torch.rand(2, 32, 16, 16),
            img_emb_l=torch.rand(2, 64, 8, 8),
            img_emb_lf=torch.rand(2, 128, 4, 4),
            image_gray=torch.rand(2, 1, 224, 224),
        ),
        patch_mask=torch.zeros(2, 1, 16, 16, dtype=torch.bool),
    )
    assert output.fused.shape == (2, 16, 16, 16)
    assert output.edge_features.shape == (2, 3, 16, 16)


def test_masked_predictor_receives_gradient_but_afloc_targets_do_not() -> None:
    module = LocalityAlignedFeaturePyramid((32, 64, 128), feature_dim=16)
    patch_mask = torch.zeros(2, 1, 16, 16, dtype=torch.bool)
    patch_mask[:, :, 4:12, 4:12] = True
    output = module(fake_image_features(), patch_mask=patch_mask)
    loss = masked_patch_distillation_loss(output)
    loss.backward()
    assert output.masked_prediction.requires_grad
    assert not output.source_targets["l2"].requires_grad
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_mrsg_feature_pyramid.py -v`

Expected: FAIL because pyramid is missing.

- [ ] **Step 3: Implement multi-scale fusion and deterministic image operators**

Define the output contract in the same file:

```python
@dataclass(frozen=True)
class PyramidOutput:
    fused: torch.Tensor
    source_targets: dict[str, torch.Tensor]
    edge_features: torch.Tensor
    masked_prediction: dict[str, torch.Tensor]
```

Use `Conv2d(1x1)`, bilinear resize, learned lateral gates, and two `TransformerEncoderLayer` blocks. Compute three annotation-free raw-image channels:

```python
sobel_magnitude = sqrt(gx.square() + gy.square() + 1e-6)
laplacian = abs(conv2d(gray, laplacian_kernel))
local_contrast = abs(gray - avg_pool2d(gray, 9, stride=1, padding=4))
```

Resize these channels to the fused grid. Do not supervise the boundary query by directly copying them.

- [ ] **Step 4: Implement block masking and latent prediction**

The masked predictor returns one reconstructed tensor per source AFLoc scale. Targets are detached source projections before masking.

- [ ] **Step 5: Run Gate A unit tests and commit**

Run: `pytest tests/test_mrsg_feature_pyramid.py -v`

Expected: PASS with finite reconstruction loss and nonzero gradients only in the trainable pyramid.

```bash
git add anaprior/models/afloc_mrsg/feature_pyramid.py tests/test_mrsg_feature_pyramid.py
git commit -m "feat: add MRSG locality-aligned feature pyramid"
```

### Task 6: Implement Phrase Projection and Soft Morphology Routing

**Files:**
- Create: `anaprior/models/afloc_mrsg/phrase_router.py`
- Test: `tests/test_mrsg_phrase_router.py`

**Interfaces:**
- Produces: `RouterOutput(projected_words, phrase_vector, route_logits, route_weights)`.
- Consumes `PhraseFeatureBatch`.

- [ ] **Step 1: Write failing router tests**

```python
def test_router_uses_text_features_without_category_ids() -> None:
    router = PhraseMorphologyRouter(text_dim=24, feature_dim=16, num_heads=4)
    output = router(fake_phrase_features(batch=3, tokens=7, text_dim=24))
    assert output.projected_words.shape == (3, 7, 16)
    assert output.route_weights.shape == (3, 4)
    assert torch.allclose(output.route_weights.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_router_route_balance_penalizes_single_query_collapse() -> None:
    collapsed = torch.tensor([[0.99, 0.003, 0.003, 0.004]]).repeat(8, 1)
    balanced = torch.full((8, 4), 0.25)
    assert route_balance_loss(collapsed) > route_balance_loss(balanced)
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_phrase_router.py -v`

Expected: FAIL because router is missing.

- [ ] **Step 3: Implement text-only routing**

Define:

```python
@dataclass(frozen=True)
class RouterOutput:
    projected_words: torch.Tensor
    phrase_vector: torch.Tensor
    route_logits: torch.Tensor
    route_weights: torch.Tensor
```

Project word, sentence, and disease-description embeddings. Use sentence/description queries over word tokens, then predict four route logits:

```python
context, _ = self.cross_attention(
    query=condition[:, None, :],
    key=projected_words,
    value=projected_words,
    key_padding_mask=~attention_mask,
)
route_weights = softmax(self.route_head(context[:, 0]) / self.temperature, dim=-1)
```

No disease ID appears in the signature or state dict.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/test_mrsg_phrase_router.py -v`

Expected: PASS.

```bash
git add anaprior/models/afloc_mrsg/phrase_router.py tests/test_mrsg_phrase_router.py
git commit -m "feat: add text-routed MRSG morphology mixture"
```

### Task 7: Implement the Four Spatial Query Operators

**Files:**
- Create: `anaprior/models/afloc_mrsg/query_operators.py`
- Test: `tests/test_mrsg_query_operators.py`

**Interfaces:**
- Produces: `QueryOperatorOutput(features, heatmap_logits, reliability, auxiliary)`.
- Produces: `MorphologyQueryBank.forward(pyramid, edge_features, phrase_vector) -> tuple[QueryOperatorOutput, QueryOperatorOutput, QueryOperatorOutput, QueryOperatorOutput]` in fixed focal/diffuse/boundary/structural order.

- [ ] **Step 1: Write failing shape and behavior tests**

```python
def test_query_bank_returns_four_distinct_operator_outputs() -> None:
    bank = MorphologyQueryBank(feature_dim=16, num_heads=4, focal_slots=3)
    outputs = bank(
        pyramid=torch.rand(2, 16, 12, 12),
        edge_features=torch.rand(2, 3, 12, 12),
        phrase_vector=torch.rand(2, 16),
    )
    assert [item.name for item in outputs] == ["focal", "diffuse", "boundary", "structural"]
    assert all(item.heatmap_logits.shape == (2, 1, 12, 12) for item in outputs)


def test_focal_straight_through_gate_is_sparse_and_differentiable() -> None:
    logits = torch.randn(2, 1, 10, 10, requires_grad=True)
    gate = straight_through_topk_gate(logits, fraction=0.1)
    assert int((gate.detach() > 0.5).sum()) == 20
    gate.sum().backward()
    assert logits.grad is not None
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_query_operators.py -v`

Expected: FAIL because operators are missing.

- [ ] **Step 3: Implement focal-set operator**

Define the shared output before implementing the operators:

```python
@dataclass(frozen=True)
class QueryOperatorOutput:
    name: str
    features: torch.Tensor
    heatmap_logits: torch.Tensor
    reliability: torch.Tensor
    auxiliary: dict[str, torch.Tensor]
```

Use `K` phrase-conditioned lesion slots. Each slot computes patch logits and a straight-through top-k sigmoid gate:

```python
soft = torch.sigmoid(logits)
k = max(1, round(height * width * fraction))
indices = logits.flatten(2).topk(k, dim=-1).indices
hard = torch.zeros_like(logits.flatten(2)).scatter_(2, indices, 1.0).view_as(logits)
gate = hard.detach() - soft.detach() + soft
```

Merge slot probabilities with noisy-OR: `1 - product(1 - slot_probability)`.

- [ ] **Step 4: Implement diffuse operator**

Use non-overlapping window tokens, one global token, window-to-global attention, and a low-pass branch. Do not use top-k or compactness inside this operator.

- [ ] **Step 5: Implement boundary operator**

Concatenate projected semantic patches and edge channels. Apply local 3x3 message passing followed by phrase-conditioned attention. Return neighbor-continuity pairs in `auxiliary` for loss computation.

- [ ] **Step 6: Implement structural operator**

Build left/right paired tokens by horizontally flipping the right half into left-half coordinates. Attend to concatenated `(left, right, abs(left-right))` tokens and broadcast the resulting relation evidence back to both halves.

- [ ] **Step 7: Run query tests and commit**

Run: `pytest tests/test_mrsg_query_operators.py -v`

Expected: PASS, including an assertion that the four operator modules do not share output-head parameters.

```bash
git add anaprior/models/afloc_mrsg/query_operators.py tests/test_mrsg_query_operators.py
git commit -m "feat: add four MRSG spatial query operators"
```

### Task 8: Implement Sparse Phrase-Patch Grounding and Direct Decoding

**Files:**
- Create: `anaprior/models/afloc_mrsg/grounding_transformer.py`
- Create: `anaprior/models/afloc_mrsg/dense_decoder.py`
- Test: `tests/test_mrsg_grounding_decoder.py`

**Interfaces:**
- Produces: `SparseGroundingOutput(query_phrase_patch_logits, query_gated_features, query_reconstructed_phrase, query_patch_gates)`.
- Consumes all four `QueryOperatorOutput.features` tensors and aligns every query independently with the complete phrase-token sequence.
- Produces: `StandaloneDenseDecoder.forward(pyramid, query_outputs, route_weights, query_patch_gates) -> torch.Tensor[B,1,H,W]` logits.

- [ ] **Step 1: Write failing grounding tests**

```python
def test_sparse_grounding_aligns_every_query_with_real_phrase_tokens() -> None:
    module = MultiQuerySparsePhrasePatchGrounder(feature_dim=16, num_heads=4)
    output = module(
        query_features=torch.rand(2, 4, 16, 8, 8),
        word_features=torch.rand(2, 6, 16),
        attention_mask=torch.ones(2, 6, dtype=torch.bool),
    )
    assert output.query_phrase_patch_logits.shape == (2, 4, 6, 8, 8)
    assert output.query_patch_gates.shape == (2, 4, 8, 8)
    assert output.query_reconstructed_phrase.shape == (2, 4, 16)


def test_decoder_output_is_independent_of_dcem_or_base_heatmap() -> None:
    signature = inspect.signature(StandaloneDenseDecoder.forward)
    assert "base_hmap" not in signature.parameters
    assert "dcem" not in signature.parameters
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_grounding_decoder.py -v`

Expected: FAIL because grounding and decoder modules are missing.

- [ ] **Step 3: Implement bidirectional sparse grounding**

Define:

```python
@dataclass(frozen=True)
class SparseGroundingOutput:
    query_phrase_patch_logits: torch.Tensor
    query_gated_features: torch.Tensor
    query_reconstructed_phrase: torch.Tensor
    query_patch_gates: torch.Tensor
```

For each query feature map, compute normalized token-patch dot products, an independent sigmoid patch gate, gated visual context, and phrase reconstruction. Share text projections across queries but use separate query-specific visual projections. Mask padding tokens before token aggregation.

- [ ] **Step 4: Implement direct dense decoder**

Concatenate fused features, four sigmoid query maps weighted by route weights, query reliability channels, and global sparse-grounding gate. Use two residual convolution blocks and a one-channel output head. Return logits without min-max normalization.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_mrsg_grounding_decoder.py -v`

Expected: PASS.

```bash
git add anaprior/models/afloc_mrsg/grounding_transformer.py anaprior/models/afloc_mrsg/dense_decoder.py tests/test_mrsg_grounding_decoder.py
git commit -m "feat: add sparse grounding and standalone decoder"
```

### Task 9: Integrate the Standalone AFLoc-MRSG Model

**Files:**
- Create: `anaprior/models/afloc_mrsg/model.py`
- Modify: `anaprior/models/afloc_mrsg/__init__.py`
- Test: `tests/test_afloc_mrsg_model.py`

**Interfaces:**
- Produces: `AFLocMRSG.forward(image_features, phrase_features, patch_mask=None) -> MRSGOutput`.
- The model consumes AFLoc features, never AFLoc/DCEM heatmaps.

- [ ] **Step 1: Write failing end-to-end model tests**

```python
def test_afloc_mrsg_forward_returns_direct_nonconstant_heatmap() -> None:
    model = AFLocMRSG(test_config(), image_channels=(32, 64, 128))
    output = model(fake_image_features(), fake_phrase_features())
    output.validate()
    assert output.final_heatmap.min() >= 0
    assert output.final_heatmap.max() <= 1
    assert float(output.final_heatmap.std()) > 0


def test_afloc_mrsg_has_no_dcem_region_or_disease_id_inputs() -> None:
    parameters = inspect.signature(AFLocMRSG.forward).parameters
    forbidden = {"base_hmap", "dcem_hmap", "region_maps", "region_scores", "disease_id"}
    assert forbidden.isdisjoint(parameters)
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_afloc_mrsg_model.py -v`

Expected: FAIL because integrated model is missing.

- [ ] **Step 3: Implement the forward graph**

```python
pyramid = self.feature_pyramid(image_features, patch_mask)
router = self.phrase_router(phrase_features)
queries = self.query_bank(pyramid.fused, pyramid.edge_features, router.phrase_vector)
query_tensor = torch.stack([item.features for item in queries], dim=1)
grounding = self.grounder(query_tensor, router.projected_words, phrase_features.attention_mask)
grounded_query_logits = torch.cat([item.heatmap_logits for item in queries], dim=1)
grounded_query_logits = grounded_query_logits + torch.logit(
    grounding.query_patch_gates.clamp(1e-4, 1.0 - 1e-4)
)
logits = self.decoder(
    pyramid.fused,
    grounded_query_logits,
    router.route_weights,
    grounding.query_patch_gates,
)
final = torch.sigmoid(logits)
```

`query_heatmaps` in `MRSGOutput` is `sigmoid(grounded_query_logits)`. `phrase_patch_logits` is the route-weighted aggregation of `query_phrase_patch_logits`. Query reliability is derived separately for every query from phrase-patch support and operator activation variance, not from DCEM agreement.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/test_afloc_mrsg_model.py -v`

Expected: PASS.

```bash
git add anaprior/models/afloc_mrsg tests/test_afloc_mrsg_model.py
git commit -m "feat: integrate standalone AFLoc-MRSG"
```

### Task 10: Implement the Four Loss Groups

**Files:**
- Create: `anaprior/models/afloc_mrsg/losses.py`
- Test: `tests/test_mrsg_losses.py`

**Interfaces:**
- Produces: `MRSGGroupedLoss(total, grounding, teacher, mask, query, diagnostics)`.
- Produces: `compute_mrsg_loss(student, positive_scores, negative_scores, pyramid, teacher_target, config)`.

- [ ] **Step 1: Write failing loss tests**

```python
def test_grounding_loss_rewards_positive_phrase_margin() -> None:
    good = cross_modal_grounding_loss(
        positive_scores=torch.tensor([0.8, 0.7]),
        negative_scores=torch.tensor([[0.1, 0.2], [0.2, 0.3]]),
        reconstructed_phrase=torch.eye(2),
        target_phrase=torch.eye(2),
        margin=0.2,
    )
    bad = cross_modal_grounding_loss(
        positive_scores=torch.tensor([0.2, 0.2]),
        negative_scores=torch.tensor([[0.5, 0.4], [0.6, 0.5]]),
        reconstructed_phrase=torch.zeros(2, 2),
        target_phrase=torch.eye(2),
        margin=0.2,
    )
    assert good < bad


def test_query_regularization_detects_constant_and_identical_maps() -> None:
    collapsed = torch.ones(4, 4, 8, 8) * 0.5
    diverse = torch.rand(4, 4, 8, 8)
    assert query_regularization_loss(collapsed, uniform_routes(4)) > query_regularization_loss(diverse, uniform_routes(4))
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_losses.py -v`

Expected: FAIL because loss functions are missing.

- [ ] **Step 3: Implement fixed internal loss composition**

Define:

```python
@dataclass(frozen=True)
class MRSGGroupedLoss:
    total: torch.Tensor
    grounding: torch.Tensor
    teacher: torch.Tensor
    mask: torch.Tensor
    query: torch.Tensor
    diagnostics: dict[str, float]
```

Use these fixed internal ratios:

```text
L_ground = 1.0 * MIL + 0.5 * cycle + 1.0 * counterfactual_margin
L_teacher = 1.0 * final_map + 0.25 * query_maps
L_mask = mean cosine distance across three AFLoc scales
L_query = 0.25 * route_balance + 0.25 * diversity + 0.25 * noncollapse + 0.25 * operator_structure
```

Only `w_ground`, `w_teacher`, `w_mask`, and `w_query` are CLI/config fields.

- [ ] **Step 4: Add finite-gradient tests for every group**

Each group must produce finite gradients on its intended module and zero gradients on detached AFLoc targets.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_mrsg_losses.py -v`

Expected: PASS.

```bash
git add anaprior/models/afloc_mrsg/losses.py tests/test_mrsg_losses.py
git commit -m "feat: add box-free AFLoc-MRSG objectives"
```

### Task 11: Add EMA Teacher, Geometry Alignment, and Confidence

**Files:**
- Create: `anaprior/models/afloc_mrsg/teacher.py`
- Test: `tests/test_mrsg_teacher.py`

**Interfaces:**
- Produces: `MRSGTeacher`, `transform_heatmap`, `teacher_confidence`, `TeacherTarget`.
- Teacher contains only trainable MRSG modules; AFLoc remains shared and frozen.

- [ ] **Step 1: Write failing EMA and transform tests**

```python
def test_ema_updates_teacher_without_gradients() -> None:
    student = AFLocMRSG(test_config(), image_channels=(32, 64, 128))
    teacher = MRSGTeacher(student, decay=0.9)
    before = clone_parameters(teacher.model)
    add_to_parameters(student, 1.0)
    teacher.update(student)
    assert parameters_changed(before, teacher.model)
    assert all(not p.requires_grad for p in teacher.model.parameters())


def test_horizontal_flip_aligns_heatmap_and_swaps_unilateral_text() -> None:
    transform = GeometryTransform(horizontal_flip=True)
    heatmap = torch.arange(16).view(1, 1, 4, 4).float()
    assert torch.equal(transform_heatmap(heatmap, transform), heatmap.flip(-1))
    assert transform_phrase("small left apical pneumothorax", transform) == "small right apical pneumothorax"
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_teacher.py -v`

Expected: FAIL because teacher utilities are missing.

- [ ] **Step 3: Implement teacher confidence intersection**

Define geometry and teacher targets in this module:

```python
@dataclass(frozen=True)
class GeometryTransform:
    horizontal_flip: bool = False
    crop_top: int = 0
    crop_left: int = 0
    crop_height: int | None = None
    crop_width: int | None = None


@dataclass(frozen=True)
class TeacherTarget:
    final_heatmap: torch.Tensor
    query_heatmaps: torch.Tensor
    confidence: torch.Tensor
    route_weights: torch.Tensor
```

Confidence is the product of four normalized terms:

```python
confidence = scale_agreement * augmentation_agreement * route_stability * phrase_margin_confidence
confidence = confidence.detach().clamp(0.0, 1.0)
```

Zero confidence excludes a patch from teacher loss. No learned confidence head is created.

- [ ] **Step 4: Add false-confidence tests**

Opposing augmentation maps or negative phrase margins must produce lower confident coverage than aligned maps with positive margins.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_mrsg_teacher.py -v`

Expected: PASS.

```bash
git add anaprior/models/afloc_mrsg/teacher.py tests/test_mrsg_teacher.py
git commit -m "feat: add box-free MRSG consistency teacher"
```

### Task 12: Add Dataset, Manifest Builder, and Geometry-Tracked Views

**Files:**
- Create: `anaprior/data/mrsg_dataset.py`
- Create: `anaprior/train/build_mrsg_image_report_cache.py`
- Test: `tests/test_build_mrsg_image_report_cache.py`
- Test: `tests/test_mrsg_dataset.py`

**Interfaces:**
- Produces JSONL train/valid manifests plus `mrsg_protocol_manifest.json`.
- Produces `MRSGDataset` batches with original/weak/strong image tensors, phrases, negatives, descriptions, and `GeometryTransform` metadata.

- [ ] **Step 1: Write failing manifest CLI tests**

```python
def test_manifest_builder_writes_leakage_free_phrase_rows(tmp_path: Path) -> None:
    report = build_mrsg_image_report_cache(
        mimic_csv=write_fake_mimic_csv(tmp_path),
        mscxr_json=write_fake_mscxr_json(tmp_path),
        descriptions_json=write_descriptions(tmp_path),
        outdir=tmp_path / "out",
        valid_fraction=0.2,
        seed=13,
    )
    assert report["uses_spatial_annotations"] is False
    assert report["sanity"]["mscxr_overlap"] == 0
    assert Path(report["train_jsonl"]).exists()
```

- [ ] **Step 2: Run tests to verify RED**

Run: `pytest tests/test_build_mrsg_image_report_cache.py tests/test_mrsg_dataset.py -v`

Expected: FAIL because builder and dataset are missing.

- [ ] **Step 3: Implement JSONL manifest rows**

Each row contains:

```json
{
  "image_path": "files/p11/p11000001/s51000001/01234567-89ab-cdef-0123-456789abcdef.jpg",
  "subject_id": "11000001",
  "study_id": "51000001",
  "dicom_id": "01234567-89ab-cdef-0123-456789abcdef",
  "phrase": "small right apical pneumothorax",
  "finding": "Pneumothorax",
  "disease_description": "air in the pleural space that may produce a thin peripheral pleural boundary pattern",
  "negative_phrases": ["small left pleural effusion", "right apical consolidation"]
}
```

No spatial field is permitted. Validate rows against a forbidden-key set containing `box`, `bbox`, `mask`, `region`, `coordinates`, `oracle`, and `dcem`.

- [ ] **Step 4: Implement geometry-tracked image views**

Dataset transforms return a `GeometryTransform` describing crop bounds, resize, and flip. Weak and strong views use the same geometry when computing photometric consistency; the separate equivariance pair uses a recorded geometry transform.

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_build_mrsg_image_report_cache.py tests/test_mrsg_dataset.py -v`

Expected: PASS.

```bash
git add anaprior/data/mrsg_dataset.py anaprior/train/build_mrsg_image_report_cache.py tests/test_build_mrsg_image_report_cache.py tests/test_mrsg_dataset.py
git commit -m "feat: add MRSG image-report manifests and views"
```

### Task 13: Add Anti-Collapse Diagnostics and Phase Gates

**Files:**
- Create: `anaprior/models/afloc_mrsg/diagnostics.py`
- Test: `tests/test_mrsg_diagnostics.py`

**Interfaces:**
- Produces: `collect_mrsg_diagnostics`, `evaluate_phase_gate`, `PhaseGateDecision`.

- [ ] **Step 1: Write failing diagnostic tests**

```python
def test_phase_gate_rejects_constant_maps_and_single_route_collapse() -> None:
    diagnostics = {
        "heatmap_std": 0.0,
        "max_route_utilization": 0.99,
        "query_pairwise_cosine": 1.0,
        "positive_negative_margin": 0.0,
        "teacher_confident_coverage": 0.2,
    }
    decision = evaluate_phase_gate("grounding", diagnostics)
    assert decision.passed is False
    assert "heatmap_collapse" in decision.reasons


def test_phase_gate_accepts_noncollapsed_grounding_diagnostics() -> None:
    diagnostics = healthy_grounding_diagnostics()
    assert evaluate_phase_gate("grounding", diagnostics).passed is True
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_mrsg_diagnostics.py -v`

Expected: FAIL because diagnostics are missing.

- [ ] **Step 3: Implement diagnostics and fixed sanity thresholds**

Define:

```python
@dataclass(frozen=True)
class PhaseGateDecision:
    phase: str
    passed: bool
    reasons: tuple[str, ...]
    diagnostics: dict[str, float]
```

Use thresholds only for collapse detection, not performance tuning:

```text
heatmap_std > 1e-3
positive_negative_margin > 0
max_route_utilization < 0.90
mean_query_pairwise_cosine < 0.95
teacher_confident_coverage in [0.02, 0.95] during consistency phase
all module gradient norms finite
```

Gate A additionally requires finite masked reconstruction with lower validation loss than an untrained checkpoint. Gate C requires teacher consistency not to reduce the phrase margin below the Phase B checkpoint.

- [ ] **Step 4: Run tests and commit**

Run: `pytest tests/test_mrsg_diagnostics.py -v`

Expected: PASS.

```bash
git add anaprior/models/afloc_mrsg/diagnostics.py tests/test_mrsg_diagnostics.py
git commit -m "feat: add MRSG anti-collapse phase gates"
```

### Task 14: Implement the Three-Phase Trainer

**Files:**
- Create: `anaprior/train/train_afloc_mrsg.py`
- Test: `tests/test_train_afloc_mrsg.py`

**Interfaces:**
- Produces phase checkpoints `mrsg_phase_a.pt`, `mrsg_phase_b.pt`, `mrsg_phase_c.pt` and `train_report.json`.
- CLI: `--phase {locality,grounding,consistency}` plus exact cache/checkpoint arguments.

- [ ] **Step 1: Write failing one-epoch trainer tests**

```python
def test_locality_phase_trains_only_pyramid_and_mask_predictor(tmp_path: Path) -> None:
    report = train_afloc_mrsg(
        phase="locality",
        train_manifest=write_tiny_manifest(tmp_path, "train"),
        valid_manifest=write_tiny_manifest(tmp_path, "valid"),
        outdir=tmp_path / "out",
        afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    assert report["phase"] == "locality"
    assert report["afloc_trainable_parameters"] == 0
    assert report["phase_gate"]["passed"] is True


def test_consistency_phase_requires_passed_grounding_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="passed Phase B"):
        train_afloc_mrsg(
            phase="consistency",
            train_manifest=write_tiny_manifest(tmp_path, "train"),
            valid_manifest=write_tiny_manifest(tmp_path, "valid"),
            outdir=tmp_path / "out",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            previous_checkpoint=write_failed_checkpoint(tmp_path),
            epochs=1,
            batch_size=2,
            device="cpu",
        )
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_train_afloc_mrsg.py -v`

Expected: FAIL because trainer is missing.

- [ ] **Step 3: Implement phase-specific trainability**

```text
locality: feature pyramid + masked predictor
grounding: pyramid + router + grounder + four queries + decoder
consistency: all MRSG student modules + EMA teacher updates
```

AFLoc calls run under `torch.no_grad()`. Checkpoint selection chooses minimum validation total loss among checkpoints that pass the phase gate.

- [ ] **Step 4: Save reproducibility and supervision manifests**

Checkpoint/report fields must include:

```text
git_commit
phase
model_config
four_top_level_loss_weights
data_protocol_sha256
description_file_sha256
uses_mscxr_annotations=false
uses_spatial_annotations=false
uses_dcem=false
afloc_trainable_parameters=0
diagnostics
per_finding_diagnostics
phase_gate
```

- [ ] **Step 5: Run trainer tests and commit**

Run: `pytest tests/test_train_afloc_mrsg.py -v`

Expected: PASS for tiny CPU locality, grounding, and consistency runs.

```bash
git add anaprior/train/train_afloc_mrsg.py tests/test_train_afloc_mrsg.py
git commit -m "feat: add phased AFLoc-MRSG training"
```

### Task 15: Add Raw MS-CXR Inference and Scoring Support

**Files:**
- Create: `anaprior/eval/eval_mscxr_afloc_mrsg.py`
- Modify: `anaprior/eval/score_mscxr_learned_repair_metrics.py`
- Test: `tests/test_eval_mscxr_afloc_mrsg.py`
- Modify: `tests/test_score_mscxr_learned_repair_metrics.py`

**Interfaces:**
- Produces raw `afloc_mrsg/hmaps.npy`, `mrsg_case_diagnostics.json`, and `mrsg_eval_summary.json`.
- Scorer creates `afloc_mrsg_vs_baseline`, `afloc_mrsg_vs_phrase_anatomy_dcem`, and ablation comparisons when methods exist.
- The evaluator accepts `--dataset MS_CXR` for phrase grounding and `--dataset CHEXLOCALIZE` for frozen external class-query evaluation; neither dataset participates in checkpoint selection.

- [ ] **Step 1: Write failing inference tests**

```python
def write_fake_mrsg_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "mrsg.pt"
    torch.save({"model_config": asdict(test_config()), "model_state_dict": {}}, path)
    return path


def fake_case_encoder(row, checkpoint, device):
    return {
        "hmap": np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16),
        "query_route_weights": [0.4, 0.2, 0.2, 0.2],
    }


def test_mscxr_mrsg_eval_uses_phrase_and_image_only(tmp_path: Path) -> None:
    result = build_mscxr_afloc_mrsg_hmaps(
        data_rows=[{
            "case_id": "case-pna",
            "path": str(tmp_path / "case-pna.jpg"),
            "label_text": "right basilar pneumonia",
            "category": "Pneumonia",
        }],
        checkpoint=write_fake_mrsg_checkpoint(tmp_path),
        encode_case=fake_case_encoder,
        device="cpu",
        method_name="afloc_mrsg",
    )
    assert set(result.hmaps) == {"case-pna"}
    assert result.hmaps["case-pna"]["hmap"].shape == (16, 16)
    assert result.summary["uses_dcem"] is False
    assert result.summary["uses_region_predictor"] is False
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_eval_mscxr_afloc_mrsg.py -v`

Expected: FAIL because evaluator is missing.

- [ ] **Step 3: Implement raw evaluator**

The CLI loads MS-CXR directly through `localization.datasets.load_data(dataset=...)` and normalizes only `path`, `label_text`, `category`, and a stable case ID. It must not call `prepare_mscxr_repair_inputs` or load Chest ImaGenome scene graphs. Load only image paths, phrase/finding text, AFLoc checkpoint, and MRSG checkpoint. Reject CLI arguments containing prepared-input NPZs, base heatmaps, region maps, region scores, gates, or lambda overrides.

For `CHEXLOCALIZE`, use the same path/query normalization and preserve the repository's existing validation/test protocol. Write a separate output root and mark it `external_evaluation=true` in the summary.

- [ ] **Step 4: Extend dynamic comparison discovery**

Change the learned-method prefix check to:

```python
if not method.startswith(("dp_msa", "dense_dp_msa", "afloc_mrsg")):
    continue
```

Prefer `afloc_mrsg*_vs_baseline` for standalone primary decisions; keep `vs_phrase_anatomy_dcem` as a secondary replacement comparison.

- [ ] **Step 5: Run evaluator/scorer tests and commit**

Run: `pytest tests/test_eval_mscxr_afloc_mrsg.py tests/test_score_mscxr_learned_repair_metrics.py -v`

Expected: PASS.

```bash
git add anaprior/eval/eval_mscxr_afloc_mrsg.py anaprior/eval/score_mscxr_learned_repair_metrics.py tests/test_eval_mscxr_afloc_mrsg.py tests/test_score_mscxr_learned_repair_metrics.py
git commit -m "feat: evaluate standalone AFLoc-MRSG"
```

### Task 16: Add the Full Server Runner and Frozen Experiment Manifest

**Files:**
- Create: `scripts/run_afloc_mrsg_full_server.sh`
- Create: `tests/test_afloc_mrsg_server_script.py`

**Interfaces:**
- Runs protocol/cache build, Phase A, Phase B, Phase C, raw inference, scoring, and report generation.
- Supports `START_STAGE` for resume without changing scientific settings.

- [ ] **Step 1: Write failing runner-content tests**

```python
def test_mrsg_runner_contains_all_phases_and_forbids_dcem_inputs() -> None:
    script = Path("scripts/run_afloc_mrsg_full_server.sh").read_text(encoding="utf-8")
    assert "build_mrsg_image_report_cache" in script
    assert '--phase "locality"' in script
    assert '--phase "grounding"' in script
    assert '--phase "consistency"' in script
    assert "eval_mscxr_afloc_mrsg" in script
    assert "START_STAGE" in script
    assert "BASE_HMAPS_NPY" not in script
    assert "REGION_SCORE_CSV" not in script
    assert "validation-gate" not in script
```

- [ ] **Step 2: Run test to verify RED**

Run: `pytest tests/test_afloc_mrsg_server_script.py -v`

Expected: FAIL because runner is missing.

- [ ] **Step 3: Implement runner preflight and stages**

Stages:

```text
0 protocol manifest and MS-CXR exclusion audit
1 Phase A locality warm-up
2 Gate A check
3 Phase B sparse grounding
4 Gate B check
5 Phase C dual consistency
6 Gate C and experiment freeze manifest
7 raw MS-CXR heatmaps
8 raw scoring against AFLoc/DCEM baselines
9 frozen CheXlocalize external evaluation
10 report and diagnostics bundle
```

Each gate reads the previous `train_report.json` and exits nonzero if `passed=false`.

- [ ] **Step 4: Freeze the experiment before Stage 7**

Write `frozen_experiment_manifest.json` containing checkpoint hashes, code commit, data manifest hash, model config, loss weights, evaluator arguments, and `test_evaluated=false`. Stage 7 updates only `test_evaluated` and output hashes; it must not modify model settings.

- [ ] **Step 5: Run shell and unit checks, then commit**

Run:

```bash
bash -n scripts/run_afloc_mrsg_full_server.sh
pytest tests/test_afloc_mrsg_server_script.py -v
```

Expected: shell exit 0 and tests PASS.

```bash
git add scripts/run_afloc_mrsg_full_server.sh tests/test_afloc_mrsg_server_script.py
git commit -m "feat: add end-to-end AFLoc-MRSG runner"
```

### Task 17: Python 3.9, Full Verification, and Research Audit

**Files:**
- Modify: `tests/test_python39_compat.py`
- Create: `docs/AFLOC_MRSG_EXPERIMENT_PROTOCOL.md`

**Interfaces:**
- Produces a verified implementation and a concise experiment protocol for server execution.

- [ ] **Step 1: Extend Python 3.9 parsing coverage**

Add every new `anaprior/models/afloc_mrsg`, data, train, feature, and eval file to the existing AST compatibility scan.

- [ ] **Step 2: Run focused model and training tests**

Run:

```bash
pytest \
  tests/test_afloc_mrsg_contracts.py \
  tests/test_mrsg_protocol.py \
  tests/test_mrsg_phrases.py \
  tests/test_afloc_mrsg_encoder.py \
  tests/test_mrsg_feature_pyramid.py \
  tests/test_mrsg_phrase_router.py \
  tests/test_mrsg_query_operators.py \
  tests/test_mrsg_grounding_decoder.py \
  tests/test_afloc_mrsg_model.py \
  tests/test_mrsg_losses.py \
  tests/test_mrsg_teacher.py \
  tests/test_mrsg_diagnostics.py \
  tests/test_build_mrsg_image_report_cache.py \
  tests/test_mrsg_dataset.py \
  tests/test_train_afloc_mrsg.py \
  tests/test_eval_mscxr_afloc_mrsg.py \
  tests/test_afloc_mrsg_server_script.py -v
```

Expected: all PASS.

- [ ] **Step 3: Run full repository verification**

Run:

```bash
pytest -q
python -m compileall -q afloc anaprior
python -m anaprior.train.build_mrsg_image_report_cache --help
python -m anaprior.train.train_afloc_mrsg --help
python -m anaprior.eval.eval_mscxr_afloc_mrsg --help
git diff --check
```

Expected: zero test failures, zero compile errors, three CLI help commands exit 0, and no whitespace errors.

- [ ] **Step 4: Write the protocol document**

Document exact server inputs, phase commands, phase-gate interpretation, prohibited inputs, output paths, and the rule that MS-CXR Stage 7 is run only once after freezing.

- [ ] **Step 5: Perform a final research-integrity audit**

Search the implementation and generated configuration:

```bash
rg -n "region_scores|region_maps|oracle|base_hmap|dcem|validation_gate|bbox|gtmask" \
  anaprior/models/afloc_mrsg \
  anaprior/train/train_afloc_mrsg.py \
  anaprior/data/mrsg_*.py \
  scripts/run_afloc_mrsg_full_server.sh
```

Expected: only explicit forbidden-key validation, documentation, and audit assertions; no model input, target, teacher, or fusion use.

- [ ] **Step 6: Commit verification and protocol**

```bash
git add tests/test_python39_compat.py docs/AFLOC_MRSG_EXPERIMENT_PROTOCOL.md
git commit -m "docs: finalize AFLoc-MRSG experiment protocol"
```

## Execution Order and Review Gates

The tasks must run in order. Reviewer checkpoints are mandatory after:

1. Task 4: confirm no AFLoc gradient path and no spatial labels in data contracts;
2. Task 9: inspect the integrated standalone forward and verify there is no DCEM/base heatmap path;
3. Task 14: inspect Phase A/B/C diagnostics and checkpoint selection;
4. Task 17: run the full research-integrity audit before any server experiment.

Do not combine Tasks 5-11 into one unreviewed change. These tasks define the paper's core method and each needs an independently passing test boundary.
