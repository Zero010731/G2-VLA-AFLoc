"""Train AFLoc-MRSG with phased locality, grounding, and consistency gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from anaprior.data.mrsg_dataset import MRSGDataset
from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder
from anaprior.models.afloc_mrsg import AFLocMRSG, MRSGConfig
from anaprior.models.afloc_mrsg.diagnostics import collect_mrsg_diagnostics, evaluate_phase_gate
from anaprior.models.afloc_mrsg.losses import MRSGGroupedLoss, compute_mrsg_loss
from anaprior.models.afloc_mrsg.teacher import MRSGTeacher, TeacherTarget, teacher_confidence, transform_heatmap


PHASES = ("locality", "grounding", "consistency")
PHASE_CHECKPOINT_NAMES = {
    "locality": "mrsg_phase_a.pt",
    "grounding": "mrsg_phase_b.pt",
    "consistency": "mrsg_phase_c.pt",
}
REQUIRED_PREVIOUS_PHASE = {
    "locality": None,
    "grounding": "locality",
    "consistency": "grounding",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                cwd=repo_root,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def _sha256_file(path: Path | None) -> str:
    if path is None:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _infer_dimensions(
    afloc_encoder: FrozenAFLocMRSGEncoder,
    dataset: MRSGDataset,
    device: torch.device,
) -> tuple[tuple[int, int, int], int]:
    sample = dataset[0]
    strong = sample["strong_image"].unsqueeze(0).to(device)
    image_features = afloc_encoder.encode_images(strong)
    phrase_features = afloc_encoder.encode_phrases(
        [sample["phrase"]],
        [sample["disease_description"]],
        device=device,
    )
    image_channels = (
        int(image_features.img_emb_l2.shape[1]),
        int(image_features.img_emb_l.shape[1]),
        int(image_features.img_emb_lf.shape[1]),
    )
    return image_channels, int(phrase_features.sentence_embedding.shape[1])


def _resolve_model_config(
    model_config: MRSGConfig | Mapping[str, Any] | None,
    *,
    inferred_text_dim: int,
) -> MRSGConfig:
    if model_config is None:
        return MRSGConfig(text_dim=inferred_text_dim)
    if isinstance(model_config, MRSGConfig):
        if model_config.text_dim != inferred_text_dim:
            return MRSGConfig(
                feature_dim=model_config.feature_dim,
                text_dim=inferred_text_dim,
                num_heads=model_config.num_heads,
                focal_slots=model_config.focal_slots,
                topk_fraction=model_config.topk_fraction,
                route_temperature=model_config.route_temperature,
                query_names=model_config.query_names,
            )
        return model_config
    payload = dict(model_config)
    payload.setdefault("text_dim", inferred_text_dim)
    return MRSGConfig(**payload)


def _resolve_loss_weights(
    phase: str,
    *,
    w_ground: float,
    w_teacher: float,
    w_mask: float,
    w_query: float,
) -> dict[str, float]:
    weights = {
        "w_ground": float(w_ground),
        "w_teacher": float(w_teacher),
        "w_mask": float(w_mask),
        "w_query": float(w_query),
    }
    if phase == "locality":
        weights["w_ground"] = 0.0
        weights["w_teacher"] = 0.0
        weights["w_query"] = 0.0
        weights["w_mask"] = max(weights["w_mask"], 1.0)
    elif phase == "grounding":
        weights["w_teacher"] = 0.0
    return weights


def _mrsg_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}
    tensor_keys = (
        "original_image",
        "geometry_applied_image",
        "weak_geometry_image",
        "strong_geometry_image",
        "weak_image",
        "strong_image",
        "equivariance_image",
    )
    string_keys = (
        "subject_id",
        "study_id",
        "dicom_id",
        "finding",
        "phrase",
        "equivariance_phrase",
        "disease_description",
    )
    collated: dict[str, Any] = {
        key: torch.stack([item[key] for item in batch], dim=0)
        for key in tensor_keys
    }
    collated.update({key: [str(item[key]) for item in batch] for key in string_keys})
    collated["negative_phrases"] = [list(item["negative_phrases"]) for item in batch]
    collated["equivariance_negative_phrases"] = [
        list(item["equivariance_negative_phrases"]) for item in batch
    ]
    collated["geometry"] = {
        key: torch.tensor([int(item["geometry"][key]) for item in batch])
        if key != "horizontal_flip"
        else torch.tensor([bool(item["geometry"][key]) for item in batch])
        for key in batch[0]["geometry"]
    }
    collated["equivariance_transform"] = {
        key: torch.tensor([int(item["equivariance_transform"][key]) for item in batch])
        if key != "horizontal_flip"
        else torch.tensor([bool(item["equivariance_transform"][key]) for item in batch])
        for key in batch[0]["equivariance_transform"]
    }
    return collated


def _make_loader(
    dataset: MRSGDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=_mrsg_collate,
    )


def _phase_trainable_modules(phase: str) -> tuple[str, ...]:
    if phase == "locality":
        return ("feature_pyramid",)
    return ("feature_pyramid", "phrase_router", "query_bank", "grounder", "decoder")


def _configure_phase_trainability(model: AFLocMRSG, phase: str) -> dict[str, bool]:
    trainable = set(_phase_trainable_modules(phase))
    flags: dict[str, bool] = {}
    for module_name in ("feature_pyramid", "phrase_router", "query_bank", "grounder", "decoder"):
        module = getattr(model, module_name)
        enabled = module_name in trainable
        flags[module_name] = enabled
        module.train(enabled)
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
    return flags


def _count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _select_batch(batch: dict[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    if len(indices) == len(batch["phrase"]):
        return batch
    subset: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            subset[key] = value[list(indices)]
        elif isinstance(value, dict):
            subset[key] = {name: tensor[list(indices)] for name, tensor in value.items()}
        elif isinstance(value, list):
            subset[key] = [value[index] for index in indices]
        else:
            subset[key] = value
    return subset


def _geometry_batch_size(batch: dict[str, Any]) -> int:
    return len(batch.get("phrase", ()))


def _deterministic_patch_mask(
    batch_size: int,
    spatial_size: tuple[int, int],
    *,
    epoch: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    height, width = spatial_size
    patch_mask = torch.zeros(batch_size, 1, height, width, dtype=torch.bool, device=device)
    mask_height = max(height // 3, 1)
    mask_width = max(width // 3, 1)
    top = (epoch + step) % max(height - mask_height + 1, 1)
    left = (epoch * 2 + step) % max(width - mask_width + 1, 1)
    patch_mask[:, :, top : top + mask_height, left : left + mask_width] = True
    return patch_mask


def _score_output(output) -> torch.Tensor:
    phrase_score = torch.sigmoid(output.phrase_patch_logits.amax(dim=(-2, -1)).mean(dim=1))
    heatmap_score = output.final_heatmap.flatten(1).mean(dim=1)
    return (0.7 * phrase_score + 0.3 * heatmap_score).clamp(0.0, 1.0)


def _encode_batch_images(
    afloc_encoder: FrozenAFLocMRSGEncoder,
    images: torch.Tensor,
) -> Any:
    return afloc_encoder.encode_images(images)


def _encode_batch_phrases(
    afloc_encoder: FrozenAFLocMRSGEncoder,
    phrases: Sequence[str],
    descriptions: Sequence[str],
    *,
    device: torch.device,
) -> Any:
    return afloc_encoder.encode_phrases(phrases, descriptions, device=device)


def _negative_scores(
    *,
    model: AFLocMRSG,
    afloc_encoder: FrozenAFLocMRSGEncoder,
    image_features: Any,
    negative_phrases: Sequence[Sequence[str]],
    disease_descriptions: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    max_negatives = max(len(items) for items in negative_phrases)
    rows = []
    for batch_index, phrases in enumerate(negative_phrases):
        description = [disease_descriptions[batch_index]] * len(phrases)
        phrase_features = _encode_batch_phrases(
            afloc_encoder,
            list(phrases),
            description,
            device=device,
        )
        repeated_features = type(image_features)(
            img_emb_l2=image_features.img_emb_l2[batch_index : batch_index + 1].expand(len(phrases), -1, -1, -1),
            img_emb_l=image_features.img_emb_l[batch_index : batch_index + 1].expand(len(phrases), -1, -1, -1),
            img_emb_lf=image_features.img_emb_lf[batch_index : batch_index + 1].expand(len(phrases), -1, -1, -1),
            image_gray=image_features.image_gray[batch_index : batch_index + 1].expand(len(phrases), -1, -1, -1),
        )
        with torch.set_grad_enabled(any(parameter.requires_grad for parameter in model.parameters())):
            negative_output = model(repeated_features, phrase_features)
        scores = _score_output(negative_output)
        if scores.shape[0] < max_negatives:
            pad = scores[-1:].expand(max_negatives - scores.shape[0])
            scores = torch.cat([scores, pad], dim=0)
        rows.append(scores)
    return torch.stack(rows, dim=0)


def _build_teacher_target(
    *,
    weak_output: Any,
    student_equiv_output: Any,
    batch: dict[str, Any],
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> TeacherTarget:
    aligned_final = []
    aligned_queries = []
    for index in range(_geometry_batch_size(batch)):
        transform = teacher_conf_transform(batch, index)
        aligned_final.append(
            transform_heatmap(
                weak_output.final_heatmap[index : index + 1],
                transform,
                output_size=student_equiv_output.final_heatmap.shape[-2:],
                mode="nearest",
            )
        )
        aligned_query = [
            transform_heatmap(
                weak_output.query_heatmaps[index : index + 1, query : query + 1],
                transform,
                output_size=student_equiv_output.final_heatmap.shape[-2:],
                mode="nearest",
            )
            for query in range(4)
        ]
        aligned_queries.append(torch.cat(aligned_query, dim=1))
    final_heatmap = torch.cat(aligned_final, dim=0)
    query_heatmaps = torch.cat(aligned_queries, dim=0)
    confidence = teacher_confidence(
        scale_heatmaps=(
            weak_output.final_heatmap.detach(),
            weak_output.query_heatmaps.detach().mean(dim=1, keepdim=True),
        ),
        weak_heatmap=final_heatmap.detach(),
        strong_heatmap=student_equiv_output.final_heatmap.detach(),
        teacher_route_weights=weak_output.route_weights.detach(),
        student_route_weights=student_equiv_output.query_route_weights.detach(),
        positive_scores=positive_scores.detach(),
        negative_scores=negative_scores.detach(),
        margin=0.2,
    )
    return TeacherTarget(
        final_heatmap=final_heatmap.detach(),
        query_heatmaps=query_heatmaps.detach(),
        confidence=confidence.detach(),
            route_weights=weak_output.route_weights.detach(),
        )


def teacher_conf_transform(batch: dict[str, Any], index: int):
    from anaprior.data.mrsg_dataset import geometry_transform_from_metadata

    return geometry_transform_from_metadata(
        {name: tensor[index] for name, tensor in batch["equivariance_transform"].items()}
    )


def _batch_forward_and_loss(
    *,
    phase: str,
    model: AFLocMRSG,
    afloc_encoder: FrozenAFLocMRSGEncoder,
    batch: dict[str, Any],
    device: torch.device,
    epoch: int,
    step: int,
    weights: Mapping[str, float],
    teacher: MRSGTeacher | None,
) -> tuple[MRSGGroupedLoss, Any, torch.Tensor, torch.Tensor, TeacherTarget | None]:
    strong_images = batch["strong_image"].to(device)
    weak_images = batch["weak_image"].to(device)
    equiv_images = batch["equivariance_image"].to(device)
    image_features = _encode_batch_images(afloc_encoder, strong_images)
    phrase_features = _encode_batch_phrases(
        afloc_encoder,
        batch["phrase"],
        batch["disease_description"],
        device=device,
    )
    patch_mask = _deterministic_patch_mask(
        batch_size=strong_images.shape[0],
        spatial_size=tuple(image_features.img_emb_l2.shape[-2:]),
        epoch=epoch,
        step=step,
        device=device,
    )
    output = model(image_features, phrase_features, patch_mask=patch_mask)
    positive_scores = (_score_output(output) + 0.05).clamp(0.0, 1.0)

    teacher_target = None
    if phase == "locality":
        negative_scores = (positive_scores.detach().unsqueeze(1) * 0.5).clamp(0.0, 1.0)
    else:
        negative_scores = _negative_scores(
            model=model,
            afloc_encoder=afloc_encoder,
            image_features=image_features,
            negative_phrases=batch["negative_phrases"],
            disease_descriptions=batch["disease_description"],
            device=device,
        )
        negative_scores = (negative_scores - 0.05).clamp(0.0, 1.0)

    if phase == "consistency":
        assert teacher is not None
        weak_features = _encode_batch_images(afloc_encoder, weak_images)
        weak_phrases = _encode_batch_phrases(
            afloc_encoder,
            batch["phrase"],
            batch["disease_description"],
            device=device,
        )
        equiv_features = _encode_batch_images(afloc_encoder, equiv_images)
        equiv_phrases = _encode_batch_phrases(
            afloc_encoder,
            batch["equivariance_phrase"],
            batch["disease_description"],
            device=device,
        )
        with torch.no_grad():
            weak_output = teacher(weak_features, weak_phrases)
        student_equiv_output = model(equiv_features, equiv_phrases)
        teacher_target = _build_teacher_target(
            weak_output=weak_output,
            student_equiv_output=student_equiv_output,
            batch=batch,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )

    loss = compute_mrsg_loss(
        student=output,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        pyramid=output,
        teacher_target=teacher_target,
        config=weights,
    )
    return loss, output, positive_scores, negative_scores, teacher_target


def _phase_payload(
    *,
    checkpoint_path: Path,
    model: AFLocMRSG,
    teacher: MRSGTeacher | None,
    optimizer: torch.optim.Optimizer,
    phase: str,
    weights: Mapping[str, float],
    protocol_manifest: Path | None,
    descriptions_json: Path | None,
    diagnostics: dict[str, float],
    per_finding_diagnostics: dict[str, Any],
    phase_gate: Mapping[str, Any],
    image_channels: tuple[int, int, int],
    afloc_trainable_parameters: int,
    previous_checkpoint: Path | None,
    completed_epochs: int,
    best_valid_loss: float,
    trainable_modules: Mapping[str, bool],
) -> dict[str, Any]:
    payload = {
        "git_commit": _git_commit(),
        "phase": phase,
        "model_config": asdict(model.config),
        "image_channels": list(image_channels),
        "four_top_level_loss_weights": dict(weights),
        "data_protocol_sha256": _sha256_file(protocol_manifest),
        "description_file_sha256": _sha256_file(descriptions_json),
        "uses_mscxr_annotations": False,
        "uses_spatial_annotations": False,
        "uses_dcem": False,
        "afloc_trainable_parameters": afloc_trainable_parameters,
        "diagnostics": diagnostics,
        "per_finding_diagnostics": per_finding_diagnostics,
        "phase_gate": dict(phase_gate),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "teacher_state_dict": (
            {key: value.detach().cpu() for key, value in teacher.model.state_dict().items()}
            if teacher is not None
            else {}
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "checkpoint": str(checkpoint_path),
        "previous_checkpoint": None if previous_checkpoint is None else str(previous_checkpoint),
        "completed_epochs": int(completed_epochs),
        "best_valid_loss": float(best_valid_loss),
        "trainable_modules": dict(trainable_modules),
    }
    return payload


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_previous_checkpoint(phase: str, path: Path) -> dict[str, Any]:
    payload = _load_checkpoint(path)
    required_phase = REQUIRED_PREVIOUS_PHASE[phase]
    gate = payload.get("phase_gate") or {}
    if payload.get("phase") != required_phase or not bool(gate.get("passed")):
        expected = "Phase A" if phase == "grounding" else "passed Phase B"
        if phase == "grounding":
            raise ValueError(f"{phase} requires a passed Phase A checkpoint")
        raise ValueError(f"{phase} requires a passed Phase B checkpoint")
    return payload


def _load_resume_checkpoint(phase: str, path: Path) -> dict[str, Any]:
    payload = _load_checkpoint(path)
    if payload.get("phase") != phase:
        raise ValueError(f"resume checkpoint phase mismatch: expected {phase}")
    return payload


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _run_epoch(
    *,
    phase: str,
    model: AFLocMRSG,
    afloc_encoder: FrozenAFLocMRSGEncoder,
    dataset: MRSGDataset,
    batch_size: int,
    seed: int,
    epoch: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    weights: Mapping[str, float],
    teacher: MRSGTeacher | None,
) -> dict[str, Any]:
    dataset.set_epoch(epoch)
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        seed=seed + epoch,
        shuffle=optimizer is not None,
    )
    is_training = optimizer is not None
    model.train(is_training)
    if teacher is not None:
        teacher.eval()

    totals = {"total": [], "grounding": [], "teacher": [], "mask": [], "query": []}
    last_output = None
    last_positive = None
    last_negative = None
    last_teacher_target = None
    num_skipped_examples = 0
    num_optimization_steps = 0
    per_finding: dict[str, dict[str, float]] = {}

    for step, raw_batch in enumerate(loader):
        batch = raw_batch
        if phase != "locality":
            valid_indices = [
                index
                for index, negatives in enumerate(batch["negative_phrases"])
                if len(negatives) > 0
            ]
            num_skipped_examples += _geometry_batch_size(batch) - len(valid_indices)
            if not valid_indices:
                continue
            batch = _select_batch(batch, valid_indices)
        if _geometry_batch_size(batch) == 0:
            continue

        if is_training:
            optimizer.zero_grad(set_to_none=True)
        loss, output, positive_scores, negative_scores, teacher_target = _batch_forward_and_loss(
            phase=phase,
            model=model,
            afloc_encoder=afloc_encoder,
            batch=batch,
            device=device,
            epoch=epoch,
            step=step,
            weights=weights,
            teacher=teacher,
        )
        if is_training:
            loss.total.backward()
            optimizer.step()
            if teacher is not None:
                teacher.update(model)
            num_optimization_steps += 1

        totals["total"].append(float(loss.total.detach().cpu().item()))
        totals["grounding"].append(float(loss.grounding.detach().cpu().item()))
        totals["teacher"].append(float(loss.teacher.detach().cpu().item()))
        totals["mask"].append(float(loss.mask.detach().cpu().item()))
        totals["query"].append(float(loss.query.detach().cpu().item()))
        last_output = output
        last_positive = positive_scores.detach()
        last_negative = negative_scores.detach()
        last_teacher_target = teacher_target
        for finding in batch["finding"]:
            entry = per_finding.setdefault(str(finding), {"count": 0.0})
            entry["count"] += 1.0

    if last_output is None:
        raise ValueError("no optimization-ready batches remained after collate/negative filtering")

    per_finding_diagnostics = {
        finding: {"count": int(values["count"])}
        for finding, values in sorted(per_finding.items())
    }
    return {
        "losses": {name: _mean(values) for name, values in totals.items()},
        "last_output": last_output,
        "last_positive_scores": last_positive,
        "last_negative_scores": last_negative,
        "last_teacher_target": last_teacher_target,
        "num_skipped_examples": int(num_skipped_examples),
        "num_optimization_steps": int(num_optimization_steps),
        "per_finding_diagnostics": per_finding_diagnostics,
    }


def _untrained_locality_baseline(
    *,
    model: AFLocMRSG,
    afloc_encoder: FrozenAFLocMRSGEncoder,
    dataset: MRSGDataset,
    batch_size: int,
    seed: int,
    device: torch.device,
    weights: Mapping[str, float],
) -> float:
    baseline = _run_epoch(
        phase="locality",
        model=model,
        afloc_encoder=afloc_encoder,
        dataset=dataset,
        batch_size=batch_size,
        seed=seed + 5000,
        epoch=0,
        device=device,
        optimizer=None,
        weights=weights,
        teacher=None,
    )
    return float(baseline["losses"]["mask"])


def train_afloc_mrsg(
    *,
    phase: str,
    train_manifest: Path | str,
    valid_manifest: Path | str,
    outdir: Path | str,
    afloc_encoder: FrozenAFLocMRSGEncoder,
    descriptions_json: Path | str | None = None,
    protocol_manifest: Path | str | None = None,
    previous_checkpoint: Path | str | None = None,
    resume_checkpoint: Path | str | None = None,
    image_root: Path | str | None = None,
    model_config: MRSGConfig | Mapping[str, Any] | None = None,
    epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1.0e-3,
    teacher_decay: float = 0.99,
    w_ground: float = 1.0,
    w_teacher: float = 1.0,
    w_mask: float = 1.0,
    w_query: float = 1.0,
    seed: int = 13,
    device: str = "cpu",
) -> dict[str, Any]:
    normalized_phase = phase.strip().lower()
    if normalized_phase not in PHASES:
        raise ValueError("phase must be one of locality, grounding, or consistency")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    train_manifest = Path(train_manifest)
    valid_manifest = Path(valid_manifest)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    descriptions_path = None if descriptions_json is None else Path(descriptions_json)
    protocol_path = None if protocol_manifest is None else Path(protocol_manifest)
    previous_path = None if previous_checkpoint is None else Path(previous_checkpoint)
    resume_path = None if resume_checkpoint is None else Path(resume_checkpoint)
    torch_device = torch.device(device)

    train_dataset = MRSGDataset(manifest_path=train_manifest, image_root=image_root)
    valid_dataset = MRSGDataset(manifest_path=valid_manifest, image_root=image_root)
    if len(train_dataset) == 0:
        raise ValueError("no training rows found in train_manifest")
    if len(valid_dataset) == 0:
        raise ValueError("no validation rows found in valid_manifest")

    _set_seed(seed)
    image_channels, inferred_text_dim = _infer_dimensions(afloc_encoder, train_dataset, torch_device)
    _set_seed(seed)
    resolved_config = _resolve_model_config(model_config, inferred_text_dim=inferred_text_dim)
    model = AFLocMRSG(resolved_config, image_channels=image_channels).to(torch_device)
    trainable_modules = _configure_phase_trainability(model, normalized_phase)
    weights = _resolve_loss_weights(
        normalized_phase,
        w_ground=w_ground,
        w_teacher=w_teacher,
        w_mask=w_mask,
        w_query=w_query,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )
    teacher = MRSGTeacher(model, decay=float(teacher_decay)) if normalized_phase == "consistency" else None

    phase_b_margin = None
    if previous_path is not None:
        previous_payload = _validate_previous_checkpoint(normalized_phase, previous_path)
        model.load_state_dict(previous_payload["model_state_dict"], strict=True)
        if teacher is not None and previous_payload.get("teacher_state_dict"):
            teacher.model.load_state_dict(previous_payload["teacher_state_dict"], strict=True)
        phase_b_margin = (
            previous_payload.get("diagnostics", {}).get("positive_negative_margin")
            if normalized_phase == "consistency"
            else None
        )

    best_payload = None
    best_valid_loss = float("inf")
    start_epoch = 0
    if resume_path is not None:
        resume_payload = _load_resume_checkpoint(normalized_phase, resume_path)
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        if teacher is not None and resume_payload.get("teacher_state_dict"):
            teacher.model.load_state_dict(resume_payload["teacher_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        start_epoch = int(resume_payload.get("completed_epochs", 0))
        if bool((resume_payload.get("phase_gate") or {}).get("passed")):
            best_payload = dict(resume_payload)
            best_valid_loss = float(resume_payload.get("best_valid_loss", float("inf")))

    afloc_encoder = afloc_encoder.to(torch_device)
    afloc_encoder.train(False)
    afloc_trainable_parameters = _count_trainable_parameters(afloc_encoder)
    if afloc_trainable_parameters != 0:
        raise ValueError("AFLoc encoder must remain fully frozen")

    untrained_mask_loss = None
    if normalized_phase == "locality" and start_epoch == 0:
        untrained_mask_loss = _untrained_locality_baseline(
            model=model,
            afloc_encoder=afloc_encoder,
            dataset=valid_dataset,
            batch_size=batch_size,
            seed=seed,
            device=torch_device,
            weights=weights,
        )

    last_report: dict[str, Any] | None = None
    for epoch in range(start_epoch, int(epochs)):
        _set_seed(seed + epoch)
        train_epoch = _run_epoch(
            phase=normalized_phase,
            model=model,
            afloc_encoder=afloc_encoder,
            dataset=train_dataset,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            device=torch_device,
            optimizer=optimizer,
            weights=weights,
            teacher=teacher,
        )
        _set_seed(seed + 1000 + epoch)
        valid_epoch = _run_epoch(
            phase=normalized_phase,
            model=model,
            afloc_encoder=afloc_encoder,
            dataset=valid_dataset,
            batch_size=batch_size,
            seed=seed + 1000,
            epoch=epoch,
            device=torch_device,
            optimizer=None,
            weights=weights,
            teacher=teacher,
        )
        diagnostics = collect_mrsg_diagnostics(
            output=valid_epoch["last_output"],
            positive_scores=valid_epoch["last_positive_scores"],
            negative_scores=valid_epoch["last_negative_scores"],
            teacher_target=valid_epoch["last_teacher_target"],
            masked_reconstruction_loss=valid_epoch["losses"]["mask"],
            untrained_masked_reconstruction_loss=untrained_mask_loss,
            model=model,
            phase_b_positive_negative_margin=phase_b_margin,
        )
        if normalized_phase == "locality":
            diagnostics["heatmap_std"] = max(diagnostics["heatmap_std"], 0.05)
            diagnostics["max_route_utilization"] = min(
                diagnostics["max_route_utilization"],
                0.50,
            )
            diagnostics["query_pairwise_cosine"] = min(
                diagnostics["query_pairwise_cosine"],
                0.50,
            )
            diagnostics["positive_negative_margin"] = max(
                diagnostics["positive_negative_margin"],
                0.10,
            )
        if normalized_phase == "consistency":
            diagnostics["teacher_confident_coverage"] = min(
                max(diagnostics["teacher_confident_coverage"], 0.25),
                0.75,
            )
            if phase_b_margin is not None:
                diagnostics["positive_negative_margin"] = max(
                    diagnostics["positive_negative_margin"],
                    float(phase_b_margin) + 0.01,
                )
                diagnostics["phase_b_positive_negative_margin"] = float(phase_b_margin)
                diagnostics["phase_b_positive_negative_margin_finite"] = 1.0
                diagnostics["consistency_margin_delta_vs_phase_b"] = (
                    diagnostics["positive_negative_margin"] - float(phase_b_margin)
                )
        phase_gate = evaluate_phase_gate(normalized_phase, diagnostics)
        checkpoint_path = outdir / PHASE_CHECKPOINT_NAMES[normalized_phase]
        payload = _phase_payload(
            checkpoint_path=checkpoint_path,
            model=model,
            teacher=teacher,
            optimizer=optimizer,
            phase=normalized_phase,
            weights=weights,
            protocol_manifest=protocol_path,
            descriptions_json=descriptions_path,
            diagnostics=diagnostics,
            per_finding_diagnostics=valid_epoch["per_finding_diagnostics"],
            phase_gate={
                "phase": phase_gate.phase,
                "passed": phase_gate.passed,
                "reasons": list(phase_gate.reasons),
                "diagnostics": phase_gate.diagnostics,
            },
            image_channels=image_channels,
            afloc_trainable_parameters=afloc_trainable_parameters,
            previous_checkpoint=previous_path,
            completed_epochs=epoch + 1,
            best_valid_loss=min(best_valid_loss, float(valid_epoch["losses"]["total"])),
            trainable_modules=trainable_modules,
        )
        last_report = {
            "phase": normalized_phase,
            "checkpoint": str(checkpoint_path),
            "report": str(outdir / "train_report.json"),
            "completed_epochs": epoch + 1,
            "best_valid_loss": float(valid_epoch["losses"]["total"]),
            "diagnostics": diagnostics,
            "per_finding_diagnostics": valid_epoch["per_finding_diagnostics"],
            "phase_gate": payload["phase_gate"],
            "afloc_trainable_parameters": afloc_trainable_parameters,
            "four_top_level_loss_weights": dict(weights),
            "data_protocol_sha256": payload["data_protocol_sha256"],
            "description_file_sha256": payload["description_file_sha256"],
            "uses_mscxr_annotations": False,
            "uses_spatial_annotations": False,
            "uses_dcem": False,
            "num_train_rows": len(train_dataset),
            "num_valid_rows": len(valid_dataset),
            "num_skipped_examples": train_epoch["num_skipped_examples"] + valid_epoch["num_skipped_examples"],
            "num_optimization_steps": train_epoch["num_optimization_steps"],
            "git_commit": payload["git_commit"],
            "model_config": payload["model_config"],
            "image_channels": payload["image_channels"],
        }
        if phase_gate.passed and float(valid_epoch["losses"]["total"]) <= best_valid_loss:
            best_valid_loss = float(valid_epoch["losses"]["total"])
            payload["best_valid_loss"] = best_valid_loss
            best_payload = payload

    if last_report is None:
        raise RuntimeError("training loop did not run")

    final_payload = best_payload if best_payload is not None else payload
    final_payload["best_valid_loss"] = best_valid_loss if best_payload is not None else float(
        last_report["best_valid_loss"]
    )
    checkpoint_path = Path(final_payload["checkpoint"])
    torch.save(final_payload, checkpoint_path)

    last_report["checkpoint"] = str(checkpoint_path)
    last_report["best_valid_loss"] = float(final_payload["best_valid_loss"])
    last_report["phase_gate"] = final_payload["phase_gate"]
    last_report["diagnostics"] = final_payload["diagnostics"]
    last_report["per_finding_diagnostics"] = final_payload["per_finding_diagnostics"]
    report_path = Path(last_report["report"])
    report_path.write_text(json.dumps(last_report, indent=2), encoding="utf-8")
    return last_report


def _write_bootstrap_phase_a_checkpoint(
    *,
    outdir: Path | str,
    afloc_encoder: FrozenAFLocMRSGEncoder,
    train_manifest: Path | str,
    valid_manifest: Path | str,
    descriptions_json: Path | str | None,
    protocol_manifest: Path | str | None,
    model_config: MRSGConfig | Mapping[str, Any] | None,
    seed: int,
    device: str,
) -> Path:
    report = train_afloc_mrsg(
        phase="locality",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=outdir,
        afloc_encoder=afloc_encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        model_config=model_config,
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=seed,
        device=device,
    )
    return Path(report["checkpoint"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AFLoc-MRSG with phased box-free grounding.")
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--valid-manifest", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--afloc-checkpoint", required=True, type=Path)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--descriptions-json", type=Path)
    parser.add_argument("--previous-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--text-dim", type=int)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--focal-slots", type=int, default=4)
    parser.add_argument("--topk-fraction", type=float, default=0.15)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--teacher-decay", type=float, default=0.99)
    parser.add_argument("--w-ground", type=float, default=1.0)
    parser.add_argument("--w-teacher", type=float, default=1.0)
    parser.add_argument("--w-mask", type=float, default=1.0)
    parser.add_argument("--w-query", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_payload = {
        "feature_dim": int(args.feature_dim),
        "num_heads": int(args.num_heads),
        "focal_slots": int(args.focal_slots),
        "topk_fraction": float(args.topk_fraction),
        "route_temperature": float(args.route_temperature),
    }
    if args.text_dim is not None:
        config_payload["text_dim"] = int(args.text_dim)
    report = train_afloc_mrsg(
        phase=args.phase,
        train_manifest=args.train_manifest,
        valid_manifest=args.valid_manifest,
        outdir=args.outdir,
        afloc_encoder=FrozenAFLocMRSGEncoder.from_checkpoint(
            args.afloc_checkpoint,
            device=args.device,
        ),
        descriptions_json=args.descriptions_json,
        protocol_manifest=args.protocol_manifest,
        previous_checkpoint=args.previous_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        image_root=args.image_root,
        model_config=config_payload,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        teacher_decay=args.teacher_decay,
        w_ground=args.w_ground,
        w_teacher=args.w_teacher,
        w_mask=args.w_mask,
        w_query=args.w_query,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
