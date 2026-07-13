from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from anaprior.features.afloc_mrsg_encoder import FrozenAFLocMRSGEncoder
from anaprior.models.afloc_mrsg import AFLocMRSG, MRSGConfig
from anaprior.models.afloc_mrsg.contracts import MRSGOutput
from anaprior.models.afloc_mrsg.diagnostics import PhaseGateDecision
from tests.mrsg_test_utils import (
    FakeAFLoc,
    healthy_grounding_diagnostics,
    write_descriptions,
    write_failed_checkpoint,
    write_tiny_manifest,
)


def trainer_module():
    from anaprior.train import train_afloc_mrsg as module

    return module


def _write_images_for_manifest(manifest_path: Path) -> None:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        Image.new(
            "RGB",
            (32, 32),
            color=(48 + index * 20, 96 + index * 10, 144 + index * 5),
        ).save(row["image_path"])


def _prepare_manifests(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train_manifest = write_tiny_manifest(tmp_path, "train")
    valid_manifest = write_tiny_manifest(tmp_path, "valid")
    _write_images_for_manifest(train_manifest)
    _write_images_for_manifest(valid_manifest)
    descriptions_json = write_descriptions(tmp_path)
    protocol_manifest = tmp_path / "mrsg_protocol_manifest.json"
    protocol_manifest.write_text(
        json.dumps(
            {
                "uses_mscxr_annotations": False,
                "uses_spatial_annotations": False,
                "uses_dcem": False,
                "sanity": {"mscxr_overlap": 0, "train_valid_subject_overlap": 0},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return train_manifest, valid_manifest, descriptions_json, protocol_manifest


def _tiny_model_config() -> MRSGConfig:
    return MRSGConfig(
        feature_dim=16,
        text_dim=24,
        num_heads=4,
        focal_slots=3,
        topk_fraction=0.25,
        route_temperature=1.0,
    )


def _load_checkpoint(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _module_changed(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    module_name: str,
) -> bool:
    prefix = f"{module_name}."
    keys = [key for key in before if key.startswith(prefix)]
    assert keys, f"no checkpoint keys found for {module_name}"
    return any(not torch.equal(before[key], after[key]) for key in keys)


def _module_unchanged(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    module_name: str,
) -> bool:
    return not _module_changed(before, after, module_name)


def _build_reference_model(seed: int, checkpoint: dict[str, object]) -> AFLocMRSG:
    torch.manual_seed(seed)
    return AFLocMRSG(
        MRSGConfig(**checkpoint["model_config"]),
        image_channels=checkpoint["image_channels"],
    )


def _epoch_output(
    *,
    batch_size: int,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> MRSGOutput:
    height = width = 2
    query_heatmaps = torch.linspace(
        0.1,
        0.9,
        steps=batch_size * 4 * height * width,
        dtype=torch.float32,
    ).view(batch_size, 4, height, width)
    return MRSGOutput(
        final_heatmap=torch.linspace(
            0.2,
            0.8,
            steps=batch_size * height * width,
            dtype=torch.float32,
        ).view(batch_size, 1, height, width),
        query_heatmaps=query_heatmaps,
        query_route_weights=torch.full((batch_size, 4), 0.25, dtype=torch.float32),
        query_reliability=torch.full((batch_size, 4), 0.5, dtype=torch.float32),
        phrase_patch_logits=torch.zeros(batch_size, 3, height, width, dtype=torch.float32),
        masked_predictions={
            name: torch.zeros(batch_size, 2, height, width, dtype=torch.float32)
            for name in ("l2", "l", "lf")
        },
        source_targets={
            name: torch.zeros(batch_size, 2, height, width, dtype=torch.float32)
            for name in ("l2", "l", "lf")
        },
        patch_mask=torch.ones(batch_size, 1, height, width, dtype=torch.bool),
        query_reconstructed_phrase=torch.ones(batch_size, 4, 6, dtype=torch.float32),
        query_patch_gates=torch.full((batch_size, 4, height, width), 0.5, dtype=torch.float32),
    )


def _epoch_result(
    *,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> dict[str, object]:
    output = _epoch_output(
        batch_size=int(positive_scores.shape[0]),
        positive_scores=positive_scores,
        negative_scores=negative_scores,
    )
    negative_mask = torch.ones_like(negative_scores, dtype=torch.bool)
    return {
        "losses": {"total": 0.5, "grounding": 0.2, "teacher": 0.0, "mask": 0.1, "query": 0.2},
        "aggregate_output": output,
        "aggregate_positive_scores": positive_scores,
        "aggregate_negative_scores": negative_scores,
        "aggregate_negative_mask": negative_mask,
        "aggregate_teacher_target": None,
        "last_output": _epoch_output(
            batch_size=1,
            positive_scores=positive_scores[-1:],
            negative_scores=negative_scores[-1:],
        ),
        "last_positive_scores": positive_scores[-1:],
        "last_negative_scores": negative_scores[-1:],
        "last_teacher_target": None,
        "num_skipped_examples": 0,
        "num_optimization_steps": 1,
        "per_finding_diagnostics": {"Pneumothorax": {"count": int(positive_scores.shape[0])}},
    }


def test_locality_phase_trains_only_pyramid_and_mask_predictor(tmp_path: Path) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )

    report = module.train_afloc_mrsg(
        phase="locality",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "locality-out",
        afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=13,
        device="cpu",
    )

    checkpoint = _load_checkpoint(Path(report["checkpoint"]))
    initial = _build_reference_model(seed=13, checkpoint=checkpoint)
    initial_state = initial.state_dict()
    trained_state = checkpoint["model_state_dict"]

    assert report["phase"] == "locality"
    assert report["phase_gate"]["passed"] is True
    assert report["afloc_trainable_parameters"] == 0
    assert report["uses_mscxr_annotations"] is False
    assert report["uses_spatial_annotations"] is False
    assert report["uses_dcem"] is False
    assert set(report["four_top_level_loss_weights"]) == {
        "w_ground",
        "w_teacher",
        "w_mask",
        "w_query",
    }
    assert Path(report["report"]).exists()
    assert Path(report["checkpoint"]).name == "mrsg_phase_a.pt"
    assert _module_changed(initial_state, trained_state, "feature_pyramid")
    assert _module_unchanged(initial_state, trained_state, "phrase_router")
    assert _module_unchanged(initial_state, trained_state, "query_bank")
    assert _module_unchanged(initial_state, trained_state, "grounder")
    assert _module_unchanged(initial_state, trained_state, "decoder")


def test_locality_phase_preserves_failing_gate_diagnostics_without_override(tmp_path: Path) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )
    failing = {
        "heatmap_std": 0.0,
        "max_route_utilization": 0.99,
        "query_pairwise_cosine": 1.0,
        "positive_negative_margin": 0.0,
        "teacher_confident_coverage": 0.0,
        "all_module_gradient_norms_finite": 1.0,
        "masked_reconstruction_loss": 1.0,
        "masked_reconstruction_finite": 1.0,
        "untrained_masked_reconstruction_loss": 1.0,
        "untrained_masked_reconstruction_finite": 1.0,
        "locality_reconstruction_ratio": 1.0,
    }

    original_collect = module.collect_mrsg_diagnostics

    def fake_collect(**kwargs):
        collected = original_collect(**kwargs)
        collected.update(failing)
        return collected

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "collect_mrsg_diagnostics", fake_collect)
    try:
        report = module.train_afloc_mrsg(
            phase="locality",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "locality-gate",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=2,
            learning_rate=5.0e-3,
            seed=17,
            device="cpu",
        )
    finally:
        monkeypatch.undo()

    assert report["phase_gate"]["passed"] is False
    assert report["diagnostics"]["heatmap_std"] == pytest.approx(0.0)


def test_grounding_phase_requires_passed_locality_checkpoint_and_updates_grounding_modules(
    tmp_path: Path,
) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )
    encoder = FrozenAFLocMRSGEncoder(FakeAFLoc())
    locality = module.train_afloc_mrsg(
        phase="locality",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-a",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=19,
        device="cpu",
    )

    wrong_phase_checkpoint = tmp_path / "wrong-phase.pt"
    torch.save(
        {"phase": "consistency", "phase_gate": {"passed": True}},
        wrong_phase_checkpoint,
    )
    with pytest.raises(ValueError, match="passed Phase A"):
        module.train_afloc_mrsg(
            phase="grounding",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "bad-phase-b",
            afloc_encoder=encoder,
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            previous_checkpoint=wrong_phase_checkpoint,
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=2,
            device="cpu",
        )

    phase_a_checkpoint = _load_checkpoint(Path(locality["checkpoint"]))
    report = module.train_afloc_mrsg(
        phase="grounding",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-b",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        previous_checkpoint=Path(locality["checkpoint"]),
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=23,
        device="cpu",
    )

    phase_b_checkpoint = _load_checkpoint(Path(report["checkpoint"]))
    before_state = phase_a_checkpoint["model_state_dict"]
    after_state = phase_b_checkpoint["model_state_dict"]

    assert report["phase"] == "grounding"
    assert report["phase_gate"]["passed"] is True
    assert Path(report["checkpoint"]).name == "mrsg_phase_b.pt"
    assert _module_changed(before_state, after_state, "feature_pyramid")
    assert _module_changed(before_state, after_state, "phrase_router")
    assert _module_changed(before_state, after_state, "query_bank")
    assert _module_changed(before_state, after_state, "grounder")
    assert _module_changed(before_state, after_state, "decoder")


def test_consistency_phase_requires_passed_grounding_checkpoint_updates_teacher_and_resumes_reproducibly(
    tmp_path: Path,
) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )
    encoder = FrozenAFLocMRSGEncoder(FakeAFLoc())
    phase_a = module.train_afloc_mrsg(
        phase="locality",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-a",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=19,
        device="cpu",
    )
    phase_b = module.train_afloc_mrsg(
        phase="grounding",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-b",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        previous_checkpoint=Path(phase_a["checkpoint"]),
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=23,
        device="cpu",
    )

    with pytest.raises(ValueError, match="passed Phase B"):
        module.train_afloc_mrsg(
            phase="consistency",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "bad-phase-c",
            afloc_encoder=encoder,
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            previous_checkpoint=write_failed_checkpoint(tmp_path),
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=2,
            device="cpu",
        )

    phase_b_checkpoint = _load_checkpoint(Path(phase_b["checkpoint"]))
    one_epoch = module.train_afloc_mrsg(
        phase="consistency",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-c-one",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        previous_checkpoint=Path(phase_b["checkpoint"]),
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=37,
        device="cpu",
    )
    resumed = module.train_afloc_mrsg(
        phase="consistency",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-c-resumed",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        previous_checkpoint=Path(phase_b["checkpoint"]),
        resume_checkpoint=Path(one_epoch["latest_checkpoint"]),
        model_config=_tiny_model_config(),
        epochs=2,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=37,
        device="cpu",
    )
    full = module.train_afloc_mrsg(
        phase="consistency",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-c-full",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        previous_checkpoint=Path(phase_b["checkpoint"]),
        model_config=_tiny_model_config(),
        epochs=2,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=37,
        device="cpu",
    )

    phase_c_checkpoint = _load_checkpoint(Path(one_epoch["checkpoint"]))
    resumed_checkpoint = _load_checkpoint(Path(resumed["checkpoint"]))
    full_checkpoint = _load_checkpoint(Path(full["checkpoint"]))

    assert one_epoch["phase"] == "consistency"
    assert Path(one_epoch["checkpoint"]).name == "mrsg_phase_c.pt"
    assert "teacher_confident_coverage" in one_epoch["diagnostics"]
    assert _module_changed(
        phase_b_checkpoint["model_state_dict"],
        phase_c_checkpoint["model_state_dict"],
        "feature_pyramid",
    )
    assert _module_changed(
        phase_b_checkpoint["model_state_dict"],
        phase_c_checkpoint["model_state_dict"],
        "phrase_router",
    )
    assert _module_changed(
        phase_b_checkpoint["model_state_dict"],
        phase_c_checkpoint["model_state_dict"],
        "query_bank",
    )
    assert _module_changed(
        phase_b_checkpoint["model_state_dict"],
        phase_c_checkpoint["model_state_dict"],
        "grounder",
    )
    assert _module_changed(
        phase_b_checkpoint["model_state_dict"],
        phase_c_checkpoint["model_state_dict"],
        "decoder",
    )
    assert phase_c_checkpoint["teacher_state_dict"]
    assert any(
        not torch.equal(
            phase_c_checkpoint["teacher_state_dict"][key],
            phase_c_checkpoint["model_state_dict"][key],
        )
        for key in phase_c_checkpoint["teacher_state_dict"]
    )
    assert resumed["completed_epochs"] == 2
    assert full["completed_epochs"] == 2
    assert resumed_checkpoint["best_valid_loss"] == pytest.approx(full_checkpoint["best_valid_loss"])
    assert resumed_checkpoint["phase_gate"]["passed"] == full_checkpoint["phase_gate"]["passed"]
    for key in full_checkpoint["model_state_dict"]:
        assert torch.equal(
            resumed_checkpoint["model_state_dict"][key],
            full_checkpoint["model_state_dict"][key],
        )
    for key in full_checkpoint["teacher_state_dict"]:
        assert torch.equal(
            resumed_checkpoint["teacher_state_dict"][key],
            full_checkpoint["teacher_state_dict"][key],
        )


def test_consistency_phase_preserves_failing_gate_diagnostics_without_override(tmp_path: Path) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )
    encoder = FrozenAFLocMRSGEncoder(FakeAFLoc())
    phase_a = module.train_afloc_mrsg(
        phase="locality",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-a",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=19,
        device="cpu",
    )
    phase_b = module.train_afloc_mrsg(
        phase="grounding",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "phase-b",
        afloc_encoder=encoder,
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        previous_checkpoint=Path(phase_a["checkpoint"]),
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=23,
        device="cpu",
    )

    original_collect = module.collect_mrsg_diagnostics

    def fake_collect(**kwargs):
        collected = original_collect(**kwargs)
        collected.update(
            {
                **healthy_grounding_diagnostics(),
                "all_module_gradient_norms_finite": 1.0,
                "teacher_confident_coverage": 0.0,
                "phase_b_positive_negative_margin": 0.4,
                "phase_b_positive_negative_margin_finite": 1.0,
                "positive_negative_margin": 0.2,
                "consistency_margin_delta_vs_phase_b": -0.2,
            }
        )
        return collected

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "collect_mrsg_diagnostics", fake_collect)
    try:
        report = module.train_afloc_mrsg(
            phase="consistency",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "phase-c-failing-gate",
            afloc_encoder=encoder,
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            previous_checkpoint=Path(phase_b["checkpoint"]),
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=2,
            learning_rate=5.0e-3,
            seed=31,
            device="cpu",
        )
    finally:
        monkeypatch.undo()

    assert report["phase_gate"]["passed"] is False
    assert report["diagnostics"]["teacher_confident_coverage"] == pytest.approx(0.0)


def test_trainer_handles_variable_negative_collate_without_skipping_empty_negative_rows_and_rejects_empty_manifest(
    tmp_path: Path,
) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )
    rows = [
        json.loads(line)
        for line in train_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["negative_phrases"] = []
    train_manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = module.train_afloc_mrsg(
        phase="grounding",
        train_manifest=train_manifest,
        valid_manifest=valid_manifest,
        outdir=tmp_path / "grounding-collate",
        afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
        descriptions_json=descriptions_json,
        protocol_manifest=protocol_manifest,
        previous_checkpoint=module._write_bootstrap_phase_a_checkpoint(
            outdir=tmp_path / "bootstrap-a",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            model_config=_tiny_model_config(),
            seed=41,
            device="cpu",
        ),
        model_config=_tiny_model_config(),
        epochs=1,
        batch_size=2,
        learning_rate=5.0e-3,
        seed=43,
        device="cpu",
    )

    assert report["phase"] == "grounding"
    assert report["num_skipped_examples"] == 0
    assert report["num_optimization_steps"] >= 1

    empty_manifest = tmp_path / "empty.jsonl"
    empty_manifest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no training rows"):
        module.train_afloc_mrsg(
            phase="locality",
            train_manifest=empty_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "empty-out",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=2,
            device="cpu",
        )


def test_teacher_alignment_uses_relative_geometry_not_absolute_equivariance_transform() -> None:
    module = trainer_module()
    batch = {
        "geometry": {
            "horizontal_flip": torch.tensor([True]),
            "crop_top": torch.tensor([0]),
            "crop_left": torch.tensor([2]),
            "crop_height": torch.tensor([4]),
            "crop_width": torch.tensor([4]),
        },
        "equivariance_transform": {
            "horizontal_flip": torch.tensor([False]),
            "crop_top": torch.tensor([0]),
            "crop_left": torch.tensor([2]),
            "crop_height": torch.tensor([4]),
            "crop_width": torch.tensor([4]),
        },
        "relative_equivariance_transform": {
            "horizontal_flip": torch.tensor([True]),
            "crop_top": torch.tensor([0]),
            "crop_left": torch.tensor([0]),
            "crop_height": torch.tensor([4]),
            "crop_width": torch.tensor([4]),
        },
    }

    relative = module.teacher_conf_transform(batch, 0)
    assert relative.horizontal_flip is True
    assert relative.crop_left == 0

    batch["geometry"]["horizontal_flip"] = torch.tensor([False])
    batch["equivariance_transform"]["horizontal_flip"] = torch.tensor([True])
    relative = module.teacher_conf_transform(batch, 0)
    assert relative.horizontal_flip is True


def test_validation_diagnostics_use_all_validation_batches_not_last_batch(tmp_path: Path) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )
    calls = {"count": 0}

    def fake_run_epoch(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _epoch_result(
                positive_scores=torch.tensor([0.2, 0.3], dtype=torch.float32),
                negative_scores=torch.tensor([[0.1], [0.1]], dtype=torch.float32),
            )
        return _epoch_result(
            positive_scores=torch.tensor([0.1, 0.9], dtype=torch.float32),
            negative_scores=torch.tensor([[0.8], [0.2]], dtype=torch.float32),
        )

    def fake_collect(**kwargs):
        output = kwargs["output"]
        positive_scores = kwargs["positive_scores"]
        negative_scores = kwargs["negative_scores"]
        assert output.final_heatmap.shape[0] == 2
        assert torch.equal(positive_scores, torch.tensor([0.1, 0.9], dtype=torch.float32))
        assert torch.equal(negative_scores, torch.tensor([[0.8], [0.2]], dtype=torch.float32))
        return {
            **healthy_grounding_diagnostics(),
            "all_module_gradient_norms_finite": 1.0,
            "positive_negative_margin": 0.0,
            "masked_reconstruction_loss": 0.1,
            "masked_reconstruction_finite": 1.0,
            "untrained_masked_reconstruction_loss": 0.2,
            "untrained_masked_reconstruction_finite": 1.0,
            "locality_reconstruction_ratio": 0.5,
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr(module, "collect_mrsg_diagnostics", fake_collect)
    monkeypatch.setattr(
        module,
        "evaluate_phase_gate",
        lambda phase, diagnostics: PhaseGateDecision(
            phase=phase,
            passed=True,
            reasons=(),
            diagnostics=dict(diagnostics),
        ),
    )
    try:
        module.train_afloc_mrsg(
            phase="locality",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "aggregate-valid",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=1,
            learning_rate=5.0e-3,
            seed=13,
            device="cpu",
        )
    finally:
        monkeypatch.undo()


def test_training_requires_valid_protocol_manifest_and_rejects_malicious_payload(tmp_path: Path) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )

    with pytest.raises(ValueError, match="protocol_manifest"):
        module.train_afloc_mrsg(
            phase="locality",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "missing-manifest",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            descriptions_json=descriptions_json,
            protocol_manifest=None,
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=2,
            device="cpu",
        )

    protocol_manifest.write_text(
        json.dumps(
            {
                "uses_mscxr_annotations": True,
                "uses_spatial_annotations": False,
                "uses_dcem": False,
                "sanity": {"mscxr_overlap": 1, "train_valid_subject_overlap": 0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protocol_manifest"):
        module.train_afloc_mrsg(
            phase="locality",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "malicious-manifest",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            model_config=_tiny_model_config(),
            epochs=1,
            batch_size=2,
            device="cpu",
        )


def test_trainer_saves_latest_checkpoint_separately_from_best_and_resume_requires_latest(
    tmp_path: Path,
) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )
    epoch_index = {"value": -1}

    def fake_run_epoch(**kwargs):
        if kwargs["optimizer"] is None:
            epoch_index["value"] += 1
            losses = [0.6, 0.5, 0.4]
            output = _epoch_result(
                positive_scores=torch.tensor([0.6], dtype=torch.float32),
                negative_scores=torch.tensor([[0.2]], dtype=torch.float32),
            )
            output["losses"] = {
                "total": losses[epoch_index["value"]],
                "grounding": 0.2,
                "teacher": 0.0,
                "mask": 0.1,
                "query": 0.1,
            }
            return output
        return _epoch_result(
            positive_scores=torch.tensor([0.5], dtype=torch.float32),
            negative_scores=torch.tensor([[0.2]], dtype=torch.float32),
        )

    def fake_collect(**kwargs):
        return {
            **healthy_grounding_diagnostics(),
            "all_module_gradient_norms_finite": 1.0,
            "masked_reconstruction_loss": 0.1,
            "masked_reconstruction_finite": 1.0,
            "untrained_masked_reconstruction_loss": 0.2,
            "untrained_masked_reconstruction_finite": 1.0,
            "locality_reconstruction_ratio": 0.5,
        }

    def fake_gate(phase, diagnostics):
        passed = epoch_index["value"] < 2
        return PhaseGateDecision(
            phase=phase,
            passed=passed,
            reasons=() if passed else ("nonpositive_phrase_margin",),
            diagnostics=dict(diagnostics),
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "_run_epoch", fake_run_epoch)
    monkeypatch.setattr(module, "_untrained_locality_baseline", lambda **kwargs: 0.2)
    monkeypatch.setattr(module, "collect_mrsg_diagnostics", fake_collect)
    monkeypatch.setattr(module, "evaluate_phase_gate", fake_gate)
    try:
        report = module.train_afloc_mrsg(
            phase="locality",
            train_manifest=train_manifest,
            valid_manifest=valid_manifest,
            outdir=tmp_path / "latest-vs-best",
            afloc_encoder=FrozenAFLocMRSGEncoder(FakeAFLoc()),
            descriptions_json=descriptions_json,
            protocol_manifest=protocol_manifest,
            model_config=_tiny_model_config(),
            epochs=3,
            batch_size=1,
            learning_rate=5.0e-3,
            seed=13,
            device="cpu",
        )
    finally:
        monkeypatch.undo()

    best_checkpoint = _load_checkpoint(Path(report["checkpoint"]))
    latest_checkpoint = _load_checkpoint(Path(report["latest_checkpoint"]))

    assert report["completed_epochs"] == 3
    assert best_checkpoint["completed_epochs"] == 2
    assert latest_checkpoint["completed_epochs"] == 3
    assert latest_checkpoint["optimizer_state_dict"]
    assert "python_random_state" in latest_checkpoint
    assert "numpy_random_state" in latest_checkpoint
    assert "torch_rng_state" in latest_checkpoint

    with pytest.raises(ValueError, match="latest"):
        module._load_resume_checkpoint("locality", Path(report["checkpoint"]))

    resumed_payload = module._load_resume_checkpoint("locality", Path(report["latest_checkpoint"]))
    assert resumed_payload["completed_epochs"] == 3


def test_cli_runs_tiny_locality_training_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = trainer_module()
    train_manifest, valid_manifest, descriptions_json, protocol_manifest = _prepare_manifests(
        tmp_path
    )

    monkeypatch.setattr(
        module.FrozenAFLocMRSGEncoder,
        "from_checkpoint",
        classmethod(
            lambda cls, ckpt_path, device="cpu": FrozenAFLocMRSGEncoder(FakeAFLoc())
        ),
    )

    exit_code = module.main(
        [
            "--phase",
            "locality",
            "--train-manifest",
            str(train_manifest),
            "--valid-manifest",
            str(valid_manifest),
            "--outdir",
            str(tmp_path / "cli-out"),
            "--afloc-checkpoint",
            str(tmp_path / "fake-afloc.ckpt"),
            "--protocol-manifest",
            str(protocol_manifest),
            "--descriptions-json",
            str(descriptions_json),
            "--feature-dim",
            "16",
            "--text-dim",
            "24",
            "--num-heads",
            "4",
            "--focal-slots",
            "3",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "53",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["phase"] == "locality"
    assert Path(payload["report"]).exists()
