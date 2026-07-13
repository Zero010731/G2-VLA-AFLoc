from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


SCRIPT_PATH = Path("scripts/run_afloc_mrsg_full_server.sh")


def _script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _python_block(function_name: str) -> str:
    script = _script()
    pattern = re.compile(
        rf"{re.escape(function_name)}\(\) \{{.*?<<'PY'\n(?P<code>.*?)\nPY",
        re.DOTALL,
    )
    match = pattern.search(script)
    assert match is not None, f"missing embedded python block for {function_name}"
    return match.group("code")


def _run_python_block(
    function_name: str,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = dict(env or {})
    return subprocess.run(
        [sys.executable, "-", *args],
        input=_python_block(function_name),
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


def _touch(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _manifest_fixture(tmp_path: Path) -> tuple[list[str], dict[str, str], dict[str, object]]:
    cache_report_path = _touch(
        tmp_path / "cache" / "mrsg_image_report_cache_report.json",
        json.dumps(
            {
                "train_jsonl": str(_touch(tmp_path / "cache" / "train_mrsg.jsonl", '{"row": "train"}\n')),
                "valid_jsonl": str(_touch(tmp_path / "cache" / "valid_mrsg.jsonl", '{"row": "valid"}\n')),
            }
        ),
    )
    protocol_manifest_path = _touch(
        tmp_path / "cache" / "mrsg_protocol_manifest.json",
        json.dumps({"sanity": {"mscxr_overlap": 0, "train_valid_subject_overlap": 0}}),
    )
    afloc_checkpoint = _touch(tmp_path / "weights" / "afloc.ckpt", "checkpoint")
    descriptions_json = _touch(tmp_path / "config" / "descriptions.json", "{}\n")

    phase_payloads: dict[str, dict[str, object]] = {}
    phase_report_paths: list[str] = []
    for phase_name in ("phase_a", "phase_b", "phase_c"):
        phase_dir = tmp_path / phase_name
        report_path = phase_dir / "train_report.json"
        checkpoint_path = _touch(phase_dir / f"{phase_name}.pt", f"{phase_name}-best")
        latest_path = _touch(phase_dir / f"{phase_name}_latest.pt", f"{phase_name}-latest")
        payload = {
            "phase_gate": {"passed": True, "reason": "ok"},
            "report": str(report_path),
            "checkpoint": str(checkpoint_path),
            "latest_checkpoint": str(latest_path),
            "best_valid_loss": 0.123,
            "git_commit": "deadbeef",
            "description_file_sha256": "desc-hash",
            "model_config": {"feature_dim": 256, "num_heads": 8},
            "four_top_level_loss_weights": {
                "w_ground": 1.0,
                "w_teacher": 1.0,
                "w_mask": 1.0,
                "w_query": 1.0,
            },
        }
        _touch(report_path, json.dumps(payload))
        phase_payloads[phase_name] = payload
        phase_report_paths.append(str(report_path))

    manifest_path = tmp_path / "frozen_experiment_manifest.json"
    eval_summary = _touch(tmp_path / "eval" / "mrsg_eval_summary.json", '{"status": "ok"}\n')
    case_diagnostics = _touch(tmp_path / "eval" / "mrsg_case_diagnostics.json", '{"cases": []}\n')

    env = {
        "DESCRIPTIONS_JSON": str(descriptions_json),
        "FEATURE_DIM": "256",
        "TEXT_DIM_JSON": "null",
        "NUM_HEADS": "8",
        "FOCAL_SLOTS": "4",
        "TOPK_FRACTION": "0.15",
        "ROUTE_TEMPERATURE": "1.0",
        "PHASE_A_EPOCHS": "5",
        "PHASE_B_EPOCHS": "10",
        "PHASE_C_EPOCHS": "10",
        "PHASE_A_BATCH_SIZE": "8",
        "PHASE_B_BATCH_SIZE": "8",
        "PHASE_C_BATCH_SIZE": "8",
        "PHASE_A_LEARNING_RATE": "5e-4",
        "PHASE_B_LEARNING_RATE": "1e-4",
        "PHASE_C_LEARNING_RATE": "1e-4",
        "PHASE_C_TEACHER_DECAY": "0.99",
        "W_GROUND": "1.0",
        "W_TEACHER": "1.0",
        "W_MASK": "1.0",
        "W_QUERY": "1.0",
        "SEED": "13",
        "DEVICE": "cpu",
        "MSCXR_EVAL_ARGS_JSON": json.dumps({"dataset": "MS_CXR", "split": "test"}, sort_keys=True),
        "CHEXLOCALIZE_EVAL_ARGS_JSON": json.dumps({"dataset": "CHEXLOCALIZE", "split": "test"}, sort_keys=True),
        "SCORE_ARGS_JSON": json.dumps({"dataset": "MS_CXR_CLS", "margin": True}, sort_keys=True),
    }
    args = [
        str(manifest_path),
        str(cache_report_path),
        str(protocol_manifest_path),
        *phase_report_paths,
        str(afloc_checkpoint),
    ]
    outputs = {
        "manifest_path": manifest_path,
        "eval_summary": eval_summary,
        "case_diagnostics": case_diagnostics,
    }
    return args, env, outputs


def test_mrsg_runner_contains_all_phases_and_forbids_dcem_inputs() -> None:
    script = _script()

    assert "set -euo pipefail" in script
    assert "build_mrsg_image_report_cache" in script
    assert '--phase "locality"' in script
    assert '--phase "grounding"' in script
    assert '--phase "consistency"' in script
    assert "eval_mscxr_afloc_mrsg" in script
    assert "score_mscxr_learned_repair_metrics" in script
    assert "report_stage_c_results" in script
    assert "START_STAGE" in script
    assert "[0/10] protocol manifest and MS-CXR exclusion audit" in script
    assert "[1/10] Phase A locality warm-up" in script
    assert "[2/10] Gate A check" in script
    assert "[3/10] Phase B sparse grounding" in script
    assert "[4/10] Gate B check" in script
    assert "[5/10] Phase C dual consistency" in script
    assert "[6/10] Gate C and experiment freeze manifest" in script
    assert "[7/10] raw MS-CXR heatmaps" in script
    assert "[8/10] raw scoring against AFLoc/DCEM baselines" in script
    assert "[9/10] frozen CheXlocalize external evaluation" in script
    assert "[10/10] report and diagnostics bundle" in script
    assert "BASE_HMAPS_NPY" not in script
    assert "REGION_SCORE_CSV" not in script
    assert "validation-gate" not in script
    assert "lambda-override" not in script
    assert "prepared-inputs-npz" not in script
    assert "--base-hmaps-npy" not in script
    assert "--region-score-csv" not in script


def test_mrsg_runner_uses_exact_cli_flags_and_best_checkpoints() -> None:
    script = _script()

    assert "--mimic-csv" in script
    assert "--mscxr-json" in script
    assert "--descriptions-json" in script
    assert "--valid-fraction" in script
    assert "--seed" in script
    assert "--train-manifest" in script
    assert "--valid-manifest" in script
    assert "--protocol-manifest" in script
    assert "--afloc-checkpoint" in script
    assert "--feature-dim" in script
    assert "--num-heads" in script
    assert "--focal-slots" in script
    assert "--topk-fraction" in script
    assert "--route-temperature" in script
    assert "--epochs" in script
    assert "--batch-size" in script
    assert "--learning-rate" in script
    assert "--teacher-decay" in script
    assert "--w-ground" in script
    assert "--w-teacher" in script
    assert "--w-mask" in script
    assert "--w-query" in script
    assert "--previous-checkpoint" in script
    assert "--resume-checkpoint" in script
    assert "--dataset" in script
    assert "--split" in script
    assert "--method-name" in script
    assert "--hmaps-root" in script
    assert "--methods" in script
    assert "--candidate-categories" in script
    assert "--bootstrap-replicates" in script
    assert "--metrics-dir" in script
    assert "mrsg_phase_a.pt" in script
    assert "mrsg_phase_b.pt" in script
    assert "mrsg_phase_c.pt" in script
    assert '--previous-checkpoint "${PHASE_A_BEST}"' in script
    assert '--previous-checkpoint "${PHASE_B_BEST}"' in script
    assert '--checkpoint "${PHASE_C_BEST}"' in script


def test_mrsg_runner_has_preflight_resume_and_frozen_manifest_strategy() -> None:
    script = _script()

    assert "DRY_RUN" in script
    assert "PREFLIGHT_ONLY" in script
    assert "ALLOW_DIRTY_OUTROOT" in script
    assert "require_nonempty_env" in script
    assert "require_file" in script
    assert "require_dir" in script
    assert "REFERENCE_HMAPS_ROOT" in script
    assert "frozen_experiment_manifest.json" in script
    assert "validate_frozen_manifest" in script
    assert "update_frozen_manifest_outputs" in script
    assert "git_commit" in script
    assert "data_protocol_sha256" in script
    assert "model_config" in script
    assert "four_top_level_loss_weights" in script
    assert "test_evaluated" in script
    assert "output_hashes" in script
    assert "MSCXR_EVAL_ARGS_JSON" in script
    assert "CHEXLOCALIZE_EVAL_ARGS_JSON" in script
    assert "SCORE_ARGS_JSON" in script


def test_mrsg_runner_rejects_implicit_latest_resume_and_documents_explicit_resume_only() -> None:
    script = _script()

    assert "AUTO_RESUME_LATEST" not in script
    assert 'PHASE_A_RESUME_CHECKPOINT="${PHASE_A_RESUME_CHECKPOINT:-}"' in script
    assert 'PHASE_B_RESUME_CHECKPOINT="${PHASE_B_RESUME_CHECKPOINT:-}"' in script
    assert 'PHASE_C_RESUME_CHECKPOINT="${PHASE_C_RESUME_CHECKPOINT:-}"' in script
    assert "Explicit same-phase resume only" in script
    assert "resume checkpoint disabled by default" in script
    assert "same-phase rerun always starts from the passed upstream best checkpoint" in script
    assert 'resolve_resume_checkpoint "PHASE_A_RESUME_CHECKPOINT"' in script
    assert 'resolve_resume_checkpoint "PHASE_B_RESUME_CHECKPOINT"' in script
    assert 'resolve_resume_checkpoint "PHASE_C_RESUME_CHECKPOINT"' in script
    assert "--resume-checkpoint" in script
    assert "PHASE_A_LATEST" not in script
    assert "PHASE_B_LATEST" not in script
    assert "PHASE_C_LATEST" not in script


def test_frozen_manifest_python_preserves_true_freeze_state_and_rejects_second_stage7(tmp_path: Path) -> None:
    args, env, outputs = _manifest_fixture(tmp_path)

    first_freeze = _run_python_block("write_frozen_manifest", *args, env=env)
    assert first_freeze.returncode == 0, first_freeze.stderr
    manifest_path = outputs["manifest_path"]
    first_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first_payload["test_evaluated"] is False

    update = _run_python_block(
        "update_frozen_manifest_outputs",
        str(manifest_path),
        "mscxr_eval_summary_json",
        str(outputs["eval_summary"]),
        "mscxr_case_diagnostics_json",
        str(outputs["case_diagnostics"]),
        env=env,
    )
    assert update.returncode == 0, update.stderr
    updated_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    scientific_before = {
        key: value
        for key, value in first_payload.items()
        if key not in {"test_evaluated", "output_hashes"}
    }
    scientific_after_update = {
        key: value
        for key, value in updated_payload.items()
        if key not in {"test_evaluated", "output_hashes"}
    }
    assert updated_payload["test_evaluated"] is True
    assert scientific_after_update == scientific_before

    second_freeze = _run_python_block("write_frozen_manifest", *args, env=env)
    assert second_freeze.returncode == 0, second_freeze.stderr
    frozen_again = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert frozen_again["test_evaluated"] is True
    scientific_after_refreeze = {
        key: value
        for key, value in frozen_again.items()
        if key not in {"test_evaluated", "output_hashes"}
    }
    assert scientific_after_refreeze == scientific_before

    guard = _run_python_block("guard_stage7_frozen_manifest", str(manifest_path), env=env)
    assert guard.returncode != 0
    assert "test_evaluated already true" in guard.stderr


def test_mrsg_runner_scores_against_afloc_and_dcem_references() -> None:
    script = _script()

    assert 'BASELINE_METHOD="${BASELINE_METHOD:-baseline}"' in script
    assert 'DCEM_METHOD="${DCEM_METHOD:-phrase_anatomy_dcem}"' in script
    assert 'SCORE_METHODS="${SCORE_METHODS:-${BASELINE_METHOD},${DCEM_METHOD},${METHOD_NAME}}"' in script
    assert 'REFERENCE_HMAPS_ROOT="${REFERENCE_HMAPS_ROOT:-' in script
    assert 'install_reference_hmap "${BASELINE_METHOD}"' in script
    assert 'install_reference_hmap "${DCEM_METHOD}"' in script
    assert '--dataset "MS_CXR"' in script
    assert '--dataset "CHEXLOCALIZE"' in script
    assert '--split "test"' in script


def test_mrsg_runner_passes_bash_n_when_available() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available in this environment")

    smoke = subprocess.run(
        [bash, "-lc", "true"],
        capture_output=True,
        text=False,
        check=False,
    )
    if smoke.returncode != 0:
        pytest.skip("bash is present but not usable in this environment")

    result = subprocess.run(
        [bash, "-n", SCRIPT_PATH.as_posix()],
        capture_output=True,
        text=False,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="ignore")
