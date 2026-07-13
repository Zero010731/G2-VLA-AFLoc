from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT_PATH = Path("scripts/run_afloc_mrsg_full_server.sh")


def _script() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


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
    assert "mrsg_phase_a_latest.pt" in script
    assert "mrsg_phase_b_latest.pt" in script
    assert "mrsg_phase_c_latest.pt" in script
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
