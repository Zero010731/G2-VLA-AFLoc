from pathlib import Path


SCRIPT = Path("scripts/run_afloc_anchor_phase0_server.sh")


def test_anchor_phase0_server_script_has_frozen_three_stage_contract() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "eval_mscxr_afloc_anchor" in script
    assert script.count("--dataset MS_CXR_CLS") == 2
    assert "score_mscxr_learned_repair_metrics" in script
    assert "report_afloc_anchor_phase0" in script
    assert "baseline,phrase_anatomy_dcem,afloc_anchor" in script
    assert "validation-gate" not in script
    assert "train_afloc_mrsg" not in script
    assert "MIMIC_CSV" not in script
    assert "PREFLIGHT_ONLY" in script
    assert "DRY_RUN" in script
    assert "START_STAGE" in script
    assert "ALLOW_DIRTY_OUTROOT" in script
    assert "AFLOC_BERT_TYPE" in script
    assert "git rev-parse HEAD" in script
    assert "export AFLOC_CHECKPOINT LOCALIZATION_MS_CXR_JSON LOCALIZATION_MIMIC_IMG_DIR" in script
    assert "export METHOD_NAME SCORE_METHODS SEED BOOTSTRAP_REPLICATES GIT_COMMIT" in script
    assert script.index('if [[ "${PREFLIGHT_ONLY}" == "1" ]]') < script.index(
        'mkdir -p "${ANCHOR_EVAL_ROOT}"'
    )
    assert 'if [[ "${DRY_RUN}" != "1" ]]; then\n    require_file "anchor hmaps"' in script
    assert 'if [[ "${DRY_RUN}" != "1" ]]; then\n    require_file "anchor summary"' in script


def test_anchor_phase0_server_script_uses_fresh_method_paths() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'ANCHOR_HMAPS="${ANCHOR_EVAL_ROOT}/${METHOD_NAME}/hmaps.npy"' in script
    assert 'SCORE_HMAP_ROOT="${OUTROOT}/score_hmaps"' in script
    assert 'REPORT_ROOT="${OUTROOT}/report"' in script
    assert 'METHOD_NAME="${METHOD_NAME:-afloc_anchor}"' in script
