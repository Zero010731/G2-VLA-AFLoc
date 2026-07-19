from pathlib import Path


SCRIPT = Path("scripts/run_afloc_anchor_preserving_phase1_server.sh")


def test_phase1_script_is_grounding_only_anchor_preserving_run() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "official_anchor_parity.json" in script
    assert '"parity_passed"' in script
    assert "--phase grounding" in script
    assert "--residual-logit-bound 0.5" in script
    assert "--w-teacher 0.0" in script
    assert "--previous-checkpoint" not in script
    assert "--validation-gate" not in script
    assert "teacher-decay" not in script
    assert "eval_mscxr_afloc_mrsg" in script
    assert "baseline,phrase_anatomy_dcem,afloc_mrsg" in script
    assert "START_STAGE must be 0, 1, 2, or 3" in script
