from pathlib import Path


SCRIPT = Path("scripts/run_afloc_mrsg_phase2_server.sh")


def test_phase2_script_runs_fixed_sequential_box_free_stages() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert "phase2a_stability" in script
    assert "phase2b_cross_correction" in script
    assert "phase2c_phrase_swap" in script
    assert "--w-residual-stability 0.01" in script
    assert "--w-cross-correction 0.25" in script
    assert "--w-phrase-swap 0.10" in script
    assert script.count("--initial-checkpoint") == 3
    assert "PHASE2A_RESUME_CHECKPOINT" in script
    assert "PHASE2B_RESUME_CHECKPOINT" in script
    assert "PHASE2C_RESUME_CHECKPOINT" in script
    assert "MS_CXR" not in script
    assert "validation-gate" not in script
    assert "teacher" not in script.lower()
