from pathlib import Path


SCRIPT = Path("scripts/run_afloc_official_anchor_phase0b_server.sh")


def test_phase0b_script_has_exact_official_parity_contract() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert 'METHOD_NAME="${METHOD_NAME:-afloc_official_anchor}"' in script
    assert "eval_mscxr_afloc_official_anchor" in script
    assert "report_afloc_official_anchor_parity" in script
    assert "--dataset MS_CXR" in script
    assert "AFLOC_BERT_TYPE" in script
    assert "AFLOC_HF_LOCAL_FILES_ONLY=1" in script
    assert "TRANSFORMERS_OFFLINE=1" in script
    assert "REFERENCE_HMAPS_ROOT" in script
    assert "START_STAGE must be 0, 1, or 2" in script
    assert "--min-matched-cases 900" in script
    assert "MIMIC_CSV" not in script
    assert "mrsg_phase" not in script.lower()
