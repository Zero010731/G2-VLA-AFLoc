from pathlib import Path
import importlib.util


def _load_hf_utils():
    path = Path("afloc") / "hf_utils.py"
    spec = importlib.util.spec_from_file_location("afloc_hf_utils_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_bert_type_can_override_checkpoint_value(monkeypatch) -> None:
    hf_utils = _load_hf_utils()
    monkeypatch.setenv("AFLOC_BERT_TYPE", "/mnt/models/Bio_ClinicalBERT")

    assert hf_utils.resolve_bert_type("emilyalsentzer/Bio_ClinicalBERT") == "/mnt/models/Bio_ClinicalBERT"


def test_hf_from_pretrained_kwargs_uses_local_files_for_existing_local_path(tmp_path: Path) -> None:
    hf_utils = _load_hf_utils()
    model_dir = tmp_path / "Bio_ClinicalBERT"
    model_dir.mkdir()

    kwargs = hf_utils.hf_from_pretrained_kwargs(model_dir)

    assert kwargs == {"local_files_only": True}


def test_hf_from_pretrained_kwargs_can_force_offline(monkeypatch) -> None:
    hf_utils = _load_hf_utils()
    monkeypatch.setenv("AFLOC_HF_LOCAL_FILES_ONLY", "1")

    kwargs = hf_utils.hf_from_pretrained_kwargs("emilyalsentzer/Bio_ClinicalBERT")

    assert kwargs == {"local_files_only": True}
