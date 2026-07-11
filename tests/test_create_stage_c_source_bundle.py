import json
import zipfile
from pathlib import Path

from anaprior.tools.create_stage_c_source_bundle import create_bundle


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_create_bundle_includes_stage_c_sources_and_excludes_outputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write(project / "afloc" / "hf_utils.py")
    _write(project / "afloc" / "models" / "afloc_model.py")
    _write(project / "afloc" / "models" / "text_model.py")
    _write(project / "anaprior" / "eval" / "stage_c.py")
    _write(project / "tests" / "test_stage_c.py")
    _write(project / "docs" / "ANAPRIOR_LOC_中文落地说明.md")
    _write(project / "README_NEXT_STAGE.md")
    _write(project / "scripts" / "run_stage_c_learned_repair_server.sh")
    _write(project / "scripts" / "run_stage_d_oracle_recoverability_server.sh")
    _write(project / "scripts" / "run_stage_e_8class_gap_diagnosis_server.sh")

    _write(project / "outputs" / "large_result.npy")
    _write(project / ".git" / "config")
    _write(project / ".pytest_cache" / "cache")
    _write(project / "anaprior" / "__pycache__" / "stage_c.cpython-39.pyc")
    _write(project / "anaprior" / "eval" / "__pycache__" / "stage_c.cpython-39.pyc")

    out_zip = project / "outputs" / "bundle" / "stage_c_source.zip"

    manifest = create_bundle(project, out_zip)

    assert manifest["file_count"] == 10
    assert manifest["files"] == [
        "README_NEXT_STAGE.md",
        "afloc/hf_utils.py",
        "afloc/models/afloc_model.py",
        "afloc/models/text_model.py",
        "anaprior/eval/stage_c.py",
        "docs/ANAPRIOR_LOC_中文落地说明.md",
        "scripts/run_stage_c_learned_repair_server.sh",
        "scripts/run_stage_d_oracle_recoverability_server.sh",
        "scripts/run_stage_e_8class_gap_diagnosis_server.sh",
        "tests/test_stage_c.py",
    ]

    with zipfile.ZipFile(out_zip) as zf:
        names = sorted(zf.namelist())
        assert names == [
            "ANAPRIOR_STAGE_C_BUNDLE_MANIFEST.json",
            "README_NEXT_STAGE.md",
            "afloc/hf_utils.py",
            "afloc/models/afloc_model.py",
            "afloc/models/text_model.py",
            "anaprior/eval/stage_c.py",
            "docs/ANAPRIOR_LOC_中文落地说明.md",
            "scripts/run_stage_c_learned_repair_server.sh",
            "scripts/run_stage_d_oracle_recoverability_server.sh",
            "scripts/run_stage_e_8class_gap_diagnosis_server.sh",
            "tests/test_stage_c.py",
        ]
        zipped_manifest = json.loads(
            zf.read("ANAPRIOR_STAGE_C_BUNDLE_MANIFEST.json").decode("utf-8")
        )
        assert zipped_manifest["files"] == manifest["files"]
        assert "outputs/large_result.npy" not in names
        assert ".git/config" not in names
        assert "anaprior/__pycache__/stage_c.cpython-39.pyc" not in names


def test_bundle_command_is_documented_for_server_transfer() -> None:
    readme = Path("README_NEXT_STAGE.md").read_text(encoding="utf-8")
    chinese_doc = Path("docs/ANAPRIOR_LOC_中文落地说明.md").read_text(encoding="utf-8")

    required = [
        "python -m anaprior.tools.create_stage_c_source_bundle",
        "anaprior_stage_c_source_bundle.zip",
        "AFLOC_BERT_TYPE",
        "AFLOC_HF_LOCAL_FILES_ONLY",
        "scp",
        "unzip",
    ]
    for text in required:
        assert text in readme
        assert text in chinese_doc


def test_delivery_audit_document_is_linked_from_docs() -> None:
    audit_path = Path("docs/ANAPRIOR_STAGE_C_DELIVERY_AUDIT.md")
    readme = Path("README_NEXT_STAGE.md").read_text(encoding="utf-8")
    chinese_doc = Path("docs/ANAPRIOR_LOC_中文落地说明.md").read_text(encoding="utf-8")

    assert audit_path.exists()
    audit = audit_path.read_text(encoding="utf-8")
    for text in [
        "Stage C 交付审计",
        "本地已完成",
        "服务器待执行",
        "ANAPRIOR_RUN_SMOKE=1",
        "ANAPRIOR_RUN_SMOKE=0",
        "anaprior_stage_c_source_bundle.zip",
    ]:
        assert text in audit

    assert "docs/ANAPRIOR_STAGE_C_DELIVERY_AUDIT.md" in readme
    assert "docs/ANAPRIOR_STAGE_C_DELIVERY_AUDIT.md" in chinese_doc
