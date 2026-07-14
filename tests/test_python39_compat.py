import ast
from pathlib import Path


MRSG_RUNTIME_PATHS = (
    tuple(sorted(Path("anaprior/models/afloc_mrsg").rglob("*.py")))
    + (
        Path("anaprior/data/mrsg_dataset.py"),
        Path("anaprior/data/mrsg_phrases.py"),
        Path("anaprior/data/mrsg_protocol.py"),
        Path("anaprior/eval/eval_mscxr_afloc_mrsg.py"),
        Path("anaprior/features/afloc_mrsg_encoder.py"),
        Path("anaprior/features/afloc_preprocessing.py"),
        Path("anaprior/train/build_mrsg_image_report_cache.py"),
        Path("anaprior/train/train_afloc_mrsg.py"),
    )
)


def test_mrsg_runtime_files_exist_and_parse_with_python39_grammar() -> None:
    missing = [str(path) for path in MRSG_RUNTIME_PATHS if not path.exists()]
    assert missing == []

    offenders: list[str] = []
    for path in MRSG_RUNTIME_PATHS:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path), feature_version=(3, 9))
        except SyntaxError as exc:
            offenders.append(f"{path}:{exc.lineno}:{exc.offset} {exc.msg}")

    assert offenders == []


def test_mrsg_runtime_has_no_pep604_union_in_runtime_assignments() -> None:
    offenders: list[str] = []
    for path in MRSG_RUNTIME_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for child in ast.walk(node.value):
                if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr):
                    offenders.append(f"{path}:{child.lineno}")

    assert offenders == []
