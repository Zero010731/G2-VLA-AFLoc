import ast
from pathlib import Path


def test_no_pep604_union_in_runtime_assignments() -> None:
    """Python 3.9 evaluates assignment RHS, so PEP 604 unions break there."""
    offenders: list[str] = []
    for path in sorted(Path("anaprior").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for child in ast.walk(node.value):
                if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr):
                    offenders.append(f"{path}:{child.lineno}")

    assert offenders == []
