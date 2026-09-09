from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "hello_ot"


def test_hello_ot_does_not_import_internal_repository_package() -> None:
    violations = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "hierarchical_ot" or name.startswith("hierarchical_ot.") for name in names):
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    assert violations == []


def test_public_modules_import_without_loading_hierarchical_ot() -> None:
    import sys

    import hello_ot
    import hello_ot.api

    assert "hierarchical_ot" not in sys.modules
    assert hello_ot.solve is not None
    assert hello_ot.api.solve_problem is not None


def test_removed_compatibility_features_do_not_reappear_in_package_sources() -> None:
    forbidden = ("use_faiss", "hprlp")
    violations = []
    source_suffixes = {".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}
    for path in PACKAGE_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in source_suffixes:
            continue
        source = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in source:
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{token}")
    assert violations == []


def test_private_observer_is_scoped_and_optional() -> None:
    from hello_ot._internal.runtime_context import record_solve_event, use_solve_runtime

    events = []
    record_solve_event("outside", value=0)
    with use_solve_runtime(observer=lambda name, payload: events.append((name, payload))):
        record_solve_event("inside", value=1)
    record_solve_event("outside_again", value=2)

    assert events == [("inside", {"value": 1})]
