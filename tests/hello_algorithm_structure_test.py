from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import hello_ot


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return node.func.id if isinstance(node.func, ast.Name) else None


def test_hierarchy_function_exposes_the_paper_control_flow() -> None:
    """
    CN: 主算法必须直接展示 hierarchy 外循环与 refinement 内循环。
    EN: The main algorithm must directly expose the hierarchy and refinement loops.
    """
    source = Path("src/hello_ot/algorithm.py").read_text(encoding="utf-8")
    solve_hierarchy = _function(ast.parse(source), "_solve_hierarchy")
    outer_loop = next(node for node in solve_hierarchy.body if isinstance(node, ast.For))
    inner_loop = next(node for node in outer_loop.body if isinstance(node, ast.While))

    solve_statement, certificate_statement, stop_statement = inner_loop.body[:3]
    assert isinstance(solve_statement, ast.Assign)
    assert _called_name(solve_statement.value) == "solve_lp"
    assert isinstance(certificate_statement, ast.Assign)
    assert _called_name(certificate_statement.value) == "check_optimality"
    assert isinstance(stop_statement, ast.If)
    assert isinstance(stop_statement.body[0], ast.Break)
    update_statement = next(node for node in inner_loop.body if isinstance(node, ast.Assign)
                            and _called_name(node.value) == "update_support")
    assert isinstance(update_statement, ast.Assign)
    assert _called_name(update_statement.value) == "update_support"


def test_hierarchy_function_contains_no_runtime_instrumentation() -> None:
    """
    CN: tracing、memory accounting 与 backend dispatch 不得污染论文级主函数。
    EN: Tracing, memory accounting, and backend dispatch must not pollute the paper-level function.
    """
    source = Path("src/hello_ot/algorithm.py").read_text(encoding="utf-8")
    solve_hierarchy = _function(ast.parse(source), "_solve_hierarchy")
    called_names = {
        name
        for node in ast.walk(solve_hierarchy)
        if (name := _called_name(node)) is not None
    }
    assert called_names.isdisjoint(
        {
            "record_solve_event",
            "begin_span",
            "end_span",
            "refine_node_by_cost",
        }
    )


def test_portable_runtime_executes_the_visible_operator_order() -> None:
    """
    CN: observer 验证可移植热路径与主函数展示的算子顺序一致。
    EN: An observer verifies that the portable hot path follows the operator order visible in the main function.
    """
    from hello_ot._internal.runtime_context import use_solve_runtime

    rng = np.random.default_rng(7)
    source = rng.normal(size=(32, 2))
    target = rng.normal(size=(32, 2))
    events = []
    with use_solve_runtime(
        observer=lambda name, payload: events.append((name, payload))
    ):
        hello_ot.solve(
            source,
            target,
            max_iterations=6,
            options=hello_ot.SolverOptions(
                backend="torch",
                torch_device="cpu",
                coarsest_size_threshold=8,
                assignment_topk=2,
                pricing_topk=1,
                support_budget_factor=6.0,
            ),
        )

    operators = [
        (name, payload)
        for name, payload in events
        if name in {"solve_lp", "check_optimality", "update_support"}
    ]
    assert any(name == "update_support" for name, _ in operators)
    for index, (name, payload) in enumerate(operators):
        if name == "solve_lp":
            assert operators[index + 1][0] == "check_optimality"
        if name == "check_optimality" and not bool(payload["converged"]):
            assert operators[index + 1][0] == "update_support"
