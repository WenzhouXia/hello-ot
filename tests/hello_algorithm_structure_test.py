from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import hello_ot
from hello_ot._internal.cost_types import normalize_cost_type_name


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


def test_internal_solver_has_no_dead_compatibility_or_reporting_methods() -> None:
    """
    CN: solver 不应保留失效兼容转发或重复的 profiling 实现。
    EN: The solver must not retain dead compatibility forwards or duplicate profiling implementations.
    """
    source = Path("src/hello_ot/_internal/core/solver.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    solver = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HierarchicalOTSolver")
    method_names = {node.name for node in solver.body if isinstance(node, ast.FunctionDef)}
    assert method_names.isdisjoint(
        {
            "_decode_keep_to_struct",
            "_remap_duals_for_warm_start",
            "_initialize_level",
            "_solve_exact_flat",
            "_print_profile_inner",
            "_print_profile_level",
            "_print_profile_init_level",
            "_print_profile_run",
        }
    )


def test_cost_type_normalization_preserves_all_supported_aliases() -> None:
    """
    CN: 单一 normalization 入口必须保留两个旧实现接受的别名并集。
    EN: The canonical normalizer must preserve the union of aliases accepted by both legacy implementations.
    """
    expected = {
        "euclidean": "l2",
        "l_inf": "linf",
        "l2sq": "l2^2",
        "sq_euclidean": "l2^2",
        "sqeuclidean": "l2^2",
        "squared_euclidean": "l2^2",
    }
    assert {name: normalize_cost_type_name(name) for name in expected} == expected


def test_cost_stage_transition_does_not_import_algorithm() -> None:
    """
    CN: cost-stage 转换只能依赖中立执行协议，不得反向导入主算法。
    EN: Cost-stage transitions must depend on neutral execution protocols, not import the main algorithm.
    """
    source = Path("src/hello_ot/perturbation.py").read_text(encoding="utf-8")
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "algorithm" not in imports
    assert "hello_ot.algorithm" not in imports


def test_hierarchy_execution_types_live_in_neutral_module() -> None:
    """
    CN: 主算法与 cost-stage 转换共享同一组 execution 类型和 finalizer 协议。
    EN: The main algorithm and cost-stage transitions share execution types and a finalizer protocol.
    """
    source = Path("src/hello_ot/hierarchy/execution.py").read_text(encoding="utf-8")
    classes = {
        node.name for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
    }

    assert {
        "DualAssignmentResult",
        "HierarchyExecution",
        "HierarchyLevelFinalizer",
        "InitializedHierarchyLevel",
    }.issubset(classes)
