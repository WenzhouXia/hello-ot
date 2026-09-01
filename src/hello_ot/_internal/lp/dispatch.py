from __future__ import annotations

from typing import Any

from .wrapper import SolverResult


def solve_lp(lp_solver: Any, **solve_kwargs: Any) -> SolverResult:
    """
    CN: 调用 HELLO 唯一支持的 CuPDLPx backend。
    EN: Invoke the only LP backend supported by HELLO, CuPDLPx.
    """
    # CN: native wrapper 只在真正选择 native backend 时导入，保持基础包 compiler-free。
    # EN: Import the native wrapper only when native is actually selected, keeping the base package compiler-free.
    from .cupdlpx import CuPDLPxSolver

    if not isinstance(lp_solver, CuPDLPxSolver):
        raise TypeError(f"Unsupported LP solver type: {type(lp_solver).__name__}")
    return lp_solver.solve(**solve_kwargs)
