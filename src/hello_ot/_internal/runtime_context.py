from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional


SolveObserver = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class SolveRuntime:
    """
    CN: 一次 solve 的私有运行时上下文；observer 只用于迁移测试。
    EN: Private per-solve runtime context; the observer is only for migration tests.
    """

    observer: Optional[SolveObserver] = None


_CURRENT_RUNTIME: ContextVar[SolveRuntime] = ContextVar(
    "hello_ot_current_solve_runtime",
    default=SolveRuntime(),
)


def current_solve_runtime() -> SolveRuntime:
    """CN: 返回当前 solve 上下文。 EN: Return the current solve context."""
    return _CURRENT_RUNTIME.get()


@contextmanager
def use_solve_runtime(*, observer: Optional[SolveObserver] = None) -> Iterator[SolveRuntime]:
    """
    CN: 为测试安装临时 solve observer；上下文结束后自动恢复。
    EN: Install a temporary solve observer for tests and restore it on exit.
    """
    runtime = SolveRuntime(observer=observer)
    token: Token[SolveRuntime] = _CURRENT_RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _CURRENT_RUNTIME.reset(token)


def record_solve_event(name: str, **payload: Any) -> None:
    """
    CN: 仅在测试 observer 存在时转发事件；正常路径不保留数据。
    EN: Forward an event only when a test observer is installed; normal solves retain no data.
    """
    observer = _CURRENT_RUNTIME.get().observer
    if observer is not None:
        observer(str(name), payload)
