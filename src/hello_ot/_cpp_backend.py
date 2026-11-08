from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import importlib
from threading import RLock

import torch


_THREAD_LOCK = RLock()


@lru_cache(maxsize=2)
def load_cpp_extension(kind: str):
    """
    CN: 开发模式构建本地 CPU 内核；公开模式按需装载独立扩展包。
    EN: Build local CPU kernels in development mode; load the independent extension package publicly.
    """
    if kind not in {"lp", "scan"}:
        raise ValueError("unknown C++ extension kind")
    from hello_ot._source_build import source_mode, load_source_extension
    if source_mode():
        from pathlib import Path
        import pybind11

        source = Path(__file__).resolve().parent / "_native/cpp" / ("solver.cpp" if kind == "lp" else "scan.cpp")
        return load_source_extension(
            "_" + kind, [str(source)], extra_include_paths=[pybind11.get_include()],
            extra_cflags=["-O3", "-std=c++17", "-fopenmp", "-ffp-contract=off"],
            extra_ldflags=["-fopenmp"],
        )
    try:
        package = importlib.import_module("hello_ot_cpp")
        if package.API_VERSION != 1:
            raise RuntimeError("Incompatible hello-ot-cpp API; install the matching release")
        return importlib.import_module(f"hello_ot_cpp._{kind}")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The C++ CPU backend requires hello-ot-cpp. Run bash scripts/install_cpp.sh "
            "ENV_NAME, or build it with bash scripts/build_cpp_wheel.sh dist. "
            f"Extension loading failed: {exc}"
        ) from exc


@contextmanager
def cpp_runtime(num_threads: int | None):
    """
    CN: 为 C++ 求解临时限制 Torch/BLAS/OpenMP 线程并恢复；线程策略是进程级设置。
    EN: Temporarily limit and restore Torch/BLAS/OpenMP threads for C++ solves; thread policy is process-wide.
    """
    load_cpp_extension("lp")
    load_cpp_extension("scan")
    if num_threads is None:
        yield
        return
    from threadpoolctl import threadpool_limits

    with _THREAD_LOCK:
        previous = torch.get_num_threads()
        try:
            with threadpool_limits(limits=num_threads):
                torch.set_num_threads(num_threads)
                yield
        finally:
            torch.set_num_threads(previous)
