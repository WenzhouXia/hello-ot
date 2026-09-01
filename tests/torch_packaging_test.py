from __future__ import annotations

import builtins
import runpy
from pathlib import Path

import setuptools


ROOT = Path(__file__).resolve().parents[1]


def test_default_setup_does_not_import_native_build_toolchain(monkeypatch) -> None:
    # CN: 默认 pip 安装的 metadata/build 阶段不得要求 torch cpp_extension 或 pybind11。
    # EN: Default pip metadata/build must not require torch cpp_extension or pybind11.
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if str(name) == "pybind11" or str(name).startswith("torch.utils.cpp_extension"):
            raise AssertionError(f"default build imported native toolchain: {name}")
        return original_import(name, *args, **kwargs)

    captured = {}
    monkeypatch.delenv("HELLO_OT_BUILD_NATIVE", raising=False)
    monkeypatch.delenv("HELLO_OT_BUILD_SUPPORT_SPARSIFIER_ONLY", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")
    assert captured["ext_modules"] == []
    assert captured["cmdclass"] == {}


def test_build_isolation_requirements_are_pure_python() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    build_system = text.split("[project]", maxsplit=1)[0]
    assert "torch" not in build_system.lower()
    assert "ninja" not in build_system.lower()
    assert "pybind11" not in build_system.lower()
