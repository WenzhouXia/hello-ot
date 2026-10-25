from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def installer():
    """
    CN: 装载安装器但不执行外部安装。
    EN: Load the installer without executing an external installation.
    """
    spec = importlib.util.spec_from_file_location("hello_installer", ROOT / "scripts/install_backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_mismatch_explains_actual_expected_and_build_path(monkeypatch, capsys):
    module = installer()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "micromamba":
            raise FileNotFoundError()
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(dict(
            python=[3, 12], system="Linux", machine="x86_64", torch="2.5.0+cpu", cuda=None,
            abi=False, legacy=False)), stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["installer", "native", "example"])
    with pytest.raises(SystemExit, match="Expected:.*torch 2.7.1.*Actual:.*2.5.0") as error:
        module.main()
    assert "build_native_wheel.sh" in str(error.value)
    assert all("pip" not in command for command in calls)


def test_cpp_install_does_not_install_cuda_package(monkeypatch):
    module = installer()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "micromamba":
            raise FileNotFoundError()
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(dict(
            python=[3, 12], system="Linux", machine="x86_64", torch="2.7.1+cpu", cuda=None,
            abi=False, legacy=False)), stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["installer", "cpp", "example", "--wheel", "/tmp/cpp.whl"])
    module.main()
    assert any("/tmp/cpp.whl" in command for command in calls)
    assert all(not any("hello_ot_native" in word for word in command) for command in calls)
    assert any("--require-cpp" in command for command in calls)


def test_public_wheels_have_disjoint_file_ownership():
    import os

    directory = Path(os.environ.get("HELLO_OT_WHEEL_DIR", ROOT / "dist"))
    wheels = list(directory.glob("*.whl"))
    if len(wheels) < 3:
        pytest.skip("build the three release wheels to validate file ownership")
    owners = {}
    names = set()
    for wheel in wheels:
        if wheel.name.startswith(("hello_ot-", "hello_ot_cpp-", "hello_ot_native-")):
            names.add(wheel.name.split("-", 1)[0])
            with zipfile.ZipFile(wheel) as archive:
                for path in archive.namelist():
                    assert path not in owners, (path, owners.get(path), wheel.name)
                    owners[path] = wheel.name
    assert names == {"hello_ot", "hello_ot_cpp", "hello_ot_native"}
    assert not any(path.startswith("hello_ot/") and path.endswith(".so") for path in owners)
