"""CN: 检查 sdist 边界并从中构建、安装 portable wheel。EN: Check sdist boundaries and build/install a portable wheel from it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import tomllib


def check_sdist(archive):
    """CN: 使用临时目录验证源码发行档，避免修改当前安装。EN: Validate a source archive in temporary directories without changing the current installation."""
    with tempfile.TemporaryDirectory(prefix="hello_sdist_check_") as temporary:
        workspace = Path(temporary)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            roots = {Path(member.name).parts[0] for member in members}
            if len(roots) != 1:
                raise ValueError("sdist must contain one root directory")
            for member in members:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unexpected archive entry: {member.name}")
                if any(part in {"paper_experiments", "export_experiments", "experiments", "tests",
                                "hierarchical_ot"} for part in path.parts):
                    raise ValueError(f"Repository-only content in sdist: {member.name}")
                if path.suffix in {".so", ".pyc"}:
                    raise ValueError(f"Build artifact in sdist: {member.name}")
            tar.extractall(workspace)
        source = workspace / roots.pop()
        project_name = tomllib.loads((source / "pyproject.toml").read_text())["project"]["name"]
        if project_name in {"hello-ot-cpp", "hello-ot-native"}:
            _check_extension_sdist(source, workspace, project_name)
            return
        for required in ("setup.py", "pyproject.toml", "src/hello_ot/__init__.py",
                         "src/hello_ot/_native/cupdlpx/pycupdlpx/cupdlpx_bindings.cpp"):
            if not (source / required).is_file():
                raise ValueError(f"Missing build source: {required}")
        wheels = workspace / "wheels"
        wheels.mkdir()
        environment = dict(os.environ, HELLO_OT_BUILD_NATIVE="0", CUDA_VISIBLE_DEVICES="")
        environment.pop("PYTHONPATH", None)
        subprocess.run(
            [sys.executable, "-c",
             "import setuptools.build_meta as b; b.build_wheel(" + repr(str(wheels)) + ")"],
            cwd=source, env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        wheel, = wheels.glob("*.whl")
        with zipfile.ZipFile(wheel) as zipped:
            names = zipped.namelist()
            if any(name.startswith(("paper_experiments/", "export_experiments/", "experiments/"))
                   or name.endswith(".so") for name in names):
                raise ValueError("Portable wheel contains experiments or native binaries")
        installed = workspace / "installed"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target",
             str(installed), str(wheel)],
            env=environment, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        environment["PYTHONPATH"] = str(installed)
        code = (
            "from pathlib import Path; import numpy as np; import hello_ot; "
            f"assert Path(hello_ot.__file__).is_relative_to(Path({str(installed)!r})); "
            "x=np.arange(24,dtype=np.float32).reshape(8,3); "
            "r=hello_ot.solve(x,x+1,random_seed=42,options=hello_ot.SolverOptions("
            "backend='torch',torch_device='cpu',cost_perturbation='off')); "
            "assert np.isfinite(r.objective); assert abs(r.objective-3)<1e-5; "
            "print('Installed portable wheel CPU solve passed')"
        )
        subprocess.run([sys.executable, "-c", code], cwd=workspace, env=environment, check=True)
        print(f"sdist boundary and wheel build/install passed: {archive}")


def _check_extension_sdist(source, workspace, name):
    """
    CN: 从独立扩展源码包构建并安装，确认只拥有自身的模块文件。
    EN: Build and install an extension source archive, verifying exclusive ownership of its modules.
    """
    kind = "cpp" if name == "hello-ot-cpp" else "native"
    expected = source / "native_src" / ("solver.cpp" if kind == "cpp" else "cupdlpx/pycupdlpx/cupdlpx_bindings.cpp")
    if not expected.is_file():
        raise ValueError(f"Missing extension build source: {expected}")
    wheels = workspace / "wheels"
    wheels.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation",
                    "--wheel-dir", str(wheels)], cwd=source, check=True)
    wheel, = wheels.glob("*.whl")
    namespace = f"hello_ot_{kind}"
    with zipfile.ZipFile(wheel) as archive:
        files = archive.namelist()
        if any(not path.startswith((namespace + "/", namespace + "-")) for path in files):
            raise ValueError("Extension wheel owns files outside its namespace")
        if not any(path.startswith(namespace + "/") and path.endswith(".so") for path in files):
            raise ValueError("Extension wheel contains no compiled modules")
    installed = workspace / "installed"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--target",
                    str(installed), str(wheel)], check=True)
    environment = dict(os.environ, PYTHONPATH=str(installed))
    code = f"import importlib, torch; m=importlib.import_module('{namespace}'); assert m.API_VERSION == 1; "
    for module in (["_lp", "_scan"] if kind == "cpp" else ["pycupdlpx", "hierot_inner_product_scan_ext", "hierot_norm_cost_scan_ext", "_support_sparsifier_ext"]):
        code += f"importlib.import_module('{namespace}.{module}'); "
    subprocess.run([sys.executable, "-c", code], cwd=workspace, env=environment, check=True)
    print(f"Independent extension sdist build/install passed: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    check_sdist(parser.parse_args().archive.resolve())
