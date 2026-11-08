from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


_LOADED = {}


def source_mode() -> bool:
    """
    CN: 仅显式启用研究开发模式时从源码构建；公开安装默认使用 wheel。
    EN: Build sources only in explicitly enabled research mode; public installs default to wheels.
    """
    return os.environ.get("HELLO_OT_SOURCE_BUILD", "0") == "1"


def _tool_version(command):
    """
    CN: 记录实际编译器及其版本用于隔离构建缓存。
    EN: Record the actual compiler and its version to isolate build caches.
    """
    executable = shutil.which(command) or command
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True)
    return [str(Path(executable).resolve()), result.stdout + result.stderr]


def wheel_cuda_paths(directory):
    """
    CN: 查找官方 PyTorch wheel 的 CUDA 头文件、库及本地链接别名。
    EN: Discover CUDA headers, libraries and local linker aliases from official PyTorch wheels.
    """
    import torch

    root = Path(torch.__file__).resolve().parent.parent / "nvidia"
    includes, flags = [], []
    links = Path(directory) / "cuda_links"
    for component in sorted(root.iterdir()) if root.is_dir() else []:
        # CN: JAX CUDA plugin 可能额外安装不同 patch 版本的 NVCC/CCCL；
        #     Torch 扩展必须使用 CUDA_HOME 中由 workstation profile 固定的工具链头文件。
        # EN: A JAX CUDA plugin may install NVCC/CCCL at a different patch level;
        #     Torch extensions must use the toolchain headers pinned by CUDA_HOME and the workstation profile.
        if component.name in {"cuda_nvcc", "cuda_cccl"}:
            continue
        if (component / "include").is_dir():
            includes.append(str(component / "include"))
        libraries = component / "lib"
        if not libraries.is_dir():
            continue
        flags.extend([f"-L{libraries}", f"-Wl,-rpath,{libraries}"])
        for library in sorted(libraries.glob("*.so.*")):
            links.mkdir(parents=True, exist_ok=True)
            alias = links / (library.name.split(".so.")[0] + ".so")
            if not alias.is_symlink() and not alias.exists():
                alias.symlink_to(library)
    if links.exists():
        flags.insert(0, f"-L{links}")
    return includes, flags


def load_source_extension(name, sources, **options):
    """
    CN: 新进程按源码、头文件及工具链指纹增量构建；进程内固定已加载版本。
    EN: Build against source, header and toolchain fingerprints in new processes; pin loaded modules within a process.
    """
    import fcntl

    if name in _LOADED:
        return _LOADED[name]
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    sources = [str(Path(path).resolve()) for path in sources]
    cuda = any(Path(path).suffix in {".cu", ".cuh"} for path in sources)
    options.pop("build_directory", None)
    options.setdefault("verbose", os.environ.get("HIER_OT_BUILD_VERBOSE") == "1")
    root = Path(os.path.commonpath([str(Path(path).parent) for path in sources]))
    files = {Path(path) for path in sources}
    for header in root.rglob("*"):
        if header.suffix in {".h", ".hpp", ".cuh", ".inc"} and ".build" not in header.parts:
            files.add(header)
    manifest = {
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)},
        "python": sys.version, "prefix": sys.prefix, "torch": torch.__version__,
        "cuda": torch.version.cuda, "abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "compiler": _tool_version(os.environ.get("CXX", "c++")), "options": options,
        "helper": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "build_environment": {key: os.environ.get(key) for key in (
            "CC", "CXX", "CUDAHOSTCXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
        )},
    }
    if cuda:
        if not CUDA_HOME:
            raise RuntimeError("Source builds require CUDA nvcc; install the workstation profile toolkit")
        manifest["nvcc"] = _tool_version(str(Path(CUDA_HOME) / "bin/nvcc"))
        architectures = os.environ.get("TORCH_CUDA_ARCH_LIST") or ";".join(sorted({
            ".".join(map(str, torch.cuda.get_device_capability(i)))
            for i in range(torch.cuda.device_count())
        }))
        if not architectures:
            raise RuntimeError("A visible CUDA device or TORCH_CUDA_ARCH_LIST is required for source builds")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", architectures)
        manifest["architectures"] = architectures
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
    cache = Path(os.environ.get("HIER_OT_BUILD_CACHE", str(Path.home() / ".cache/hier_ot/extensions")))
    directory = cache / name / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".build.flock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if cuda:
            includes, flags = wheel_cuda_paths(directory)
            options["extra_include_paths"] = [*options.get("extra_include_paths", []), *includes]
            options["extra_ldflags"] = [*options.get("extra_ldflags", []), *flags]
        options["extra_ldflags"] = [*options.get("extra_ldflags", []), f"-L{Path(sys.prefix) / 'lib'}"]
        # CN: 外层进程锁保证此处没有活跃构建，可清理被中断构建留下的 Torch 锁。
        # EN: The outer process lock excludes active builds, allowing removal of stale Torch locks.
        (directory / "lock").unlink(missing_ok=True)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[source-build] {name}: {directory}", flush=True)
        module = load(name=name, sources=sources, build_directory=str(directory), **options)
    _LOADED[name] = module
    return module


class LazyNativeModule:
    """
    CN: 延迟编译导入时注册的求解器，普通 Python 包导入不触发编译。
    EN: Defer compilation of import-registered solvers so ordinary Python imports do not build kernels.
    """

    def __init__(self, name):
        """
        CN: 保存待加载的扩展模块名。
        EN: Store the extension module name to load.
        """
        self.name = name

    def __getattr__(self, attribute):
        """
        CN: 首次使用扩展属性时加载实际模块。
        EN: Load the actual module when an extension attribute is first used.
        """
        return getattr(load_native_module(self.name), attribute)


def load_native_module(module_name):
    """
    CN: 公开模式导入发行扩展，开发模式仅构建当前工作区源码且不回退。
    EN: Import release extensions publicly, or exclusively build the current checkout in development mode.
    """
    if not source_mode():
        return importlib.import_module(module_name)
    name = module_name.rsplit(".", 1)[-1]
    root = Path(__file__).resolve().parent / "_native"
    options = {"extra_cflags": ["-O3"], "extra_cuda_cflags": ["-O3"]}
    if name == "hierot_norm_cost_scan_ext":
        directory = root / "norm_cost_scan"
        files = ["norm_cost_scan_ext.cpp", "norm_cost_scan_kernel.cu"]
    elif name.startswith("hierot_generic_") and name.endswith("_ext"):
        component = name[len("hierot_generic_"):-len("_ext")]
        if component not in {"initial", "initial_small", "pricing", "pricing_small", "cost"}:
            raise ValueError(f"Unknown generic scan component: {component}")
        directory = root / "generic_scan"
        files = [component + ".cu"]
        if component in ("cost", "pricing_small"):
            options["extra_ldflags"] = ["-lcublas"]
    elif name == "_support_sparsifier_ext":
        directory = root / "support_sparsifier"
        files = ["support_sparsifier_bindings.cpp", "support_sparsifier_lct.cpp"]
        options["extra_cflags"].append("-std=c++17")
    elif name == "pycupdlpx":
        directory = root / "cupdlpx"
        files = ["pycupdlpx/cupdlpx_bindings.cpp", *[f"cupdlpx/{file}" for file in ("io.c", "utils.cu", "solver.cu", "cupdlpx.c")]]
        options.update(extra_include_paths=[str(directory / "cupdlpx")],
                       extra_ldflags=["-lcublas", "-lcusparse", "-lz"], extra_cflags=["-O3", "-fpermissive"])
    else:
        raise ValueError(f"Unknown HELLO source extension: {module_name}")
    return load_source_extension(name, [str(directory / file) for file in files], **options)
