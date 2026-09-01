from __future__ import annotations

import argparse
import importlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from hello_ot._native_compat import native_runtime_compatibility


_NATIVE_MODULES = {
    "inner_product_scan": "hello_ot._native.inner_product_scan.hierot_inner_product_scan_ext",
    "norm_cost_scan": "hello_ot._native.norm_cost_scan.hierot_norm_cost_scan_ext",
    "support_sparsifier": "hello_ot._native.support_sparsifier._support_sparsifier_ext",
    "cupdlpx": "hello_ot._native.pycupdlpx",
}


def _distribution_version() -> str:
    """
    CN: 返回已安装发行包版本；源码树运行时使用明确占位符。
    EN: Return the installed distribution version, or an explicit placeholder from a source tree.
    """
    try:
        return metadata.version("hello-ot")
    except metadata.PackageNotFoundError:
        return "source-tree"


def _is_wsl2() -> bool:
    """
    CN: 通过 Linux kernel release 标识识别 WSL2，不读取用户路径或环境变量。
    EN: Detect WSL2 from the Linux kernel release without reading user paths or environment variables.
    """
    release = platform.release().lower()
    try:
        release = f"{release} {Path('/proc/sys/kernel/osrelease').read_text(encoding='utf-8').lower()}"
    except OSError:
        pass
    return "microsoft" in release or "wsl2" in release


def _native_status() -> dict[str, dict[str, Any]]:
    """
    CN: 独立导入四个 native 扩展，使缺失或动态链接错误可定位。
    EN: Import all four native extensions independently so missing modules and linker failures are attributable.
    """
    status: dict[str, dict[str, Any]] = {}
    for name, module_name in _NATIVE_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on the installed binary stack
            status[name] = {
                "available": False,
                "module": module_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        else:
            status[name] = {
                "available": True,
                "module": module_name,
                "file": str(getattr(module, "__file__", "")),
            }
    return status


def collect_diagnostics() -> dict[str, Any]:
    """
    CN: 收集可公开分享的运行时与 native 扩展诊断，不包含用户名或环境变量。
    EN: Collect shareable runtime and native-extension diagnostics without usernames or environment variables.
    """
    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    native_compatible, native_compatibility_detail = native_runtime_compatibility()
    return {
        "hello_ot": _distribution_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wsl2": _is_wsl2(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "native_runtime_compatible": native_compatible,
        "native_runtime_compatibility": native_compatibility_detail,
        "devices": devices,
        "native_extensions": _native_status(),
    }


def _print_human(report: dict[str, Any]) -> None:
    """
    CN: 以适合安装排错的紧凑格式打印诊断。
    EN: Print diagnostics in a compact installation-troubleshooting format.
    """
    print(f"hello-ot: {report['hello_ot']}")
    print(f"python: {report['python']}")
    print(f"platform: {report['platform']} wsl2={str(report['wsl2']).lower()}")
    print(f"torch: {report['torch']} cuda_runtime={report['torch_cuda_runtime']}")
    print(f"cuda_available: {str(report['cuda_available']).lower()}")
    print(
        "native_runtime_compatible: "
        f"{str(report['native_runtime_compatible']).lower()} "
        f"({report['native_runtime_compatibility']})"
    )
    for device in report["devices"]:
        capability = ".".join(str(value) for value in device["compute_capability"])
        print(f"gpu[{device['index']}]: {device['name']} sm={capability}")
    for name, item in report["native_extensions"].items():
        if item["available"]:
            print(f"native.{name}: available")
        else:
            print(f"native.{name}: unavailable ({item['error_type']}: {item['error']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the HELLO-OT runtime and native extensions.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="exit nonzero unless CUDA and all native extensions are available",
    )
    args = parser.parse_args()
    report = collect_diagnostics()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    native_ok = all(item["available"] for item in report["native_extensions"].values())
    if args.require_native and (
        not report["cuda_available"] or not report["native_runtime_compatible"] or not native_ok
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
