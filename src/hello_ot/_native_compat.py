from __future__ import annotations

import torch


NATIVE_TORCH_VERSION = "2.7.1"
NATIVE_CUDA_RUNTIME = "11.8"
NATIVE_CXX11_ABI = True


def native_runtime_compatibility() -> tuple[bool, str]:
    """
    CN: 检查预编译 native 扩展所要求的 PyTorch/CUDA ABI 组合。
    EN: Check the PyTorch/CUDA ABI combination required by the prebuilt native extensions.
    """
    torch_version = str(torch.__version__).split("+", 1)[0]
    cuda_runtime = torch.version.cuda
    cxx11_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    compatible = (
        torch_version == NATIVE_TORCH_VERSION
        and cuda_runtime == NATIVE_CUDA_RUNTIME
        and cxx11_abi == NATIVE_CXX11_ABI
    )
    detail = (
        f"expected torch={NATIVE_TORCH_VERSION} with CUDA {NATIVE_CUDA_RUNTIME} "
        f"and cxx11_abi={NATIVE_CXX11_ABI}; found torch={torch.__version__} "
        f"with CUDA {cuda_runtime} and cxx11_abi={cxx11_abi}"
    )
    return compatible, detail


def require_native_runtime_compatibility() -> None:
    """
    CN: 在导入 native 扩展前拒绝未经验证的 ABI，避免含糊的动态链接错误。
    EN: Reject an unvalidated ABI before importing native extensions to avoid opaque linker errors.
    """
    compatible, detail = native_runtime_compatibility()
    if not compatible:
        raise RuntimeError(
            "The native backend requires the v0.1.0 binary compatibility stack: "
            f"{detail}. The explicit torch backend remains available."
        )
