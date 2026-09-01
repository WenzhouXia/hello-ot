from __future__ import annotations

import os

from setuptools import setup


INNER_PRODUCT_SCAN_EXT_DIR = "src/hello_ot/_native/inner_product_scan"
NORM_COST_SCAN_EXT_DIR = "src/hello_ot/_native/norm_cost_scan"
RESTRICTED_OT_EXT_DIR = "src/hello_ot/_native/support_sparsifier"
CUPDLPX_DIR = "src/hello_ot/_native/cupdlpx/cupdlpx"
CUPDLPX_BINDING = "src/hello_ot/_native/cupdlpx/pycupdlpx/cupdlpx_bindings.cpp"


def _torch_python_extra_link_args() -> list[str]:
    """CN: 某些 conda 环境把 libtorch_python.so 装在 $CONDA_PREFIX/lib 而非
    site-packages/torch/lib，链接阶段需要额外的 -L 才能找到，否则报
    'cannot find -ltorch_python'。
    EN: Some conda envs install libtorch_python.so under $CONDA_PREFIX/lib instead
        of site-packages/torch/lib, so the linker needs an extra -L to resolve
        -ltorch_python.
    """
    try:
        import torch
    except Exception:  # pragma: no cover
        return []
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(os.path.join(lib_dir, "libtorch_python.so")):
        return []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_lib = os.path.join(conda_prefix, "lib", "libtorch_python.so")
    if conda_prefix and os.path.exists(conda_lib):
        return [f"-L{os.path.join(conda_prefix, 'lib')}"]
    return []


def _native_extensions() -> tuple[list[object], type]:
    """
    CN: 仅在显式请求 native 构建时导入 PyTorch/pybind11 构建依赖。
    EN: Import PyTorch/pybind11 build dependencies only for an explicit native build.
    """
    import pybind11
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    extra_link_args = _torch_python_extra_link_args()
    native_rpath_args = ["-Wl,-rpath,$ORIGIN", "-Wl,--disable-new-dtags"]
    cupdlpx_extension = CUDAExtension(
        name="hello_ot._native.pycupdlpx",
        sources=[
            CUPDLPX_BINDING,
            f"{CUPDLPX_DIR}/io.c",
            f"{CUPDLPX_DIR}/utils.cu",
            f"{CUPDLPX_DIR}/solver.cu",
            f"{CUPDLPX_DIR}/cupdlpx.c",
        ],
        include_dirs=[CUPDLPX_DIR, pybind11.get_include()],
        libraries=["cublas", "cusparse", "z"],
        # CN: CUDAExtension 用 C++ driver 编译上游 .c；仅放宽合法的 void* 转换。
        # EN: CUDAExtension compiles upstream .c with C++; only relax valid void* conversions.
        extra_compile_args={"cxx": ["-O3", "-fpermissive"], "nvcc": ["-O3"]},
        extra_link_args=native_rpath_args + extra_link_args,
    )
    support_extension = CppExtension(
        name="hello_ot._native.support_sparsifier._support_sparsifier_ext",
        sources=[
            f"{RESTRICTED_OT_EXT_DIR}/support_sparsifier_bindings.cpp",
            f"{RESTRICTED_OT_EXT_DIR}/support_sparsifier_lct.cpp",
        ],
        extra_compile_args=["-O3", "-std=c++17"],
        extra_link_args=extra_link_args,
    )
    if os.environ.get("HELLO_OT_BUILD_SUPPORT_SPARSIFIER_ONLY", "0") == "1":
        return [support_extension], BuildExtension
    return [
        CUDAExtension(
            name="hello_ot._native.inner_product_scan.hierot_inner_product_scan_ext",
            sources=[
                f"{INNER_PRODUCT_SCAN_EXT_DIR}/inner_product_scan_ext.cpp",
                f"{INNER_PRODUCT_SCAN_EXT_DIR}/inner_product_stats_kernel.cu",
                f"{INNER_PRODUCT_SCAN_EXT_DIR}/inner_product_scan_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
            extra_link_args=["-lcublas"] + extra_link_args,
        ),
        CUDAExtension(
            name="hello_ot._native.norm_cost_scan.hierot_norm_cost_scan_ext",
            sources=[
                f"{NORM_COST_SCAN_EXT_DIR}/norm_cost_scan_ext.cpp",
                f"{NORM_COST_SCAN_EXT_DIR}/norm_cost_scan_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
            extra_link_args=extra_link_args,
        ),
        support_extension,
        cupdlpx_extension,
    ], BuildExtension


# CN: 默认发行包是纯 Python；native 扩展必须由用户明确选择构建。
# EN: The default distribution is pure Python; native extensions require an explicit opt-in.
if os.environ.get("HELLO_OT_BUILD_NATIVE", "0") == "1":
    _EXT_MODULES, _BUILD_EXT = _native_extensions()
    _CMDCLASS = {"build_ext": _BUILD_EXT}
else:
    _EXT_MODULES = []
    _CMDCLASS = {}


setup(
    ext_modules=_EXT_MODULES,
    cmdclass=_CMDCLASS,
)
