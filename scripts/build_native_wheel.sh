#!/usr/bin/env bash
set -euo pipefail

# CN: 正式 wheel 显式包含 A100、Ampere 消费卡、Ada 消费卡、H100 与最高目标的 PTX。
# EN: Release wheels explicitly include A100, consumer Ampere, consumer Ada, H100, and PTX at the highest target.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.6 8.9 9.0+PTX}"
export HELLO_OT_BUILD_NATIVE=1
export MAX_JOBS="${MAX_JOBS:-8}"

output_directory="${1:-dist}"
mkdir -p "${output_directory}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "native release wheel builds require nvcc from CUDA 11.8 on PATH" >&2
    exit 1
fi
nvcc_version="$(nvcc --version)"
if [[ "${nvcc_version}" != *"release 11.8"* ]]; then
    echo "native release wheel builds require nvcc 11.8; found:" >&2
    printf '%s\n' "${nvcc_version}" >&2
    exit 1
fi
python3 - <<'PY'
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        "native release wheels must be built with Python 3.12; "
        f"found Python {sys.version_info.major}.{sys.version_info.minor}"
    )
torch_version = torch.__version__.split("+", 1)[0]
if torch_version != "2.7.1" or torch.version.cuda != "11.8":
    raise SystemExit(
        "native release wheels must be built with torch==2.7.1+cu118; "
        f"found torch={torch.__version__} cuda={torch.version.cuda}"
    )
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit(
        "native release wheels must use the C++ ABI of the documented official "
        "PyTorch 2.7.1 cu118 wheel (_GLIBCXX_USE_CXX11_ABI=True); "
        "the selected build environment uses False"
    )
site_packages = Path(torch.__file__).resolve().parent.parent
required_headers = (
    site_packages / "nvidia/cuda_runtime/include/cuda_runtime.h",
    site_packages / "nvidia/cublas/include/cublas_v2.h",
    site_packages / "nvidia/cusparse/include/cusparse.h",
    site_packages / "nvidia/cusolver/include/cusolverDn.h",
)
missing_headers = [str(path) for path in required_headers if not path.is_file()]
if missing_headers:
    raise SystemExit(
        "native release wheel builds require CUDA development headers from the "
        "official PyTorch cu118 installation; missing: " + ", ".join(missing_headers)
    )
cuda_root = Path(CUDA_HOME or "")
if not (
    (cuda_root / "include/thrust/complex.h").is_file()
    or (cuda_root / "targets/x86_64-linux/include/thrust/complex.h").is_file()
):
    raise SystemExit(
        "native release wheel builds require CUDA 11.8 CCCL headers; "
        "install cuda-cccl=11.8.89"
    )
print(
    f"building with torch={torch.__version__} cuda={torch.version.cuda} "
    f"cxx11_abi={torch._C._GLIBCXX_USE_CXX11_ABI}"
)
PY

echo "building with $(command -v nvcc)"

python3 -m pip wheel . --no-deps --no-build-isolation --wheel-dir "${output_directory}"
sha256sum "${output_directory}"/*.whl
