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
import torch

torch_version = torch.__version__.split("+", 1)[0]
compatible_torch = torch_version == "2.5.1" or torch_version.startswith("2.5.1.post")
if not compatible_torch or torch.version.cuda != "11.8":
    raise SystemExit(
        "native release wheels must be built with torch==2.5.1+cu118; "
        f"found torch={torch.__version__} cuda={torch.version.cuda}"
    )
print(f"building with torch={torch.__version__} cuda={torch.version.cuda}")
PY

echo "building with $(command -v nvcc)"

python3 -m pip wheel . --no-deps --no-build-isolation --wheel-dir "${output_directory}"
sha256sum "${output_directory}"/*.whl
