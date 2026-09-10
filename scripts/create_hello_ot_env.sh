#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -gt 1 ]]; then
    echo "usage: bash scripts/create_hello_ot_env.sh [environment_name]" >&2
    exit 2
fi
ENV_NAME="${1:-hello_ot}"
if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "error: environment name may contain only letters, digits, '.', '_' and '-'" >&2
    exit 2
fi
MAMBA_CACHE_DIR="${HELLO_OT_MAMBA_CACHE_DIR:-/tmp/hello-ot-mamba-cache}"

run_in_environment() {
    if [[ "$ENV_MANAGER" == "conda" ]]; then
        XDG_CACHE_HOME="$MAMBA_CACHE_DIR" conda run --no-capture-output -n "$ENV_NAME" "$@"
    else
        XDG_CACHE_HOME="$MAMBA_CACHE_DIR" micromamba run -n "$ENV_NAME" "$@"
    fi
}

environment_exists() {
    local manager="$1"
    "$manager" run -n "$ENV_NAME" python3 -c 'pass' >/dev/null 2>&1
}

available_managers=()
for manager in conda micromamba; do
    if command -v "$manager" >/dev/null 2>&1; then
        available_managers+=("$manager")
        if environment_exists "$manager"; then
            echo "error: environment '$ENV_NAME' already exists in $manager" >&2
            echo "remove or rename the existing environment, then run this script again" >&2
            exit 1
        fi
    fi
done

if [[ "${#available_managers[@]}" -eq 0 ]]; then
    echo "error: conda or micromamba is required" >&2
    exit 1
fi

# CN: 普通用户更常安装 Conda；没有 Conda 时使用 Micromamba。
# EN: Prefer Conda for general users and fall back to Micromamba.
ENV_MANAGER="${available_managers[0]}"

mkdir -p "$MAMBA_CACHE_DIR/pip"

echo "creating clean environment: $ENV_NAME"
XDG_CACHE_HOME="$MAMBA_CACHE_DIR" "$ENV_MANAGER" create \
    -n "$ENV_NAME" -c conda-forge python=3.12 pip -y

echo "installing PyTorch 2.7.1 with CUDA 11.8 runtime"
PIP_CACHE_DIR="$MAMBA_CACHE_DIR/pip" \
    run_in_environment \
    python3 -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118

echo "installing the shared numerical environment"
PIP_CACHE_DIR="$MAMBA_CACHE_DIR/pip" \
    run_in_environment python3 -m pip install \
    numpy==1.26.4 scipy==1.15.3

echo "installing HELLO-OT from: $REPO_ROOT"
PIP_CACHE_DIR="$MAMBA_CACHE_DIR/pip" \
    run_in_environment \
    python3 -m pip install "$REPO_ROOT"

run_in_environment python3 - <<'PY'
import platform

import numpy
import scipy
import torch

if platform.python_version_tuple()[:2] != ("3", "12"):
    raise SystemExit(f"expected Python 3.12; found {platform.python_version()}")
if torch.__version__.split("+", 1)[0] != "2.7.1" or torch.version.cuda != "11.8":
    raise SystemExit(
        "expected torch==2.7.1+cu118; "
        f"found torch={torch.__version__} cuda={torch.version.cuda}"
    )
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit("expected the PyTorch 2.7.1 C++11 ABI")
if numpy.__version__ != "1.26.4" or scipy.__version__ != "1.15.3":
    raise SystemExit(
        "expected numpy==1.26.4 and scipy==1.15.3; "
        f"found numpy={numpy.__version__} scipy={scipy.__version__}"
    )
print(
    f"runtime verified: python={platform.python_version()} "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"numpy={numpy.__version__} scipy={scipy.__version__}"
)
PY

echo "running the torch-backend quickstart"
(
    cd "$REPO_ROOT"
    run_in_environment python3 examples/quickstart.py --backend torch
)

echo "environment ready: $ENV_MANAGER activate $ENV_NAME"
echo "optional native backend: bash scripts/install_native.sh $ENV_NAME"
echo "optional GPU JAX: bash scripts/install_jax_gpu.sh $ENV_NAME"
