#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -gt 1 ]]; then
    echo "usage: bash scripts/install_jax_gpu.sh [environment_name]" >&2
    exit 2
fi
ENV_NAME="${1:-hello_ot}"
CACHE_ROOT="${HELLO_OT_MAMBA_CACHE_DIR:-/tmp/hello-ot-mamba-cache}"
JAX_WHEEL_INDEX="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

run_with_manager() {
    local manager="$1"
    shift
    if [[ "$manager" == "conda" ]]; then
        conda run --no-capture-output -n "$ENV_NAME" "$@"
    else
        micromamba run -n "$ENV_NAME" "$@"
    fi
}

select_manager() {
    local matches=()
    local manager
    for manager in conda micromamba; do
        if command -v "$manager" >/dev/null 2>&1 && \
            run_with_manager "$manager" python3 -c \
            'import sys, torch; raise SystemExit(not (sys.version_info[:2] == (3, 12) and torch.__version__.split("+", 1)[0] == "2.7.1" and torch.version.cuda == "11.8"))' \
            >/dev/null 2>&1; then
            matches+=("$manager")
        fi
    done
    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "error: expected exactly one compatible Conda or Micromamba environment named '$ENV_NAME'" >&2
        exit 1
    fi
    ENV_MANAGER="${matches[0]}"
}

select_manager
mkdir -p "$CACHE_ROOT/pip"
ENV_PREFIX="$(run_with_manager "$ENV_MANAGER" python3 -c 'import sys; print(sys.prefix)')"
ENV_PYTHON="$ENV_PREFIX/bin/python3"

echo "installing GPU JAX and reproduction dependencies in $ENV_MANAGER/$ENV_NAME"
PIP_CACHE_DIR="$CACHE_ROOT/pip" run_with_manager "$ENV_MANAGER" \
    python3 -m pip install \
    numpy==1.26.4 scipy==1.15.3 ml-dtypes==0.5.4 \
    jax==0.4.25 'jaxlib==0.4.25+cuda11.cudnn86' ott-jax==0.4.6 \
    chex==0.1.86 jaxopt==0.8.5 lineax==0.0.4 optax==0.2.2 \
    equinox==0.11.4 \
    -f "$JAX_WHEEL_INDEX"
PIP_CACHE_DIR="$CACHE_ROOT/pip" run_with_manager "$ENV_MANAGER" \
    python3 -m pip install pykeops==2.3 'matplotlib>=3.8' 'pytest>=8'

# CN: PyTorch 2.7.1 使用 cuDNN 9；JAX CUDA 11 使用独立的 cuDNN 8。
# EN: PyTorch 2.7.1 uses cuDNN 9; CUDA 11 JAX uses an isolated cuDNN 8.
JAX_CUDA_ROOT="$ENV_PREFIX/hello_ot_jax_cuda"
JAX_CUDNN_LIB="$JAX_CUDA_ROOT/nvidia/cudnn/lib"
if [[ ! -f "$JAX_CUDNN_LIB/libcudnn.so.8" ]]; then
    PIP_CACHE_DIR="$CACHE_ROOT/pip" "$ENV_PYTHON" -m pip install \
        --target "$JAX_CUDA_ROOT" --no-deps nvidia-cudnn-cu11==8.9.6.50
fi
TORCH_BIN="$($ENV_PYTHON -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "bin")')"
PYTORCH_CUDA_LIBS="$($ENV_PYTHON -c '
from pathlib import Path
import torch

site_packages = Path(torch.__file__).resolve().parent.parent
print(":".join(str(path) for path in sorted((site_packages / "nvidia").glob("*/lib")) if path.is_dir()))
')"

[[ -x "$TORCH_BIN/ptxas" ]] || {
    echo "error: CUDA 11.8 ptxas is missing from the PyTorch installation" >&2
    exit 1
}
[[ -f "$JAX_CUDNN_LIB/libcudnn.so.8" ]] || {
    echo "error: the isolated cuDNN 8 installation is incomplete" >&2
    exit 1
}

ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$ENV_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"
cat > "$ACTIVATE_DIR/hello_ot_jax.sh" <<EOF
#!/usr/bin/env bash
export _HELLO_OT_PATH_BEFORE_JAX="\${PATH-}"
export _HELLO_OT_LD_LIBRARY_PATH_BEFORE_JAX="\${LD_LIBRARY_PATH-}"
export PATH="$TORCH_BIN:\${PATH-}"
export LD_LIBRARY_PATH="$JAX_CUDNN_LIB:$PYTORCH_CUDA_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
cat > "$DEACTIVATE_DIR/hello_ot_jax.sh" <<'EOF'
#!/usr/bin/env bash
export PATH="${_HELLO_OT_PATH_BEFORE_JAX-}"
if [[ -n "${_HELLO_OT_LD_LIBRARY_PATH_BEFORE_JAX-}" ]]; then
    export LD_LIBRARY_PATH="$_HELLO_OT_LD_LIBRARY_PATH_BEFORE_JAX"
else
    unset LD_LIBRARY_PATH
fi
unset _HELLO_OT_PATH_BEFORE_JAX _HELLO_OT_LD_LIBRARY_PATH_BEFORE_JAX
EOF

echo "checking PyTorch, JAX and OTT-JAX on the GPU"
PATH="$TORCH_BIN:$PATH" \
LD_LIBRARY_PATH="$JAX_CUDNN_LIB:$PYTORCH_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$ENV_PYTHON" - <<'PY'
import jax
import jax.numpy as jnp
import ott
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access a CUDA GPU")
torch_value = torch.ones(16, device="cuda").sum().item()
jax_devices = [device for device in jax.devices() if device.platform == "gpu"]
if not jax_devices:
    raise SystemExit("JAX cannot access a CUDA GPU")
jax_value = float(jnp.ones(16).sum().block_until_ready())
if torch_value != 16.0 or jax_value != 16.0:
    raise SystemExit("GPU arithmetic verification failed")
if torch.backends.cudnn.version() != 90100:
    raise SystemExit(
        "PyTorch cuDNN changed unexpectedly; "
        f"found {torch.backends.cudnn.version()}"
    )
print(
    f"GPU JAX verified: torch={torch.__version__} cudnn={torch.backends.cudnn.version()} "
    f"jax={jax.__version__} ott={getattr(ott, '__version__', 'unknown')} "
    f"device={jax_devices[0]}"
)
PY
run_with_manager "$ENV_MANAGER" python3 -m pip check

echo "GPU JAX ready: $ENV_MANAGER activate $ENV_NAME"
