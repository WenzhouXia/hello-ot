#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$#" -gt 1 ]]; then
    echo "usage: bash scripts/install_native.sh [environment_name]" >&2
    exit 2
fi
ENV_NAME="${1:-hello_ot}"
WHEEL_URL="https://github.com/WenzhouXia/hello-ot/releases/download/v0.1.0/hello_ot-0.1.0-cp312-cp312-linux_x86_64.whl"

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
            'import sys, torch; raise SystemExit(not (sys.version_info[:2] == (3, 12) and torch.__version__.split("+", 1)[0] == "2.7.1" and torch.version.cuda == "11.8" and bool(torch._C._GLIBCXX_USE_CXX11_ABI)))' \
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
echo "installing the native wheel in $ENV_MANAGER/$ENV_NAME"
run_with_manager "$ENV_MANAGER" python3 -m pip install \
    --no-cache-dir --no-deps --force-reinstall "$WHEEL_URL"

echo "checking native extensions"
run_with_manager "$ENV_MANAGER" python3 -m hello_ot.diagnose --require-native

echo "running the native-backend quickstart"
(
    cd "$REPO_ROOT"
    run_with_manager "$ENV_MANAGER" python3 examples/quickstart.py --backend native
)

echo "native backend ready: $ENV_MANAGER activate $ENV_NAME"
