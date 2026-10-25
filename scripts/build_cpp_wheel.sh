#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 - <<'PY'
import platform
import shutil
import sys
import pybind11
if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("C++ release builds require Linux x86-64 and Python 3.12")
if not shutil.which("g++"):
    raise SystemExit("C++ builds require g++ with C++17/OpenMP support; install the compiler first")
PY
python3 "$REPO_ROOT/scripts/build_extension.py" cpp --output "${1:-dist}"
