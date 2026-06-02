#!/bin/bash -l
set -euo pipefail
POLARIS_ROOT="${POLARIS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${POLARIS_PYTHON:-python}"
export POLARIS_ROOT
"${PYTHON_BIN}" "${POLARIS_ROOT}/scripts/verify_install.py"
