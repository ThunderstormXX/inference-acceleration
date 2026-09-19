#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
export INFERENCE_PYTHON="${INFERENCE_PYTHON:-$PROJECT_ROOT/.venv-transformers/bin/python}"
run_python "$SCRIPT_DIR/transformers.py" "$@"
