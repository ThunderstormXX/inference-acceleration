#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
export INFERENCE_PYTHON="${INFERENCE_PYTHON:-$PROJECT_ROOT/.venv-vllm/bin/python}"
export VLLM_CACHE_ROOT="$PROJECT_ROOT/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/.cache/torchinductor"
run_python "$SCRIPT_DIR/vllm.py" "$@"
