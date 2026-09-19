#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export VLLM_CACHE_ROOT="$PROJECT_ROOT/.cache/vllm"
run_python "$SCRIPT_DIR/doctor.py" "$@"
