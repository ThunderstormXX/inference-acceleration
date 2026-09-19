#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
cd "$PROJECT_ROOT"
run_python() {
  local interpreter="${INFERENCE_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
  if [[ ! -x "$interpreter" ]]; then
    echo "Python environment missing. Run bash scripts/setup/environment.sh" >&2
    return 1
  fi
  "$interpreter" "$@"
}
