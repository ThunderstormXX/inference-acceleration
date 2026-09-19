#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
export INFERENCE_CAFFEINATE_ASSERTIONS=di
exec /usr/bin/caffeinate -di "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT_DIR/trace.py" "$@"
