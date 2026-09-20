#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../_common.sh"
run_python "$SCRIPT_DIR/mtp_confidence.py" "$@"
exec "$PROJECT_ROOT/.venv-analysis/bin/python" "$SCRIPT_DIR/mtp_confidence.py" "$@" --render-only
