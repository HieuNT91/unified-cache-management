#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
for script in "$RUN_SCRIPTS_DIR"/*.sh; do bash -n "$script"; done
PYTHON_BIN="$(command -v -- "$PYTHON_BIN")"
cd /tmp
exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON_BIN" -m unittest discover -s "$RUN_SCRIPTS_DIR/remote_prophetkv" -p 'test_*.py' -v
