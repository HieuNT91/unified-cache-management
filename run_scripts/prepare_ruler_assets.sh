#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/config.sh"
PYTHON_BIN="$(command -v -- "$PYTHON_BIN")"
cd /tmp
exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" \
    "$RUN_SCRIPTS_DIR/remote_prophetkv/assets.py"
