#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME=/home/thnguyen/unified-cache-management/.tools/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export WATCHDOG_SECONDS=1800
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /tmp
exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' \
  /home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python \
  -u "$TASK_DIR/suite.py" "$@"
