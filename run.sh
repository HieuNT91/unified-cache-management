#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT"
export ENABLE_SPARSE=TRUE VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PLATFORM=cuda
export VLLM_USE_REROPE=0
exec "${PYTHON_BIN:-python3}" "$ROOT/run.py" "$@"
