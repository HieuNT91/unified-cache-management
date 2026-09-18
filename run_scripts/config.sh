#!/usr/bin/env bash
# Edit paths here, or export these variables before invoking a launch script.
RUN_SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$RUN_SCRIPTS_DIR/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
# A local, complete Qwen/Qwen3-32B checkpoint, including tokenizer files.
export MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-32B}"
export RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/.results/qwen3-32b-prophetkv-niah-multivalue}"
export CACHE_ROOT="${CACHE_ROOT:-$RESULT_ROOT/cache}"
export RULER_ROOT="${RULER_ROOT:-$REPO_ROOT/benchmarks/vendor/RULER}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export CHUNK_SIZE="${CHUNK_SIZE:-4096}"
# Physical indices on the REMOTE machine; resolved and pinned to UUIDs there.
# All eight form one TP=8 engine. No other GPU workloads are launched.
export GPU_INDICES="${GPU_INDICES:-0,1,2,3,4,5,6,7}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-1800}"
# Optional: export CUDA_HOME=/path/to/cuda before launching. Existing PATH and
# LD_LIBRARY_PATH are preserved; no packages are installed or upgraded.
