#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the original local Qwen3-32B checkpoint}"
export RESULT_ROOT="${RESULT_ROOT:-$repo/.results/a800-qwen3-32b-expansion-nonthinking}"
export CACHE_ROOT="${CACHE_ROOT:-$RESULT_ROOT/cache}"
export RULER_ROOT="${RULER_ROOT:-$repo/benchmarks/vendor/RULER}"
export LONGBENCH_DATA="${LONGBENCH_DATA:-$repo/.data/LongBench-v2/data.json}"
export WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-1800}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTHONDONTWRITEBYTECODE=1
python_bin="$(command -v -- "$PYTHON_BIN")"
command="${1:?Usage: run.sh prepare|detach|status|stop|report|aggregate|plan|preflight|run|logs [0|1]}"
shift
cd /tmp
if [[ "$command" == aggregate ]]; then
    exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' "$python_bin" -u "$here/report.py" --root "$RESULT_ROOT" "$@"
fi
if [[ $# -gt 0 && ( "$1" == 0 || "$1" == 1 ) ]]; then
    export EXPANSION_JOB="$1"
    shift
    exec env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' "$python_bin" -u "$here/suite.py" "$command" "$@"
fi
if [[ "$command" == run || "$command" == logs ]]; then
    echo 'Choose job 0 (GPUs 4,5) or 1 (GPUs 6,7) for run/logs.' >&2
    exit 2
fi
for expansion_job in 0 1; do
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' EXPANSION_JOB="$expansion_job" \
        "$python_bin" -u "$here/suite.py" "$command" "$@"
done
