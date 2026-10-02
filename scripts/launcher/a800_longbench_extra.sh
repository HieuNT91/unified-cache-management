#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
command="${1:-}"
if [[ -z "$command" ]]; then
  echo 'Usage: a800_longbench_extra.sh {configure|prepare|detach|resume|stop|status|status_same_count|report}' >&2
  exit 2
fi
shift
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/longbench_a800_control.py" "$command" \
  --role extra --root "${EXPERIMENT_DIR:-$CODE_ROOT/outputs/longbench-v2-a800-router-v1}" \
  --model "${MODEL_PATH:-$CODE_ROOT/models/Qwen3-32B}" \
  --data "${LONGBENCH_DATA:-$CODE_ROOT/.data/LongBench-v2/data.json}" \
  --prepared "${PREPARED_DIR:-$CODE_ROOT/inputs/longbench-v2-a800-router-v1}" \
  --cache-root "${CACHE_ROOT:-$CODE_ROOT/.cache/longbench-v2-a800-router-v1}" \
  --gpu-a "${GPU_A:-}" --gpu-b "${GPU_B:-}" "$@"
