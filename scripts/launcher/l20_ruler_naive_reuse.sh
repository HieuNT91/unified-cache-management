#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/.env.l20.naive}"
ucm_load_server_env "$CODE_ROOT"
command="${1:?Usage: l20_ruler_naive_reuse.sh configure|prepare|detach|resume|stop|status|report}"
shift
: "${EXPERIMENT_DIR:?Set a fresh EXPERIMENT_DIR}" "${PREPARED_DIR:?Set the existing full prepared cohort}" "${CACHE_ROOT:?Set a fresh CACHE_ROOT}"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/longbench_a800_control.py" "$command" \
  --naive-reuse --role ruler --root "$EXPERIMENT_DIR" \
  --model "${MODEL_PATH:?Set MODEL_PATH}" --data "${RULER_PATH:-$CODE_ROOT/.cache/vendor/RULER}" \
  --prepared "$PREPARED_DIR" --cache-root "$CACHE_ROOT" \
  --tp "${TP:-2}" --devices "${GPU_DEVICES:-0,1,2,3,4,5,6,7,8,9}" \
  --samples-per-task "${SAMPLES_PER_TASK:-500}" --seed "${SEED:-42}" "$@"
