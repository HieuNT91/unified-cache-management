#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/config/.env.l20.tree-prophetkv}"
ucm_load_server_env "$CODE_ROOT"
command="${1:?Usage: l20_ruler_thinking_tree_prophetkv.sh configure|start|detach|resume|stop|status|report}"
shift
: "${EXPERIMENT_DIR:?Set a fresh EXPERIMENT_DIR}" "${CACHE_ROOT:?Set a fresh CACHE_ROOT}"
read -r -a percentages <<< "${PERCENTAGES:-5 20 40 60}"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/tree_prophetkv_data.py" "$command" \
  --dataset ruler --root "$EXPERIMENT_DIR" --model "${MODEL_PATH:?Set MODEL_PATH}" \
  --prepared "${PREPARED_DIR:-$EXPERIMENT_DIR/prepared}" --cache-root "$CACHE_ROOT" \
  --devices "${GPU_DEVICES:?Set ten full L20 UUIDs in config/.env.l20.tree-prophetkv}" \
  --tp "${TP:-2}" --samples-per-task "${SAMPLES_PER_TASK:-500}" --seed "${SEED:-42}" \
  --answers-per-question 1 --percentages "${percentages[@]}" "$@"
