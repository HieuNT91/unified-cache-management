#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/config/.env.a800.tree-prophetkv}"
ucm_load_server_env "$CODE_ROOT"
command="${1:?Usage: a800_longbench_tree_prophetkv.sh configure|start|detach|resume|stop|status|report}"
shift
: "${EXPERIMENT_DIR:?Set a fresh EXPERIMENT_DIR}" "${CACHE_ROOT:?Set a fresh CACHE_ROOT}"
read -r -a percentages <<< "${PERCENTAGES:-5 20 40 60}"
seed_args=()
if [[ -n "${ANSWER_SEEDS:-}" ]]; then
  read -r -a answer_seeds <<< "$ANSWER_SEEDS"
  seed_args=(--answer-seeds "${answer_seeds[@]}")
fi
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/tree_prophetkv_data.py" "$command" \
  --dataset longbench-v2 --root "$EXPERIMENT_DIR" --model "${MODEL_PATH:?Set MODEL_PATH}" \
  --prepared "${PREPARED_DIR:-$EXPERIMENT_DIR/prepared}" --cache-root "$CACHE_ROOT" \
  --devices "${GPU_PRIMARY:?Set four full A800 UUIDs in config/.env.a800.tree-prophetkv}" \
  --tp "${TP:-2}" --seed "${SEED:-42}" --answers-per-question "${ANSWERS_PER_QUESTION:-2}" \
  --percentages "${percentages[@]}" "${seed_args[@]}" "$@"
