#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/config/.env.a800.adatree-v1.example}"
ucm_load_server_env "$CODE_ROOT"
command="${1:?Usage: a800_longbench_adatree_v1.sh start|resume|stop|status|report}"
shift
case "$command" in start|resume|stop|status|report) ;; *) echo 'Unsupported AdaTreeV1 command' >&2; exit 2 ;; esac
if [[ $# -ne 0 || -n "${KV_CHUNK_SIZE:-}" || "${TP:-2}" != 2 || "${SEED:-42}" != 42 || "${ANSWERS_PER_QUESTION:-2}" != 2 || "${ANSWER_SEEDS:-0 1}" != '0 1' ]]; then
  echo 'AdaTreeV1 requires TP2, seeds 0/1 and exact prepared layout; no extra overrides.' >&2
  exit 2
fi
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:?Set PYTHON_BIN}" "$CODE_ROOT/scripts/collection/tree_prophetkv_data.py" "$command" \
  --adatree-v1 --adatree-thinking-8192 --followup-nocache --dataset longbench-v2 --root "${EXPERIMENT_DIR:?Set fresh EXPERIMENT_DIR}" \
  --model "${MODEL_PATH:?Set MODEL_PATH}" --prepared "${PREPARED_DIR:?Set existing PREPARED_DIR}" \
  --cache-root "${CACHE_ROOT:?Set fresh CACHE_ROOT}" \
  --devices "${GPU_PRIMARY:?Set four full A800 UUIDs}" --tp 2 --seed 42 \
  --answers-per-question 2 --answer-seeds 0 1 --percentages 5 20 40
