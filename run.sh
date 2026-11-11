#!/usr/bin/env bash
set -euo pipefail

# Experiment hyperparameters: edit here; paths and GPU UUIDs belong in config/.env.
MODEL_NAME="Qwen3-32B"
TP=2                               # Supported: 1, 2 or 4
NUMBER_GPUS=4
CHUNK_SIZE=4096                     # 4096 or 2048
GPU_MEMORY_UTILIZATION=0.95         # Increase only if measured KV capacity needs it
RUN_NAME="adatree-v1.1"
RULER_SAMPLES_PER_TASK=500
PREPARATION_SEED=42
LONGBENCH_ANSWER_SEEDS=(0 1)
# Frozen output policy: thinking <=8192, total <=10000; loop32..128 repeated4.

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/config/.env}"
ucm_load_server_env "$CODE_ROOT"
command="${1:?Usage: run.sh configure|start|resume|stop|status DATASET}"
dataset="${2:?Dataset must be ruler or longbench-v2}"
shift 2
[[ $# == 0 ]] || { echo 'Expected a command and dataset only.' >&2; exit 2; }
[[ "$MODEL_NAME" == Qwen3-32B ]] || { echo 'This run supports Qwen3-32B only.' >&2; exit 2; }
[[ "$CHUNK_SIZE" == 4096 || "$CHUNK_SIZE" == 2048 ]] || { echo 'CHUNK_SIZE must be 4096 or 2048.' >&2; exit 2; }
case "$dataset" in
  ruler)
    prepared="${RULER_PREPARED:-}"
    seeds=() ;;
  longbench-v2)
    prepared="${LONGBENCH_PREPARED:-}"
    seeds=(--answer-seeds "${LONGBENCH_ANSWER_SEEDS[@]}") ;;
  *) echo 'Dataset must be ruler or longbench-v2.' >&2; exit 2 ;;
esac
python_command=("${PYTHON_BIN:?Set PYTHON_BIN in config/.env}")
[[ -z "${PYTHON_WRAPPER:-}" ]] || python_command=("$PYTHON_WRAPPER" "${python_command[@]}")
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
run_label="$RUN_NAME/$dataset/chunk$CHUNK_SIZE"
if [[ "$dataset" == longbench-v2 ]]; then
  seed_label="$(IFS=-; echo "${LONGBENCH_ANSWER_SEEDS[*]}")"
  run_label="$run_label-seeds-$seed_label"
fi
result="${RESULTS_DIR:-$CODE_ROOT/results}/$run_label"
if [[ "$command" == status ]]; then
  exec "${python_command[@]}" "$CODE_ROOT/status.py" "$result"
fi
if [[ "$command" == stop ]]; then
  exec "${python_command[@]}" "$CODE_ROOT/scripts/collection/run_controller.py" stop --root "$result"
fi
case "$command" in configure|start|resume|stop) ;; *) echo 'Unknown run command.' >&2; exit 2 ;; esac
percentages=(5 20 40)
[[ "$CHUNK_SIZE" != 2048 ]] || percentages=(5 20 60)
exec "${python_command[@]}" "$CODE_ROOT/scripts/collection/run_controller.py" "$command" \
  --adatree-v1 --adatree-thinking-8192 --followup-nocache \
  --dataset "$dataset" --root "$result" --model "${MODEL_PATH:?Set MODEL_PATH in config/.env}" \
  --prepared "${prepared:?Set RULER_PREPARED or LONGBENCH_PREPARED in config/.env}" \
  --cache-root "${CACHE_ROOT:?Set CACHE_ROOT in config/.env}/$run_label" \
  --devices "${GPU_UUIDS:?Set explicit GPU_UUIDS in config/.env}" --tp "$TP" --number-gpus "$NUMBER_GPUS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --kv-chunk-size "$CHUNK_SIZE" \
  --samples-per-task "$RULER_SAMPLES_PER_TASK" --seed "$PREPARATION_SEED" \
  "${seeds[@]}" --percentages "${percentages[@]}"
