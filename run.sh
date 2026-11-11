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
SMOKE_TP=4
SMOKE_NUMBER_GPUS=4
# Frozen output policy: thinking <=8192, total <=10000; loop32..128 repeated4.

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/config/.env}"
ucm_load_server_env "$CODE_ROOT"
command="${1:?Usage: run.sh download|prepare|configure|start|resume|stop|status|smoke|smoke-stop|smoke-status DATASET [PROMPT_ID ...]}"
dataset="${2:?Dataset must be ruler or longbench-v2}"
shift 2
[[ "$MODEL_NAME" == Qwen3-32B ]] || { echo 'This run supports Qwen3-32B only.' >&2; exit 2; }
[[ "$CHUNK_SIZE" == 4096 || "$CHUNK_SIZE" == 2048 ]] || { echo 'CHUNK_SIZE must be 4096 or 2048.' >&2; exit 2; }
case "$dataset" in
  ruler)
    source_path="${RULER_PATH:-}"
    prepared="${RULER_PREPARED:-}"
    seeds=() ;;
  longbench-v2)
    source_path="${LONGBENCH_DATA:-}"
    prepared="${LONGBENCH_PREPARED:-}"
    seeds=(--answer-seeds "${LONGBENCH_ANSWER_SEEDS[@]}") ;;
  *) echo 'Dataset must be ruler or longbench-v2.' >&2; exit 2 ;;
esac
python_command=("${PYTHON_BIN:?Set PYTHON_BIN in config/.env}")
[[ -z "${PYTHON_WRAPPER:-}" ]] || python_command=("$PYTHON_WRAPPER" "${python_command[@]}")
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ "$command" == download || "$command" == prepare ]]; then
  [[ $# == 0 ]] || { echo 'Data commands accept no prompt IDs.' >&2; exit 2; }
  [[ -n "$source_path" ]] || { echo 'Set RULER_PATH or LONGBENCH_DATA in config/.env.' >&2; exit 2; }
  data_args=()
  if [[ "$command" == prepare ]]; then
    data_args=(--model "${MODEL_PATH:?Set MODEL_PATH in config/.env}" \
      --output "${prepared:?Set RULER_PREPARED or LONGBENCH_PREPARED in config/.env}")
  fi
  exec "${python_command[@]}" "$CODE_ROOT/scripts/run.py" "$command" "$dataset" \
    --source "$source_path" --samples-per-task "$RULER_SAMPLES_PER_TASK" --seed "$PREPARATION_SEED" "${data_args[@]}"
fi
local_args=()
smoke_run=false
run_label="$RUN_NAME/$dataset/chunk$CHUNK_SIZE"
if [[ "$command" == smoke-stop || "$command" == smoke-status ]]; then
  run_label="$RUN_NAME/smoke/$dataset/chunk$CHUNK_SIZE"
  command="${command#smoke-}"
fi
if [[ "$command" == smoke ]]; then
  [[ $# -gt 0 ]] || { echo 'Smoke requires exact existing prepared prompt IDs.' >&2; exit 2; }
  TP=$SMOKE_TP; NUMBER_GPUS=$SMOKE_NUMBER_GPUS
  local_args=(--local-validation --validation-prompt-ids "$@")
  run_label="$RUN_NAME/smoke/$dataset/chunk$CHUNK_SIZE"
  smoke_run=true
  command=start
elif [[ $# != 0 ]]; then
  echo 'Only smoke accepts explicit prompt IDs.' >&2; exit 2
fi
if [[ "$dataset" == longbench-v2 ]]; then
  seed_label="$(IFS=-; echo "${LONGBENCH_ANSWER_SEEDS[*]}")"
  run_label="$run_label-seeds-$seed_label"
fi
result="${RESULTS_DIR:-$CODE_ROOT/results}/$run_label"
if $smoke_run && [[ -f "$result/settings.json" ]]; then command=resume; fi
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
  "${seeds[@]}" --percentages "${percentages[@]}" "${local_args[@]}"
