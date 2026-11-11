#!/usr/bin/env bash
set -euo pipefail

# Preparation hyperparameters; keep these aligned with run.sh for RULER runs.
RULER_SAMPLES_PER_TASK=500
PREPARATION_SEED=42

CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
dataset="${1:?Usage: prepare_data.sh ruler|longbench-v2}"
[[ $# == 1 ]] || { echo 'Expected a dataset only.' >&2; exit 2; }
case "$dataset" in
  ruler) source_path="${RULER_PATH:?Set RULER_PATH in config/.env}"; prepared="${RULER_PREPARED:-}" ;;
  longbench-v2) source_path="${LONGBENCH_DATA:?Set LONGBENCH_DATA in config/.env}"; prepared="${LONGBENCH_PREPARED:-}" ;;
  *) echo 'Dataset must be ruler or longbench-v2.' >&2; exit 2 ;;
esac
python_command=("${PYTHON_BIN:?Set PYTHON_BIN in config/.env}")
[[ -z "${PYTHON_WRAPPER:-}" ]] || python_command=("$PYTHON_WRAPPER" "${python_command[@]}")
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${python_command[@]}" "$CODE_ROOT/scripts/run.py" prepare "$dataset" \
  --source "$source_path" --model "${MODEL_PATH:?Set MODEL_PATH in config/.env}" \
  --output "${prepared:?Set RULER_PREPARED or LONGBENCH_PREPARED in config/.env}" \
  --samples-per-task "$RULER_SAMPLES_PER_TASK" --seed "$PREPARATION_SEED"
