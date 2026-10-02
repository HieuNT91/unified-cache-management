#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/.env.a800}"
ucm_load_server_env "$CODE_ROOT"
command="${1:-}"
dataset="${2:-}"
if [[ "$command" != train && "$command" != evaluate ]] || [[ "$dataset" != ruler && "$dataset" != longbench-v2 && "$dataset" != both ]]; then
  echo 'Usage: router_training.sh {train|evaluate} {ruler|longbench-v2|both} [--tree PATH] [--output PATH]' >&2
  exit 2
fi
shift 2
if [[ "$command" == evaluate && " $* " != *" --tree "* && " $* " != *" --tree="* ]]; then
  echo 'evaluate requires --tree PATH (never refits)' >&2
  exit 2
fi
if [[ "$command" == train && ( " $* " == *" --tree "* || " $* " == *" --tree="* ) ]]; then
  echo 'Use evaluate for a saved tree' >&2
  exit 2
fi
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/train_router.py" \
  --dataset "$dataset" \
  --ruler-data "${RULER_DATA_FILE:-${DATA_IMPORT_DIR:-$CODE_ROOT/inputs/router-imports}/ruler-data.json}" \
  --longbench-data "${LONGBENCH_DATA_FILE:-${EXPERIMENT_DIR:-$CODE_ROOT/outputs/longbench-v2-a800-tp2-v2}/longbench-data.json}" \
  --output "${TRAINING_OUTPUT_DIR:-$CODE_ROOT/outputs/router-training}/$dataset/$command-$(date -u +%Y%m%dT%H%M%S)-$$" "$@"
