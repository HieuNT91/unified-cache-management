#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/.env.l20}"
ucm_load_server_env "$CODE_ROOT"
command="${1:-}"
if [[ -z "$command" ]]; then
  echo 'Usage: l20_ruler_data.sh {configure|prepare|detach|resume|stop|status|status_same_count|report}' >&2
  exit 2
fi
shift
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/longbench_a800_control.py" "$command" \
  --role ruler --root "${EXPERIMENT_DIR:-$CODE_ROOT/outputs/ruler-l20-tp2-v2}" \
  --model "${MODEL_PATH:-$CODE_ROOT/models/Qwen3-32B}" \
  --data "${RULER_PATH:-$CODE_ROOT/.cache/vendor/RULER}" \
  --prepared "${PREPARED_DIR:-$CODE_ROOT/inputs/ruler-l20-tp2-v2}" \
  --cache-root "${CACHE_ROOT:-$CODE_ROOT/.cache/ruler-l20-tp2-v2}" \
  --tp "${TP:-2}" --devices "${GPU_DEVICES:-0,1,2,3,4,5,6,7,8,9}" --samples-per-task "${SAMPLES_PER_TASK:-100}" --seed "${SEED:-42}" "$@"
