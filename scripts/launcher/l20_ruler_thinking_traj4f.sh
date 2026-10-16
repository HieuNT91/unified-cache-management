#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/.env.l20.thinking.traj4f}"
ucm_load_server_env "$CODE_ROOT"
command="${1:-}"
if [[ -z "$command" ]]; then
  echo 'Usage: l20_ruler_thinking_traj4f.sh {start|detach|resume|stop|status|report}' >&2
  exit 2
fi
shift
read -r -a percentages <<< "${PERCENTAGES:-1 5 10 20 30 40 60 80}"
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/l20_ruler_traj4f_data.py" "$command" \
  --root "${EXPERIMENT_DIR:-$CODE_ROOT/outputs/ruler-l20-tp2-thinking-traj4f-500-v1}" \
  --model "${MODEL_PATH:-$CODE_ROOT/models/Qwen3-32B}" \
  --prepared "${PREPARED_DIR:-$CODE_ROOT/inputs/ruler-l40-tp2-thinking-500-v3}" \
  --cache-root "${CACHE_ROOT:-$CODE_ROOT/.cache/ruler-l20-tp2-thinking-traj4f-500-v1}" \
  --devices "${GPU_DEVICES:?Set ten full L20 UUIDs in .env.l20.thinking.traj4f}" \
  --tp "${TP:-2}" --samples-per-task "${SAMPLES_PER_TASK:-500}" --seed "${SEED:-42}" \
  --percentages "${percentages[@]}" "$@"
