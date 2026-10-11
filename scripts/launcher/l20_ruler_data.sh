#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
UCM_ENV_FILE="${UCM_ENV_FILE:-$CODE_ROOT/.env.l20}"
ucm_load_server_env "$CODE_ROOT"
command="${1:-}"
if [[ -z "$command" ]]; then
  echo 'Usage: l20_ruler_data.sh {configure|prepare|detach|resume|stop|status|status_same_count|report|merge|compute feature|compute-feature}' >&2
  exit 2
fi
shift
if [[ "$command" == compute ]]; then
  [[ "${1:-}" == feature ]] || { echo 'Usage: compute feature' >&2; exit 2; }
  shift
  command=compute-feature
fi
collection_args=(--collection-version v3)
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
extension_args=()
if [[ -n "${EXTEND_FROM:-}" ]]; then
  collection_args=(--collection-version legacy)
  extension_args=(--extend-from "$EXTEND_FROM")
  SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-500}"
  for key in EXPERIMENT_DIR PREPARED_DIR CACHE_ROOT; do
    [[ -n "${!key:-}" ]] || { echo "Set a fresh $key for EXTEND_FROM" >&2; exit 2; }
  done
fi
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/longbench_a800_control.py" "$command" \
  --role ruler --root "${EXPERIMENT_DIR:-$CODE_ROOT/outputs/ruler-l20-tp2-500-v3}" \
  --model "${MODEL_PATH:-$CODE_ROOT/models/Qwen3-32B}" \
  --data "${RULER_PATH:-$CODE_ROOT/.cache/vendor/RULER}" \
  --prepared "${PREPARED_DIR:-$CODE_ROOT/inputs/ruler-l20-tp2-500-v3}" \
  --cache-root "${CACHE_ROOT:-$CODE_ROOT/.cache/ruler-l20-tp2-500-v3}" \
  --tp "${TP:-2}" --devices "${GPU_DEVICES:-0,1,2,3,4,5,6,7,8,9}" --samples-per-task "${SAMPLES_PER_TASK:-500}" --seed "${SEED:-42}" "${collection_args[@]}" "${extension_args[@]}" "$@"
