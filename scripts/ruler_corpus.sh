#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$CODE_ROOT/scripts/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
command="${1:-}"
if [[ -z "$command" ]]; then echo 'Usage: ruler_corpus.sh {configure|prepare|verify|detach|resume|status|status_same_count|report|snapshot|train|replay|test|relocate} [options]' >&2; exit 2; fi
shift
prepared_args=()
case "$command" in
  train|snapshot|replay|test)
    # Offline readers inherit the original saved path when no override is set.
    if [[ -n "${PREPARED_DIR:-}" ]]; then prepared_args=(--prepared "$PREPARED_DIR"); fi
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    ;;
  *) prepared_args=(--prepared "${PREPARED_DIR:-$CODE_ROOT/inputs/ruler-corpus-v1}") ;;
esac
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/corpus_control.py" "$command" \
  --root "${EXPERIMENT_DIR:-$CODE_ROOT/outputs/ruler-corpus-v1}" \
  --model "${MODEL_PATH:-$CODE_ROOT/models/Qwen3-32B}" \
  "${prepared_args[@]}" \
  --cache-root "${CACHE_ROOT:-$CODE_ROOT/.cache/ruler-corpus-v1}" \
  --ruler "${RULER_ROOT:-$CODE_ROOT/.cache/vendor/RULER}" \
  --gpu-a "${GPU_A:-}" --gpu-b "${GPU_B:-}" --workers "${PREPARE_WORKERS:-4}" "$@"
