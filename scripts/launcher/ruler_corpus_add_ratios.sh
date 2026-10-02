#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
command="${1:-}"
if [[ -z "$command" ]]; then
  echo 'Usage: ruler_corpus_add_ratios.sh {detach|resume|stop|status|status_same_count|report} [--root PATH]' >&2
  exit 2
fi
shift
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/corpus_add_ratios.py" "$command" \
  --root "${EXPERIMENT_DIR:-$CODE_ROOT/outputs/ruler13-120-l20}" "$@"
