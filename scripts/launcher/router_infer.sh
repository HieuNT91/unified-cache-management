#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
# No command means configure. Launching always requires an explicit detach/resume.
command=configure
if [[ "${1:-}" != --* && $# -gt 0 ]]; then command="$1"; shift; fi
extra=()
[[ -z "${ROUTER_POLICY_FILE:-}" ]] || extra+=(--tree "$ROUTER_POLICY_FILE")
[[ -z "${CORPUS_ROOT:-}" ]] || extra+=(--corpus "$CORPUS_ROOT")
[[ -z "${LONGBENCH_DATA:-}" ]] || extra+=(--data "$LONGBENCH_DATA")
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
exec env CUDA_VISIBLE_DEVICES='' "${PYTHON_BIN:-python}" "$CODE_ROOT/scripts/corpus_control.py" "$command" --inference \
  --root "${EXPERIMENT_DIR:-$CODE_ROOT/outputs/tree-inference-v1}" \
  --model "${MODEL_PATH:-$CODE_ROOT/models/Qwen3-32B}" \
  --prepared "${PREPARED_DIR:-$CODE_ROOT/inputs/tree-inference-v1}" \
  --cache-root "${CACHE_ROOT:-$CODE_ROOT/.cache/tree-inference-v1}" \
  --ruler "${RULER_ROOT:-$CODE_ROOT/.cache/vendor/RULER}" \
  --gpu-a "${GPU_A:-}" --gpu-b "${GPU_B:-}" "${extra[@]}" "$@"
