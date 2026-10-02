#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$CODE_ROOT/scripts/launcher/server_env.sh"
ucm_load_server_env "$CODE_ROOT"
export PYTHON_BIN="${PYTHON_BIN:-/mnt/sde/jh/envs/ucm/bin/python}"
export MODEL_PATH="${MODEL_PATH:-/mnt/sde/jh/ckpts/Qwen3-32B}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$CODE_ROOT/outputs/router13-native-a800}"
export PREPARED_DIR="${PREPARED_DIR:-$CODE_ROOT/inputs/router13-frozen}"
export CACHE_ROOT="${CACHE_ROOT:-$CODE_ROOT/.cache/router13-native}"
export ROUTER_POLICY_FILE="${ROUTER_POLICY_FILE:-$CODE_ROOT/router13-native-trees.json}"
export RULER_ROOT="${RULER_ROOT:-$CODE_ROOT/.cache/vendor/RULER}"
GPU_A="${GPU_A:-GPU-6f2a33e5-aa6e-6681-3d49-1de956c38b3e,GPU-87070a52-3745-1eae-7b16-69594603f277,GPU-73ecc591-94c9-c261-412c-5e0cf51fa103,GPU-5224ff6c-63bb-3a1e-249c-15de01751b5d}"
GPU_B="${GPU_B:-GPU-1d441d83-c33c-4797-23be-b7355d92e3b6,GPU-83ea256b-cf28-6476-ce48-f34e04955379,GPU-b6ece736-772f-8ff9-e3a6-f4a0b369e7b4,GPU-9eed921f-1ef7-c923-f4ec-acf7181237a9}"
command="${1:-}"
case "$command" in
    configure|prepare|detach|resume|stop|status|status_same_count|report) ;;
    *) echo 'Usage: a800_router.sh {configure|prepare|detach|resume|stop|status|status_same_count|report}' >&2; exit 2 ;;
esac
export PYTHONPATH="$CODE_ROOT" PYTHONDONTWRITEBYTECODE=1
exec env CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$CODE_ROOT/scripts/router_control.py" "$command" \
    --root "$EXPERIMENT_DIR" --model "$MODEL_PATH" --prepared "$PREPARED_DIR" \
    --policy "$ROUTER_POLICY_FILE" --cache-root "$CACHE_ROOT" --ruler "$RULER_ROOT" \
    --gpu-a "$GPU_A" --gpu-b "$GPU_B" --workers "${PREPARE_WORKERS:-8}"
