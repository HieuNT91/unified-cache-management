#!/usr/bin/env bash
# Launch one TP1 `run.sh sweep` shard per GPU over a prepared RULER manifest:
# baseline, then all-layer ProphetKV at each ratio (KV built once per prompt).
#   PYTHON_BIN MODEL MANIFEST OUT CACHE_ROOT GPUS="1 2 3" [PERCENTAGES="10 30 50 70 90"] [EXTRA_ARGS="--probe-query suffix"]
#   [SHARD_OFFSET=0 SHARDS_TOTAL=<len GPUS>] to split one manifest across machines
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PYTHON_BIN:?}" "${MODEL:?}" "${MANIFEST:?}" "${OUT:?}" "${CACHE_ROOT:?}" "${GPUS:?}"
PERCENTAGES="${PERCENTAGES:-10 30 50 70 90}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Multi-machine: this machine runs shards SHARD_OFFSET.. of SHARDS_TOTAL (default: all shards here).
SHARD_OFFSET="${SHARD_OFFSET:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
read -r -a gpus <<< "$GPUS"
[[ ${#gpus[@]} -gt 0 ]] || { echo "GPUS is empty" >&2; exit 1; }
for g in "${gpus[@]}"; do
  busy=$(nvidia-smi --id="$g" --query-compute-apps=pid --format=csv,noheader 2>&1 || echo unreadable)
  [[ -z "$busy" ]] || { echo "GPU $g is busy or unreadable: $busy" >&2; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "$OUT exists; use a new root" >&2; exit 1; }
mkdir -p "$OUT/_logs" "$CACHE_ROOT"
cp "$MANIFEST" "$OUT/manifest.jsonl"
git -C "$ROOT" rev-parse HEAD > "$OUT/_logs/git_sha.txt"
git -C "$ROOT" status --porcelain > "$OUT/_logs/git_status.txt"
"$PYTHON_BIN" -c "import importlib.metadata as m;print({p:m.version(p) for p in ('vllm','torch','transformers','uc-manager')})" \
  > "$OUT/_logs/versions.txt"
shards="${SHARDS_TOTAL:-${#gpus[@]}}"
for i in "${!gpus[@]}"; do
  uuid=$(nvidia-smi --id="${gpus[$i]}" --query-gpu=uuid --format=csv,noheader)
  log="$OUT/_logs/shard$((SHARD_OFFSET + i))_gpu${gpus[$i]}.log"
  CUDA_VISIBLE_DEVICES="$uuid" PYTHON_BIN="$PYTHON_BIN" setsid nohup bash "$ROOT/run.sh" sweep \
    --model "$MODEL" --manifest "$MANIFEST" --output "$OUT" --cache-root "$CACHE_ROOT" \
    --tp 1 --shard "$((SHARD_OFFSET + i))" --shards "$shards" --percentages $PERCENTAGES --no-selective \
    --context-length 73728 $EXTRA_ARGS > "$log" 2>&1 < /dev/null &
  echo "shard $((SHARD_OFFSET + i)) -> GPU ${gpus[$i]} ($uuid) pid $! log $log"
done
