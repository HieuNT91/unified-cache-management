#!/usr/bin/env bash
# The tree + masked-recompute method (see TREE_RMASK.md) on a prepared RULER 64K manifest, one TP1 shard per GPU.
#   PYTHON_BIN MODEL MANIFEST OUT CACHE_ROOT GPUS="0 1" [PERCENTAGES="15 20"] [SHARD_OFFSET SHARDS_TOTAL]
# Runs the dense baseline first, then the method at each recompute percentage (KV built once per prompt).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UCM_TREE=1 UCM_S4=1 UCM_PROBE_S4=1 UCM_MASK_QUESTION=1 UCM_RMASK=1
PERCENTAGES="${PERCENTAGES:-15 20}" EXTRA_ARGS="--probe-query suffix ${EXTRA_ARGS:-}" \
  exec bash "$ROOT/scripts/ruler64k_launch.sh"
