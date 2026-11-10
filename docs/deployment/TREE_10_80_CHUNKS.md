# Tree 10% / 80%, one rollout at seed 0

Use the original launchers with six new `.10-80.example` env files. These keep
original model, UUIDs, TP2/BF16, automatic KV 95%, preparation seed 42 and
read-only prepared paths. L20 uses 13 x 500 thinking prompts; A800 uses all
503 thinking prompts. Each chunk has separate results/cache. No launcher,
controller, progress or export format changes are required.

In the updated checkout on either server, define:

```bash
tree_run() (
  set -e
  server="${1:?l20 or a800}"; chunk="${2:?512, 2048 or 4096}"; command="${3:?configure, start, resume, status, report, stop or detach}"
  case "$server" in
    l20) launcher=scripts/launcher/l20_ruler_thinking_tree_prophetkv.sh ;;
    a800) launcher=scripts/launcher/a800_longbench_tree_prophetkv.sh ;;
    *) return 2 ;;
  esac
  case "$chunk" in 512|2048|4096) ;; *) return 2 ;; esac
  cp -n "config/.env.$server.tree-prophetkv.chunk$chunk.10-80.example" \
        "config/.env.$server.tree-prophetkv.chunk$chunk.10-80"
  unset EXPERIMENT_DIR CACHE_ROOT PREPARED_DIR PERCENTAGES KV_CHUNK_SIZE
  unset ANSWER_SEEDS ANSWERS_PER_QUESTION SAMPLES_PER_TASK TP SEED
  unset PROJECT PROJECT_DIR PYTHON_BIN MODEL_PATH GPU_DEVICES GPU_PRIMARY
  export UCM_ENV_FILE="$PWD/config/.env.$server.tree-prophetkv.chunk$chunk.10-80"
  bash "$launcher" "$command"
)
```

Set `server=l20` on L20 or `server=a800` on A800. Begin with:

```bash
tree_run "$server" 512 configure
tree_run "$server" 512 start
tree_run "$server" 512 status
```

`start` detaches exactly as before. After chunk 512 completes and releases its
GPUs, run `configure` then `start` for 2048. After 2048 completes, do the same
for 4096. Do not put all three starts in a loop: they share GPUs.
Use `tree_run "$server" CHUNK resume|status|report|stop` for the selected run
(one actual command per invocation). The original progress scripts still take
the selected experiment root. There is no new queue or supervisor.

Result roots (replace SIZE with 512, 2048 or 4096):

- L20: `/data/jh/unified-cache-management/outputs/ruler-l20-tp2-thinking-tree-prophetkv-chunkSIZE-10-80-seed0-500-v1`
- A800: `/mnt/sde/jh/projects/unified-cache-management/.results/longbench-v2-a800-tp2-tree-prophetkv-chunkSIZE-10-80-seed0-v1`

Export using the same script and the new experiment path:

```bash
# L20
/data/jh/envs/ucm/bin/python scripts/export_features.py EXPERIMENT_DIR
# A800
/mnt/sde/jh/envs/ucm/bin/python scripts/export_features.py EXPERIMENT_DIR
```

Output remains `EXPERIMENT_DIR/tree-prophetkv-outcomes.npz`.
