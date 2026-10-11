# A800 collection v3

Use the [v3 protocol](COLLECTION_V3.md) for this fresh `longbench-v2-a800-tp2-4seed-v3` run. Physical GPUs:
0–3, paired 0/1 and 2/3. All devices are explicit UUIDs below. These are stored assignments; no
remote hardware status was checked during implementation.

After reviewing and pushing the local change, run these commands **on A800**.
Choose the tested commit from the fetched branch; keep all old checkouts/runs intact.

```bash
PROJECT_DIR=/mnt/sde/jh/projects/unified-cache-management
cd "$PROJECT_DIR"
git fetch origin prophetkv/tp2-router-data
COLLECTION_COMMIT=$(git rev-parse origin/prophetkv/tp2-router-data)
COLLECTION_CHECKOUT="$PROJECT_DIR/.worktrees/longbench-v2-a800-tp2-4seed-v3"
git worktree add --detach "$COLLECTION_CHECKOUT" "$COLLECTION_COMMIT"
cd "$COLLECTION_CHECKOUT"
cp .env.a800.example .env.a800
```

The copied environment contains:

```bash
# Fresh collection v3; copy to .env.a800. UUIDs are saved assignments, not a live availability check.
# BF16 / TP2; automatic KV memory budget 95%.
PROJECT_DIR="/mnt/sde/jh/projects/unified-cache-management"
PYTHON_BIN="/mnt/sde/jh/envs/ucm/bin/python"
MODEL_PATH="/mnt/sde/jh/ckpts/Qwen3-32B"
LONGBENCH_DATA="$PROJECT_DIR/.data/LongBench-v2/data.json"
EXPERIMENT_DIR="$PROJECT_DIR/.results/longbench-v2-a800-tp2-4seed-v3"
PREPARED_DIR="$PROJECT_DIR/.data/prepared/longbench-v2-a800-tp2-4seed-v3"
CACHE_ROOT="$PROJECT_DIR/.cache/longbench-v2-a800-tp2-4seed-v3"
TP="2"
GPU_PRIMARY="GPU-6f2a33e5-aa6e-6681-3d49-1de956c38b3e,GPU-87070a52-3745-1eae-7b16-69594603f277,GPU-73ecc591-94c9-c261-412c-5e0cf51fa103,GPU-5224ff6c-63bb-3a1e-249c-15de01751b5d"
ANSWERS_PER_QUESTION="4"
```

Primary runs all nine actions. Do not start `a800_longbench_extra.sh` for this run. Set `ANSWERS_PER_QUESTION` before configure; any later change requires fresh prepared/result/cache paths.

Prepare then launch:

```bash
bash scripts/launcher/a800_longbench_primary.sh configure
bash scripts/launcher/a800_longbench_primary.sh prepare
bash scripts/launcher/a800_longbench_primary.sh detach
```

Inspect or export while running:

```bash
bash scripts/launcher/a800_longbench_primary.sh status
bash scripts/launcher/a800_longbench_primary.sh status_same_count
bash scripts/launcher/a800_longbench_primary.sh compute feature
# Equivalent CPU-only alias:
bash scripts/launcher/a800_longbench_primary.sh compute-feature
```

After a failed/stopped attempt, inspect its saved error and resume from the same
checkout and unchanged environment:

```bash
bash scripts/launcher/a800_longbench_primary.sh resume
```

A completed run cannot be restarted. Resume preserves accepted seed receipts and
only schedules missing work. The exporter writes `longbenchv2.npz` at `EXPERIMENT_DIR` once
at least one sequence has all methods/seeds and a probe. Full reports are
`status.md` / `status_same_count.md` with matching JSON/CSV. No separate verify
stage, replay of all diagnostics, or automatic training is performed.
