# L20 collection v3

Use the [v3 protocol](COLLECTION_V3.md) for this fresh `ruler-l20-tp2-500-v3` run. Physical GPUs:
0–9, paired 0/1, 2/3, 4/5, 6/7, 8/9. All devices are explicit UUIDs below. These are stored assignments; no
remote hardware status was checked during implementation.

After reviewing and pushing the local change, run these commands **on L20**.
Choose the tested commit from the fetched branch; keep all old checkouts/runs intact.

```bash
PROJECT_DIR=/data/jh/unified-cache-management
cd "$PROJECT_DIR"
git fetch origin prophetkv/tp2-router-data
COLLECTION_COMMIT=$(git rev-parse origin/prophetkv/tp2-router-data)
COLLECTION_CHECKOUT="$PROJECT_DIR/.worktrees/ruler-l20-tp2-500-v3"
git worktree add --detach "$COLLECTION_CHECKOUT" "$COLLECTION_COMMIT"
cd "$COLLECTION_CHECKOUT"
cp .env.l20.example .env.l20
```

The copied environment contains:

```bash
# Fresh collection v3; copy to .env.l20. UUIDs are saved assignments, not a live availability check.
# BF16 / TP2; automatic KV memory budget 95%.
PROJECT_DIR="/data/jh/unified-cache-management"
PYTHON_BIN="/data/jh/envs/ucm/bin/python"
MODEL_PATH="/data/jh/ckpts/Qwen3-32B"
RULER_PATH="$PROJECT_DIR/.cache/vendor/RULER"
EXPERIMENT_DIR="$PROJECT_DIR/.results/ruler-l20-tp2-500-v3"
PREPARED_DIR="$PROJECT_DIR/.data/prepared/ruler-l20-tp2-500-v3"
CACHE_ROOT="$PROJECT_DIR/.cache/ruler-l20-tp2-500-v3"
TP="2"
GPU_DEVICES="GPU-9af1e932-db0c-d64a-e1d4-e86854f6168b,GPU-9487c377-cb87-c9cf-bdfd-7e9e0c5e93cf,GPU-568bf4a9-91fe-67c9-c454-c3f39345b17b,GPU-63808146-028b-5fb1-2e0c-0f4361156677,GPU-21292522-6b1d-d6fd-517d-4f5ad92c228f,GPU-c7cf4603-1994-5a84-cba8-a9005bc62dc9,GPU-1ef3727d-cbd0-0c94-eb49-7d58c12344d9,GPU-34e0b17f-40d7-6140-6f41-b92ab321b9f3,GPU-b8608e28-e8e5-0b2f-be5a-05bcd629d768,GPU-b155ae3b-2172-c9ff-8810-bed32cb1164d"
SAMPLES_PER_TASK="500"
SEED="42"
```

Prepare then launch:

```bash
bash scripts/launcher/l20_ruler_data.sh configure
bash scripts/launcher/l20_ruler_data.sh prepare
bash scripts/launcher/l20_ruler_data.sh detach
```

Inspect or export while running:

```bash
bash scripts/launcher/l20_ruler_data.sh status
bash scripts/launcher/l20_ruler_data.sh status_same_count
bash scripts/launcher/l20_ruler_data.sh compute feature
# Equivalent CPU-only alias:
bash scripts/launcher/l20_ruler_data.sh compute-feature
```

After a failed/stopped attempt, inspect its saved error and resume from the same
checkout and unchanged environment:

```bash
bash scripts/launcher/l20_ruler_data.sh resume
```

A completed run cannot be restarted. Resume preserves accepted seed receipts and
only schedules missing work. The exporter writes `ruler-features.npz` at `EXPERIMENT_DIR` once
at least one sequence has all methods/seeds and a probe. Full reports are
`status.md` / `status_same_count.md` with matching JSON/CSV. No separate verify
stage, replay of all diagnostics, or automatic training is performed.
