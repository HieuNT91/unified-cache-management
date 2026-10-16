# L40 collection v3

For `PC::S2D timeout(120000)` or a 300-second `execute_model` RPC timeout during priming, see
[the transfer diagnosis and bounded-load patch](L40_TRANSFER_TIMEOUT.md).
ProphetKV attention remains unchanged. Source hashes are provenance only; the
guide includes pull/resume commands that preserve old no-cache results and inputs.

## Resume ProphetKV only

For the existing prepared run, set this in the checkout's `.env.l40.thinking`:

```bash
PROPHETKV_ONLY="1"
```

`bash scripts/launcher/l40_ruler_thinking_data.sh resume` now launches only the
`cached` phase: missing ProphetKV answers at1/5/10/20/30/40/60/80%, with the saved
seed schedule, TP2 groups and original prepared inputs. Existing accepted answers
are skipped. No baseline or independent feature probe is scheduled. The launcher
reads the env file automatically and forwards `--cached-only` to the detached
supervisor. Stop the previous owned supervisor with the launcher's `stop` command
before resuming in this mode; changing an env file does not change a live process.

This changes scheduling only. Cached answers still perform construction, readiness,
`read:prime`, native sparse scoring and retirement; the prime timeout is not fixed
by skipping independent feature collection.

Completion writes `ruler/cached-complete.json` and updates the normal status report.
It does not claim the full collection is complete or export a dataset with missing
probes. Settings, plan, inputs, old baseline records and partial features are retained.
Set `PROPHETKV_ONLY="0"` to resume the normal full schedule later. Keep it0 for a new
full baseline + features + ProphetKV collection.

## Fresh full collection

Use the [v3 protocol](COLLECTION_V3.md) for this fresh `ruler-l40-tp2-thinking-500-v3` run. Physical GPUs:
0–7, paired 0/1, 2/3, 4/5, 6/7. All devices are explicit UUIDs below. These are stored assignments; no
remote hardware status was checked during implementation.

After reviewing and pushing the local change, run these commands **on L40**.
Choose the tested commit from the fetched branch; keep all old checkouts/runs intact.

```bash
PROJECT_DIR=/home/zhufangzhou/jh/projects/unified-cache-management
cd "$PROJECT_DIR"
git fetch origin prophetkv/tp2-router-data
COLLECTION_COMMIT=$(git rev-parse origin/prophetkv/tp2-router-data)
COLLECTION_CHECKOUT="$PROJECT_DIR/.worktrees/ruler-l40-tp2-thinking-500-v3"
git worktree add --detach "$COLLECTION_CHECKOUT" "$COLLECTION_COMMIT"
cd "$COLLECTION_CHECKOUT"
cp .env.l40.thinking.example .env.l40.thinking
```

The copied environment contains:

```bash
# Fresh collection v3; copy to .env.l40.thinking. UUIDs are saved assignments, not a live availability check.
# BF16 / TP2; automatic KV memory budget 96%.
# Physical pairs 0/1, 2/3, 4/5, 6/7. User checks L40 availability.
PROJECT="/home/zhufangzhou/jh/projects/unified-cache-management"
PYTHON_BIN="/home/zhufangzhou/jh/envs/ucm/bin/python"
CUDA_HOME="/home/zhufangzhou/jh/envs/cuda-12.8"
LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64"
NLTK_DATA="/home/zhufangzhou/jh/nltk_data"
MODEL_PATH="/home/zhufangzhou/jh/ckpts/Qwen3-32B"
RULER_PATH="$PROJECT/.cache/vendor/RULER"
EXPERIMENT_DIR="$PROJECT/outputs/ruler-l40-tp2-thinking-500-v3"
PREPARED_DIR="$PROJECT/inputs/ruler-l40-tp2-thinking-500-v3"
CACHE_ROOT="$PROJECT/.cache/ruler-l40-tp2-thinking-500-v3"
TP="2"
PROPHETKV_ONLY="1"
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="2400"
GPU_DEVICES="GPU-f1523b4e-7652-3c0a-a4ce-00b6a9035eb7,GPU-0ba0b63b-10f2-8b99-022f-53776add2f98,GPU-7e88e081-b436-d6b1-1566-fef586a00bd2,GPU-0a78dc8d-9e48-ee78-6f8c-16ab702264a4,GPU-b05b518f-f9cc-5fec-dece-a622a320171f,GPU-d1b70968-82cf-fa4f-b832-e07ec6141b51,GPU-fff11353-d6c1-e416-e681-a6c345fd8c67,GPU-d570360a-f461-72cb-d8df-d9ad0da25252"
SAMPLES_PER_TASK="500"
SEED="42"
```

Check availability of the eight assigned L40 devices yourself before `configure`/`detach`. The launcher does not scan inventory, free VRAM or busy processes. Runtime identity/capacity/ownership checks still apply.

For a new full collection, change `PROPHETKV_ONLY="0"` in the copied env file,
then prepare and launch:

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh configure
bash scripts/launcher/l40_ruler_thinking_data.sh prepare
bash scripts/launcher/l40_ruler_thinking_data.sh detach
```

Inspect or export while running:

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh status
bash scripts/launcher/l40_ruler_thinking_data.sh status_same_count
bash scripts/launcher/l40_ruler_thinking_data.sh compute feature
# Equivalent CPU-only alias:
bash scripts/launcher/l40_ruler_thinking_data.sh compute-feature
```

After a failed/stopped attempt, inspect its saved error and resume from the same
checkout and unchanged environment:

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh resume
```

A completed run cannot be restarted. Resume preserves accepted seed receipts and
only schedules missing work. The exporter writes `ruler-features.npz` at `EXPERIMENT_DIR` once
at least one sequence has all methods/seeds and a probe. Full reports are
`status.md` / `status_same_count.md` with matching JSON/CSV. No separate verify
stage, replay of all diagnostics, or automatic training is performed.
