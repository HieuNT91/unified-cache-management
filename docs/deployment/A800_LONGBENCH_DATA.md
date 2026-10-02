# A800: LongBench v2 controls and attention data

Use a separate checkout for this data run. Do not update its source while it is
running or before resume: configured source hashes are frozen. This package is
CPU-tested; remote A800 GPU execution has not been verified in this session.

## Scope

- All 503 LongBench v2 prompts, same prepared original tokens across methods.
- Physical GPUs 0,1,2,3: no-cache, ProphetKV 1%, 5%, 20% (2012 answers).
- Physical GPUs 4,5,6,7: ProphetKV 10%, 30%, 40%, 50%, 60% (2515 answers).
- Original all64 ProphetKV, Qwen3-32B BF16 TP4, YaRN4, native thinking,
  output cap 16384, input limit 114688 with the existing official middle
  truncation path. Full question/choices/tail retained. No selective-layer method.
- After both groups complete and engines exit, primary collects 503 fresh
  one-token probes on GPUs 0-3. Probe tokens are discarded, not benchmark answers.
- Fixed answers use native diagnostics with no independent probes. The later
  feature capture does not enter fixed-method TTFT.
- Five features: coverage5_median, head_coverage1_p10, coverage5_min,
  group_agreement, top20_mass. Head layers are [7,15,23,31,39,47,55,63],
  all 64 Q heads across four ranks at each layer. Masks exclude the exact first
  chunk and fresh suffix; coverage divides by each layer/head's eligible mass.
  Softmax still includes every context key. Head p10 uses linear interpolation.
- No automatic training: final portable files are router-data.json and
  router-data.json.sha256. Separate training/evaluation is the next package.

## Publish the local deployment branch

The assistant creates a local branch only; pushing is a separate user command:

```bash
git push origin rpkv-a800-longbench-data
```

On the A800 server, from an existing clone of this repository:

```bash
git fetch origin rpkv-a800-longbench-data
git worktree add --detach ../rpkv-a800-data FETCH_HEAD
cd ../rpkv-a800-data
```

Use the compatible existing environment: Python 3.10, vLLM 0.9.2, PyTorch 2.7.0,
transformers 4.53.2 and the repository's UCM runtime. Model weights and the
LongBench v2 data.json must already be available locally on the server.

## Configure the server

Create `.env.a800` in this new checkout; replace paths with actual server paths:

```bash
cat > .env.a800 <<'ENV'
PYTHON_BIN=/mnt/sde/jh/envs/ucm/bin/python
MODEL_PATH=/mnt/sde/jh/ckpts/Qwen3-32B
LONGBENCH_DATA=/mnt/sde/jh/projects/unified-cache-management/.data/LongBench-v2/data.json
EXPERIMENT_DIR=/mnt/sde/jh/results/longbench-v2-a800-data-v1
PREPARED_DIR=/mnt/sde/jh/data/longbench-v2-a800-data-v1
CACHE_ROOT=/mnt/sde/jh/cache/longbench-v2-a800-data-v1
ENV
export UCM_ENV_FILE="$PWD/.env.a800"
export GPU_A="$(nvidia-smi -i 0,1,2,3 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
export GPU_B="$(nvidia-smi -i 4,5,6,7 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
```

Use fresh result/prepared/cache paths. UUIDs are resolved on this A800 server,
then frozen; GPU ordinals are not passed to workers. Both groups must be A800
with sufficient free memory; occupied GPUs cause launch to fail. No other jobs
are stopped. Keep ample disk space for temporary KV and per-rank feature
archives: the latter can occupy hundreds of GB before compression. The portable
JSON contains features and measured outcomes, not the raw attention arrays.

## Prepare and launch

```bash
bash scripts/launcher/a800_longbench_primary.sh configure
bash scripts/launcher/a800_longbench_primary.sh prepare
bash scripts/launcher/a800_longbench_extra.sh detach
bash scripts/launcher/a800_longbench_primary.sh detach
```

Configure/prepare apply once to the shared experiment. Both launchers detach
using nohup, independent sessions and CPU-only supervisors. Each uses only its
assigned four UUIDs. Primary waits without an engine if extra is still running,
then collects features automatically. A failed/stopped extra group prevents
feature collection. There is no separate verify or qualification run.

## Status, reports, stop and resume

```bash
bash scripts/launcher/a800_longbench_primary.sh status
bash scripts/launcher/a800_longbench_primary.sh status_same_count
# Either launcher shows the combined nine-method report:
bash scripts/launcher/a800_longbench_extra.sh status
bash scripts/launcher/a800_longbench_extra.sh status_same_count
```

Both commands print overall and **short / medium / long** tables, with accepted
count, accuracy, mean TTFT, paired baseline speedup, thinking/answer token counts
and output-cap counts. Labels come from original dataset metadata, not current
truncated token lengths. JSON/CSV include additional denominators and metrics.
`status_same_count` intersects exact prompt IDs across all nine methods; it may
show zero until every method has accepted some of the same prompts.
Timing comparisons span two different TP4 groups and are labelled accordingly.

Reports: EXPERIMENT_DIR/status.{md,csv,json}, status_same_count.{md,csv,json},
and final.{md,csv,json} after all fixed answers complete. Supervisor/worker logs
are under primary/ and extra/; feature records are under features/records/probe/.

```bash
bash scripts/launcher/a800_longbench_primary.sh stop
bash scripts/launcher/a800_longbench_extra.sh stop
# Later, with the same checkout, .env and frozen configuration:
bash scripts/launcher/a800_longbench_extra.sh resume
bash scripts/launcher/a800_longbench_primary.sh resume
```

Stop only the requested workflow; preserve accepted results/cache. Primary stop
also stops its active feature worker. Resume skips committed answers/probes.
Completed scopes refuse restart. Status/export do not load a model or replay
large diagnostic archives. Do not run stop as part of normal successful launch.

## Compatible RULER data on another server

For a NEW RULER corpus, use the existing corpus launcher with the feature profile
and action inventory fixed at configure time (use that server's own .env/UUIDs):

```bash
bash scripts/launcher/ruler_corpus.sh configure --feature-profile coverage-five \
  --actions "$PWD/scripts/router_actions_1_5_10_20_30_40_50_60.json"
bash scripts/launcher/ruler_corpus.sh prepare
bash scripts/launcher/ruler_corpus.sh detach
```

After collection exits successfully, export on that server:

```bash
CUDA_VISIBLE_DEVICES='' /path/to/python scripts/router_dataset.py \
  --dataset ruler --root /path/to/completed-ruler-corpus --output /path/to/ruler-data.json
```

Transfer ruler-data.json plus its .sha256 sidecar, and LongBench router-data.json
plus its .sha256 sidecar, to the training machine. Legacy five-feature/layer-mean
archives cannot reconstruct head_coverage1_p10; do not relabel old data. Existing
corpora and running experiments remain unchanged. Training will support ruler,
longbench-v2 and both, with all available training prompts and explicit reporting
of training overlap.
