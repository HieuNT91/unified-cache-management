# A800: all 503 LongBench v2 prompts, 13 configurations

This launcher uses the user-specified server paths and all eight GPU UUIDs.
It starts no historical scheduler. GPU execution has not been validated by the
local preparation task. CPU lifecycle checks exercise the temporary-cache sweep;
GPU equivalence remains an opt-in check.

Configurations: baseline; vanilla ProphetKV 1/5/10/15/20/30%; selective ProphetKV
at the same ratios with **zero-based layers 11,12,13,14,15**. Total: **6539 measured
generations**. All use Qwen3-32B BF16, TP4, YaRN4 (32768 → 131072), thinking enabled,
4096-token KV chunks, 16384 scheduled prefill tokens per step, and a combined
16384-generated-token cap (thinking, answer and control tokens). Sampling is
seed0, temperature0.6, top_p0.95, top_k20, min_p0.

The adapter ports `benchmarks/longbenchv2_qwen3_32b_yarn4/prepare.py`: equal
beginning/end slices of the tokenized rendered user prompt, decode, then rebuild
chat/chunks/padding and recount until formatted input is at most 114688 tokens.
The entire question, choices and answer instructions remain. All 503 rows stay
included. The pinned local preparation truncated 235 rows and kept 268 intact;
remote preparation reports its actual counts and records retained spans.
Accuracy uses the official LongBench v2 `pred.py` extractor on the answer channel:
https://github.com/THUDM/LongBench/blob/main/pred.py

## Get the code with Git and use the existing dataset

The server already has LongBench v2 under
`/mnt/sde/jh/projects/unified-cache-management/.data/LongBench-v2`.
Use its `data.json` directly. No archive transfer or dataset download is needed.

First push the experiment branch from the development machine:

```bash
git -C /home/thnguyen/spark/unified-cache-management/.worktrees/prophetkv-clean \
  push -u origin prophetkv/clean-qwen3-32b-yarn4
```

Then, on the A800 server, fetch the branch and create a separate code worktree:

```bash
export PROJECT=/mnt/sde/jh/projects/unified-cache-management
export CODE_ROOT="$PROJECT/.worktrees/longbench-a800"
git -C "$PROJECT" fetch origin
git -C "$PROJECT" worktree add --detach "$CODE_ROOT" \
  origin/prophetkv/clean-qwen3-32b-yarn4
cd "$CODE_ROOT"
```

Run `worktree add` once. For later code updates, while this experiment is stopped:

```bash
git -C "$PROJECT" fetch origin
git -C "$CODE_ROOT" merge --ff-only origin/prophetkv/clean-qwen3-32b-yarn4
cd "$CODE_ROOT"
```

Configure the existing model, environment and dataset. Keep these exported
variables in the shell used for preparation, launch and status commands:

```bash
export MODEL_PATH=/mnt/sde/jh/ckpts/Qwen3-32B
export PYTHON_BIN=/mnt/sde/jh/envs/ucm/bin/python
export LONGBENCH_DATA="$PROJECT/.data/LongBench-v2/data.json"
export EXPERIMENT_DIR="$PROJECT/outputs/longbench-v2-503-yarn4-thinking16k"
export PREPARED_DIR="$PROJECT/.data/LongBench-v2/prepared-qwen3-yarn4"
export CACHE_ROOT="$PROJECT/.cache/longbench-temporary"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

test -f "$LONGBENCH_DATA"
test -f "$MODEL_PATH/config.json"
test -x "$PYTHON_BIN"
```

Use a **new EXPERIMENT_DIR** for each sweep. The separate PREPARED_DIR can be
reused when its preparation settings and source hashes match. All preparation
uses the local dataset and Qwen3 tokenizer/configuration; it does not download
files or compute KV. The model checkpoint must also be local for GPU execution.

The environment must contain the repository-compatible vLLM0.9.2, torch2.7.0,
transformers4.53.2 and CUDA uc-manager0.3.0 native libraries. This Git branch
supplies process-local patches; it does not install the Python/CUDA environment.

## Disk requirements

The sweep **does not precompute or retain the collection's KV**. Each TP4 group
processes its own prompts as follows:

1. Run its no-cache baselines with a connector-free engine.
2. Start one cached engine. For each prompt, construct independent context chunks
   once, wait for every TP shard to commit, then run all 12 cached configurations.
3. Verify every measured request and transfer retirement, delete that prompt's
   KV, then move to the next prompt. Switch scoring policies only while idle.

CPU preparation still freezes token IDs and truncation once; these small inputs
are shared by every method. Baseline never looks up or writes KV. Measured cached
requests cannot write or reconstruct KV. Original disk KV stays unchanged across
all 12 methods, while alignment and repaired tensors are private GPU state.

Qwen3-32B BF16 KV uses `64 × 8 × 128 × 2 × 2 = 262144 bytes/token`.
With formatted input<=114688 and a fresh suffix>=256, retained context is at most
114432 tokens: **27.94 GiB per group**, or **55.88 GiB across both groups**.
TP4 splits this payload across ranks; it does not multiply it. The disk check
reserves twice this payload plus 16 GiB scratch: at most **127.75 GiB free for KV**.
Results, raw diagnostics and prepared inputs require additional space. Recomputation
ratios do not change this KV ceiling. Inter-prompt deduplication is deliberately
not retained; identical chunks inside a prompt are constructed only once.

```bash
mkdir -p "$EXPERIMENT_DIR" "$CACHE_ROOT"
df -h "$CACHE_ROOT" "$EXPERIMENT_DIR"
# CPU tokenization/truncation only; resumable and idempotent:
bash scripts/a800_longbench.sh prepare
"$PYTHON_BIN" scripts/longbench_v2.py disk --prepared "$PREPARED_DIR" \
  --cache-root "$CACHE_ROOT" --groups 2
# CPU-only configuration check; no engine or KV construction:
CUDA_VISIBLE_DEVICES='' bash run.sh sweep --model "$MODEL_PATH" \
  --manifest "$PREPARED_DIR/manifest.jsonl" --output "$EXPERIMENT_DIR" \
  --cache-root "$CACHE_ROOT" --tp 4 --shard 0 --shards 2 --dry-run
```

`CACHE_ROOT` replaces `SETUP_DIR` for this launcher. Existing persistent setups are
left untouched. The generic `setup` and `run --setup` commands remain available
for separate experiments that explicitly want a persistent collection.

## Detached launch

```bash
nohup setsid bash scripts/a800_longbench.sh run \
  > "$EXPERIMENT_DIR/launcher.log" 2>&1 < /dev/null &
echo "Coordinator PID: $!"
```

After CPU preparation and the disk check, the coordinator launches two groups:

| Group | GPUs | Source rows (zero-based) | Configurations | Generations |
|---|---|---|---|---:|
| A | 0,1,2,3 | even rows: 252 prompts | all 13 | 3276 |
| B | 4,5,6,7 | odd rows: 251 prompts | all 13 | 3263 |

A group loads the model twice in total: once for baseline and once for all cached
measurements. Every prompt stays on the same GPU group across all methods. The
launcher pins the exact supplied UUIDs, not numeric CUDA visibility. Group A uses
GPU-6f2a33e5-aa6e-6681-3d49-1de956c38b3e,
GPU-87070a52-3745-1eae-7b16-69594603f277,
GPU-73ecc591-94c9-c261-412c-5e0cf51fa103,
GPU-5224ff6c-63bb-3a1e-249c-15de01751b5d. Group B uses
GPU-1d441d83-c33c-4797-23be-b7355d92e3b6,
GPU-83ea256b-cf28-6476-ce48-f34e04955379,
GPU-b6ece736-772f-8ff9-e3a6-f4a0b369e7b4,
GPU-9eed921f-1ef7-c923-f4ec-acf7181237a9.

An exclusive coordinator lock rejects duplicate sweeps. The launcher refuses
existing group or per-method result directories, including partial runs; it does
not silently rerun accepted measurements. Use a new EXPERIMENT_DIR for a new sweep.
On an ordinary failure, the owned engine shuts down before its private temporary
cache is removed. A hard kill or failed engine shutdown can leave at most the
current prompt's KV per group; `group-N/plan.json` records the unique owned cache
path. Only remove such leftovers after verifying those engines have exited.
A group failure preserves results and marks live reports failed; the other group
continues. Shared disk/CPU contention affects timings. Historical experiments and
persistent setup directories are never modified.

## Results and progress

```bash
bash scripts/a800_longbench.sh status
# CPU preparation and launcher progress:
tail -n 40 "$EXPERIMENT_DIR/launcher.log"
# Per-group logs:
tail -n 40 "$EXPERIMENT_DIR/group-a.log" "$EXPERIMENT_DIR/group-b.log"
cat "$EXPERIMENT_DIR/prophetkv-10/live_aggregation.md"
# After that configuration completes:
cat "$EXPERIMENT_DIR/prophetkv-10/final_aggregation.md"
```

Each of baseline, prophetkv-{1,5,10,15,20,30}, and
selective-{1,5,10,15,20,30} has live/final aggregation in Markdown, JSON and CSV.
Reports contain per-subtask (`domain / sub_domain`) and overall mean/median TTFT,
accuracy, mean thinking/answer content-token lengths, control-token counts,
scored denominators, output-cap counts and unfinished thinking. All references
are supplied, so completed prompts all contribute to accuracy. TTFT is to the
first generated token, including thinking; KV construction/priming are separate.
Concurrent report updates are serialized. Final reports require all 503
validations and both groups' engines to shut down. Cached final reports appear
after both cached sweeps finish, including deletion of their temporary caches. Per-prompt files
retain predictions, output token IDs and diagnostics. `group-N/construction/`
records each prompt's one-time construction/readiness cost, separately from TTFT;
`group-N/cleanup/` records verified deletion and `*-session.json` engine timings.

## Runtime estimate

No throughput measurements from this A800 server are available yet. The 16384
output setting is a cap, not an expected generation length. Long decoding can
dominate the sweep despite faster cached prefill. A planning estimate is:

`wall seconds ≈ CPU preparation + max(group A total seconds, group B total seconds)`.

Each group includes its own per-prompt KV construction, model loads, warmup,
priming, generation, validation and deletion. There is no collection KV setup phase.

For illustration, approximate group A's 3276 requests using the assumed average
output length/rate below and **60 seconds per request** for measured prefill,
priming, cache reads, validation and retirement combined:

| Mean generated tokens | Assumed decode tokens/s per TP4 engine | Sweep time excluding CPU prep / KV building |
|---:|---:|---:|
| 4096 | 30 | 7.5 days |
| 8192 | 25 | 14.7 days |
| 16384 | 20 | 33.3 days |

These rates and overheads are scenarios, not measured A800 performance or an
upper bound. Allow additional time for CPU preparation and per-prompt KV building.
The retained KV footprint is bounded, but repeated cache reads still make disk
throughput material. A provisional planning range is
**roughly 1–3 weeks if average output is 4–8K tokens**, potentially **5 weeks or
longer if outputs repeatedly reach 16K or decoding/storage is slower**. Re-estimate
from the first representative results using generation time (which already
includes TTFT), plus separate priming/readiness/validation/retirement times. Do not
add TTFT again to generation time. Account for the different method workloads and
small-cohort bias; the first few samples need not represent all 503.

## Optional GPU equivalence check

This is a separate GPU job; the launcher does not run it automatically. Supply
at least two small prepared prompts and explicitly assigned GPU UUIDs. It checks
all 13 configurations against fresh single-input engines, compares output tokens
and exact scores/masks, and verifies the temporary cache is gone after the sweep.
It uses greedy one-token generation while preserving input token IDs.

```bash
"$PYTHON_BIN" tests/gpu_temporary_sweep.py --model "$MODEL_PATH" \
  --inputs /path/to/prepared-a.json /path/to/prepared-b.json \
  --output outputs/temporary-sweep-gpu-check \
  --gpu-uuids GPU-6f2a33e5-aa6e-6681-3d49-1de956c38b3e \
    GPU-87070a52-3745-1eae-7b16-69594603f277 \
    GPU-73ecc591-94c9-c261-412c-5e0cf51fa103 \
    GPU-5224ff6c-63bb-3a1e-249c-15de01751b5d
```
