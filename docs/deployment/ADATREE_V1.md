# AdaTreeV1 fixed native TreeKV router

`adatree-v1` executes this supplied tree without fitting or threshold changes:

```python
if S60_p90 <= 0.73:
    action = 'tree-prophetkv-5'
elif G5_mean <= 0.22:
    action = 'tree-prophetkv-20'
else:
    action = 'tree-prophetkv-40'
```

The current runtime uses Qwen3-32B, all 64 layers, and the existing TreeKV
post-S4 native probe. It retains that probe's per-layer attention on the GPU;
there is no independent feature request or discarded full-input priming answer.
TP averaging uses FP64 for feature reductions while the native selection scores
retain their existing FP32 computation. Only the compact feature/decision receipt
is transferred to the host during routing. Native TreeKV score/F3 diagnostics
remain available through the existing validation path after generation.

For every layer, normalize attention over eligible context tokens (excluding
the freshly computed prefix and suffix). Stable descending native score order
breaks ties by original position. For a hypothetical mask removing the first
`floor(r * eligible_tokens)` tokens, sum the remaining normalized mass in each
eligible chunk to obtain `q`:

- `G5_mean = mean_layers(sqrt(sum_chunks(q_5**2)))`.
- `S60_p90 = percentile90_layers(sum(q_60)**2 / (num_chunks * sum(q_60**2)))`,
  using linear percentile interpolation.

The 5% and 60% masks are used only for feature arithmetic. In particular, the
router never executes a 60% repair before deciding. After the decision, native
TreeKV selects tokens once using the chosen 5/20/40% budget, including its usual
four forced S4 tokens per reused chunk. Original token IDs, question positions,
all-layer alignment, attention masks, cache immutability and retirement remain
checked. Invalid features or disagreeing TP ranks fail the request.

## Usage

Use an existing prepared JSON with a fresh output/cache directory. The regular
entry point retains its normal engine memory configuration:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUIDS" PYTHON_BIN="$PYTHON" ./run.sh run \
  --model "$MODEL" --input "$PREPARED_JSON" \
  --method adatree-v1 --tp 4 --output "$FRESH_OUTPUT"
```

Manual ratio, layer and policy overrides are rejected. Persistent `--setup`
execution is not currently exposed for this method. Existing baseline,
ProphetKV, fixed TreeKV and learned-router commands remain available.

The local debug command reads an existing preparation receipt and manifest;
it has no preparation/generation-of-inputs path. It checks the original 4096-token
chunk layout and model/tokenizer hashes, uses the full original output budget and
seed, and checks input hashes and mtimes after execution. It uses the existing
`rtx4500ada-tree-validation` TP4 profile: BF16, 95% GPU memory, 4 GiB CPU weight
offload per rank, and the dataset's full KV window. Offload affects latency, so
these timings describe this local debug profile.

```bash
PLATFORM=cuda "$PYTHON" scripts/router/debug_adatree.py \
  --dataset longbench-v2 --model "$MODEL" \
  --prepared "$EXISTING_LONGBENCH_PREPARED" \
  --prompt-id longbench-v2-461 --devices "$GPU_UUIDS" \
  --output "$FRESH_OUTPUT" --cache-root "$FRESH_CACHE"
```

For RULER64K, use `--dataset ruler`, the existing thinking prepared directory,
and an exact manifest ID such as `niah_single_1-000`. Add `--dry-run` for CPU-only
inspection without creating output or touching GPUs. The command requires four
explicit GPU UUIDs and idle assigned devices; it never cancels other jobs itself.

On this host, invoke Python through
`/home/thnguyen/.local/bin/nvidia-driver-run` while kernel driver 580.173.02 is
loaded. That existing wrapper selects matching userspace libraries; the system's
580.196.07 NVML otherwise fails before model initialization. It changes only the
launched process environment.

## Routing receipts

At the user's clarification on 2026-11-05, AdaTree-specific feature/routing
timers, timing-only CUDA synchronization, overhead fractions, threshold flags and
runtime warnings were removed. Current routing receipts contain features, the
selected action and native validation metadata. Overhead concerns are reported
in chat when supported by measurements, not enforced or warned about in code.
The existing request TTFT, generation and native TreeKV metrics remain available.

A debug `complete.json` is published only after native answer validation,
cache/F3 retirement, unchanged-input checks and engine shutdown.

After removing the instrumentation, 25 focused CPU tests passed across feature
math, the integrated native probe, routing validation and prepared-input debug.
The tests check that routing receipts have no timing/warning fields and that
AdaTree adds no timing-only CUDA synchronization. Native action and thinking
policy validation remain covered.

## Validation on 2026-11-05

Before the timing correction, 116 distinct CPU tests passed across the feature, integrated-probe, routing,
debug-input, CPU configuration, TreeKV, sweep, router, chunked-prefill and method
scope suites. Coverage includes all three leaves and exact threshold equality,
NumPy feature parity, TP/native selection checks, hypothetical masks, immutable
input/cache handling, frozen thinking budgets and strict `>1%` warnings.

Two real prepared Qwen3-32B prompts completed on the local TP4 profile above.
The following timings describe that earlier instrumented debug version; its
receipts are preserved and do not imply timing in the current router:

| Dataset / prompt | Input tokens | Selected action | Feature + routing | Complete answer request | Overhead / request | Overhead / TTFT | Answer score |
|---|---:|---|---:|---:|---:|---:|---:|
| LongBench v2 / `longbench-v2-461` | 10,150 | TreeKV_40 | 0.175873 s | 207.146927 s | 0.084903% | 2.629585% | 0 |
| RULER64K thinking / `niah_single_1-000` | 65,378 | TreeKV_20 | 0.048517 s | 119.059972 s | 0.040750% | 0.182802% | 1 |

Both used the existing full 16,384-token output budget and seed 0, produced one
native probe and routing event per rank, and passed native selection/F3 replay,
all-layer alignment, cache immutability, retirement and unchanged prepared-input
hash/mtime checks. Neither exceeded the specified complete-request 1% threshold;
the LongBench fraction of TTFT alone did exceed 1%, as reported separately.
These are individual debug samples, not a dataset accuracy or latency benchmark.

Local receipts are under `outputs/adatree-debug-20261105/longbench-461-v2/` and
`outputs/adatree-debug-20261105/ruler-niah-single1-000-v1/`. Their `complete.json`
files pin the result hashes and confirm shutdown. The first LongBench startup
failed on the host's NVML version mismatch before inference; its failure receipt
remains in `longbench-461-v1/`. Subsequent runs used the existing driver wrapper.

At the user's request, the LongBench input was also run with fixed TreeKV_40,
the same seed, model, devices and full budget, using fresh caches/results.
All 836 generated token IDs and the prediction text matched the routed answer
exactly. Native scores, selected positions and F3 links matched on every TP rank.
Both answers were D while the reference was C; routing introduced no answer
difference on this sample. The comparison receipt is
`outputs/adatree-debug-20261105/longbench-461-fixed40-v1/comparison.json`;
the original routed results and prepared files remained unchanged. All three
successful debug engines exited and released their GPU allocations.

## A800: exact prepared inputs, new thinking policy and sequential no-cache

`scripts/launcher/a800_longbench_adatree_v1.sh` reads
`config/.env.a800.adatree-v1.example` directly. No env-copy, configure,
preparation or tokenization step is needed. Paths and UUIDs follow the chat
“Merge tree rmask method” (`01a16c5b-da6f-74b2-bb23-7ede10e3eb98`), with the
current two-seed and generation-budget requests taking precedence.

1. Run AdaTreeV1 for all 503 prompts at seeds 0 and 1: 1006 answers.
2. Wait for every owned router engine to exit, then run connector-free no-cache
   on the same 503 prompts and seeds: another 1006 answers.
3. Publish the final comparison from exact shared input/seed pairs, using the
   newly generated no-cache records. Old no-cache results remain untouched.

Both phases use the same explicit generation policy:

- At most **8192 thinking content tokens**. Reaching this cap forces the native
  `</think>` token and transition into answer generation in the same request.
- At most **10000 total output tokens**, including thinking, answer, transition
  and control tokens. Earlier thinking closure leaves more room for the answer.
- Exact tail repetition: minimum pattern32, maximum128, minimum4 copies.
  Detection closes thinking early and continues the answer; it never stops an
  answer because that answer contains a repeated pattern.
- Qwen3-32B, BF16, two TP2 pairs, 95% automatic KV sizing, full131072 window.
  There are no separate fixed5/20/40 controls or independent feature probes.

The existing prepared directory is
`/mnt/sde/jh/projects/unified-cache-management/.results/longbench-v2-a800-tp2-tree-prophetkv-2seed-v1/prepared`.
Its bytes, token IDs, question positions and chunk4096 boundaries stay unchanged.
A request-local generation view overrides only the output policy; records retain
both the original `prepared_generation_policy` and the effective8192/10000
policy plus the breaker configuration. Original preparation receipts and rows
keep their original generation provenance.

Fresh result/cache names are
`longbench-v2-a800-tp2-adatree-v1-think8192-total10000-2seed-v2`.
A run configured with the previous budgets cannot silently resume with these
settings. The former1006-answer, reused-baseline workflow remains readable under
its original protocol. The new launcher does not depend on the old baseline files.

After pushing branch `AdaTreeV1`, use a fresh checkout on A800:

```bash
(
set -e
repo=/mnt/sde/jh/projects/unified-cache-management
code="$repo/.worktrees/adatree-v1-a800-budget-v2"
git -C "$repo" fetch origin AdaTreeV1
git -C "$repo" worktree add --detach "$code" origin/AdaTreeV1
cd "$code"
unset UCM_ENV_FILE PROJECT_DIR PYTHON_BIN MODEL_PATH EXPERIMENT_DIR CACHE_ROOT
unset PREPARED_DIR BASELINE_SOURCE GPU_PRIMARY KV_CHUNK_SIZE TP SEED
unset ANSWER_SEEDS ANSWERS_PER_QUESTION
bash scripts/launcher/a800_longbench_adatree_v1.sh start
bash scripts/launcher/a800_longbench_adatree_v1.sh status
)
```

`start` launches the detached supervisor. `resume` schedules only missing answers
and preserves the cached-then-baseline phase order. A failed router phase stops
before baseline startup. Use `report` for method progress and `comparison.md`;
`comparison-final.md` is published after both phases finish. While no-cache is
still pending, the paired comparison reports only completed shared pairs.
Per-request checks and process ownership/retirement checks remain enabled.

For later source updates to this new run, stop its supervisor first, then:

```bash
cd /mnt/sde/jh/projects/unified-cache-management/.worktrees/adatree-v1-a800-budget-v2
git pull --ff-only origin AdaTreeV1
bash scripts/launcher/a800_longbench_adatree_v1.sh resume
```

CPU tests cover the budget boundaries, forced transition, repetition detection,
vLLM parameter transport/scheduler stop, shared generation configuration for
both methods, exact prepared input preservation, phase ordering/failure, absence
of a KV connector in no-cache, resume policy checks and paired record validation.
This launcher has not been executed on A800 in this task. Existing local debug
jobs retain their original generation budgets.
