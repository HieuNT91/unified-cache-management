# Native attention router on 8×A800

Start here for the router in the clean checkout. This workflow supersedes the
benchmark-based portable deployment adapter. It uses this checkout's chunked
prefill runtime, ProphetKV implementation, connector, and cache lifecycle.
No benchmark Python module or benchmark runtime is imported, copied, or invoked.
The compact cohort manifest and finalized training trees are data inputs.

Training remains separate and continues unchanged. Its runtime differs from the
clean A800 inference runtime. All three frozen policies are evaluated; **router1
is always primary**. There is no A800 timing recalibration, test-based refit, or
selection of a new primary. CPU validation does not establish GPU equivalence;
the scheduled A800 measurements enforce the runtime acceptance gates below.
No local test inference or separate GPU qualification is part of this workflow.

## Deploy code once through Git

On the source machine, commit the clean worktree changes to the intended branch
and push that branch to your normal Git remote. This implementation does not push
or contact a server automatically. On A800, fetch the resulting revision and
create a separate worktree, keeping any existing runs and their code unchanged:

```bash
cd /mnt/sde/jh/projects/unified-cache-management
git fetch origin
git worktree add /mnt/sde/jh/projects/router13-native <published-clean-revision>
cd /mnt/sde/jh/projects/router13-native
```

Use the clean checkout's environment installation in [README.md](../README.md):
Python 3.10, vLLM 0.9.2, torch 2.7.0, transformers 4.53.2, uc-manager 0.3.0.
The launcher applies this checkout's patches in drivers and spawned workers.
Do not substitute the training benchmark's patched runtime. Model files must
already exist locally and match the frozen Qwen3-32B revision's metadata/tokenizer
hashes. Preparation additionally requires the CPU versions recorded in
[router_cohort.json](router_cohort.json): NumPy 2.2.6, SciPy 1.15.3, NLTK 3.10.3,
wonderwords 3.0.1, matplotlib 3.10.6, and PyYAML.

## Export the real policies after training finishes

The clean exporter is a **one-shot CPU command**, not another detached watcher.
Leave the existing training supervisor, reporter, and legacy export consumer
alone. The command refuses incomplete training, live owned engines, altered
locked artifacts, incompatible schemas, or non-equivalent tree decisions.
It reads finalized policies and saved training features as data; it never imports
training code or fits trees. Run it only after training has finalized:

```bash
cd /home/thnguyen/spark/unified-cache-management/.worktrees/prophetkv-clean
CUDA_VISIBLE_DEVICES='' /home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python \
  -m runner.router_export \
  --training-root /home/thnguyen/spark/unified-cache-management/.results/adaptive-router32b-ruler13-yarn4-all64-train15-test100-20260928 \
  --output /absolute/export/path/router13-native-trees.json
```

Outputs: one versioned JSON containing three exact feature-threshold/action
trees, full-precision thresholds, frozen costs and provenance; `.json.sha256`;
readable `.txt` if/else rules; and a local verification receipt. Transfer only the
JSON and its checksum to the server. The old `router13-trees.json` adapter format
is intentionally a different schema; use this exporter for the native checkout.
No genuine native tree file is manufactured before training completes.

## Local server settings

Create a **new** `.env` in the clean server worktree. The same trusted Bash loader
and precedence rules as the existing launchers apply: exported environment values
win, then `.env`, then defaults. `UCM_ENV_FILE` can select a different file. Paths
in later assignments expand using the values already selected. Quote paths with
spaces. Avoid reusing the LongBench/RULER result, prepared, or cache directories.

```bash
PROJECT=/mnt/sde/jh/projects/router13-native
PYTHON_BIN=/mnt/sde/jh/envs/ucm/bin/python
MODEL_PATH=/mnt/sde/jh/ckpts/Qwen3-32B
EXPERIMENT_DIR="$PROJECT/outputs/router13-native-a800"
PREPARED_DIR="$PROJECT/inputs/router13-frozen"
CACHE_ROOT="$PROJECT/.cache/router13-native"
ROUTER_POLICY_FILE="$PROJECT/router13-native-trees.json"
RULER_ROOT="$PROJECT/.cache/vendor/RULER"
PREPARE_WORKERS=8
# Optional: replace defaults with two disjoint groups of four full GPU UUIDs.
# GPU_A=GPU-...,GPU-...,GPU-...,GPU-...
# GPU_B=GPU-...,GPU-...,GPU-...,GPU-...
```

The default UUID groups are the existing A800 guide's physical 0–3 and 4–7 groups.
`detach` verifies all eight UUIDs resolve to idle A800 devices. Numeric device
indices are not accepted. Existing processes are never stopped to acquire GPUs.

## Configure → prepare → verify → detach → status

Official source assets must be available at `RULER_ROOT`. The existing clean
CPU downloader can obtain them; its pinned NVIDIA revision and every required
source/asset checksum must match `router_cohort.json`:

```bash
CUDA_VISIBLE_DEVICES='' /mnt/sde/jh/envs/ucm/bin/python scripts/ruler_64000.py download \
  --ruler "$PWD/.cache/vendor/RULER"
bash scripts/a800_router.sh configure
bash scripts/a800_router.sh prepare
# After transferring the finalized native JSON and its .sha256:
bash scripts/a800_router.sh verify
bash scripts/a800_router.sh detach
bash scripts/a800_router.sh status
```

`configure` may precede policy availability. `prepare` is CPU-only: it regenerates
all 1,300 prompts from official generators with frozen seeds/source identities.
It requires exact historical input, metadata, token, question-span and QA identity
hashes, not just matching lengths. All 13 tasks have 100 rows; FWE alpha is 2.0.
No source truncation or replacement prompt is allowed. Source download drift
(e.g. an essay changing) halts preparation; use an independently verified copy
of the pinned official asset if needed. `verify` freezes code, model, inputs,
policy and group settings before a GPU process can be launched.

Preparation writes a normal clean `manifest.jsonl` plus a richer `cohort-index.json`.
Both use relative prepared-input paths. Git checkout paths and preparation roots
may differ from the training machine. Once configured, keep a run's paths and
hashes unchanged; resume rejects drift rather than modifying accepted evidence.

The supervisor is CPU-only, in a separate session, with ignored SIGHUP and
`/dev/null` stdin. It owns two separate TP4 worker process groups, duplicate locks,
a 900-second accepted-progress watchdog and one operational retry. Validation
failures stop acceptance and are not retried automatically. Resume requires no
live owned supervisor/engine and validates all previously accepted records:

```bash
bash scripts/a800_router.sh resume
bash scripts/a800_router.sh status
# Recover final CPU reporting after investigating a reporting failure:
bash scripts/a800_router.sh report
```

Failed/unvalidated attempts are preserved separately; only missing accepted
records are scheduled. A completed scope cannot restart. Status is read-only;
there is no ongoing assistant polling or SSH launch.

## Inference and acceptance

Each task's even/odd ordinal selects its TP4 group (50 rows/task/group). Method
order is baseline → all64/1% → all64/20% → all64/50% → router1 → router2 → router3.
Each group retains one engine per configuration, processes prompts sequentially,
builds one prompt's independent chunk cache, verifies all committed shards, then
retires and deletes that prompt's KV before proceeding. Each configuration may
rebuild those independent chunks; the router probe and answer share the same
immutable cache. Construction/readiness/ordinary priming are reported separately.

The profile is Qwen3-32B BF16 TP4 eager, native YaRN4 original32768/table131072,
independent KV65920/1031 blocks, chunks and activation tiles4096, chunked prefill
at most16384 scheduled tokens, input≤64000, whole-question fresh suffix≥256,
greedy non-thinking256. Scheduled engine initialization verifies every layer's
native YaRN formula and the KV allocation. Eight warmup TP shards must commit.

Each of the 3,900 router answers runs its own all64/1% probe and discards its single
internal token. The five features are top1/20/50 eligible mass, layer0–31/32–63
cosine agreement, and group top20 Jaccard. Native ascending FP32 accumulation and
TP averaging remain in inference. Feature mass/agreements use the frozen float64
postprocessing and stable ascending-position ties. Per-rank native FP32 layer
replay, identical TP scores/masks and all64 selected sets are acceptance gates.
The probe retires before a tree decision is synchronized to all ranks and the
final answer starts from the unchanged chunk cache. Missing features select
native dense with a reason; corrupt policies, arrays or rank disagreement halt.

Sparse answers use clean ProphetKV. Dense fallback uses native attention across
all64 layers, zero reused KV, and request-scoped UCM lookup/load/store bypass.
The standalone baseline has no connector. Router scores/masks and output token
IDs must equal their selected fixed action on the same group; dense must equal
baseline. Total TTFT starts before the routing probe and ends at the first final
answer token, including export, features, retirement, decision and TP synchronization.

After 9,100 validated answers, 3,900 independent probes and owned-engine exit,
`final/` contains TXT/HTML/CSV/JSON results, raw predictions/output tokens,
paired uncertainty, timing decomposition, routing/fallback counts, output caps,
PNG/PDF charts and three tree diagrams. Each frozen policy is assessed against
**macro loss≤2 percentage points AND mean total-TTFT speedup≥4×**. Claims remain
limited to this cohort. Full final validation hashes are in `final-validation.json`.

## Direct clean method interfaces and CPU checks

```bash
CUDA_VISIBLE_DEVICES='GPU-...,GPU-...,GPU-...,GPU-...' bash run.sh run \
  --method router --router-policy /path/router13-native-trees.json --router-id router1 \
  --model /path/Qwen3-32B --input /path/prepared.json --output /path/new-result
# Or use an existing clean TP4 collection prepared with the same chunk profile:
CUDA_VISIBLE_DEVICES='GPU-...,GPU-...,GPU-...,GPU-...' bash run.sh run \
  --method router --router-policy /path/router13-native-trees.json --router-id router1 \
  --setup /path/collection --max-output-tokens 256 --output /path/new-result
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

Manual `--ratio`, `--layers`, and `--num-layers` overrides are rejected for router.
`--dry-run` verifies the policy and sample profile and prints configuration without
GPU loading. Direct interfaces use the same probe/retirement/answer mechanism;
full paired 9,100-answer acceptance and reporting belong to the A800 launcher.

For metrics over the same accepted prompts across methods, use
`status_same_count`; see [matched status](STATUS_SAME_COUNT.md).
