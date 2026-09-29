# Reusable RULER corpus: A800 and L20

This workflow is independent of `a800_router.sh` and its frozen 1/20/50% schema.
Use this clean worktree. Do not run historical benchmark launchers or alter
existing training/export jobs. These commands configure future server execution;
GPU acceptance requires a separately authorized launch with explicit UUIDs.

## Deployment and environment

Publish your reviewed clean code through Git and fetch it into a separate server
worktree. Install the pinned environment in the main README. Model weights and
pinned official RULER sources must already be present. Download official sources
with the clean `scripts/ruler_64000.py download --ruler ...` command (see its
`--help`); `scripts/corpus_sources.json` pins every required source/tokenizer file
and preparation package version. A fresh download whose hashes differ is rejected.
Never substitute a different source or tokenizer under the same collection version.

Use a new, Git-ignored `.env` at the worktree root, or select a trusted file with
`UCM_ENV_FILE=/absolute/path/corpus.env`. The existing `server_env.sh` loader is
used: exported environment variables win over file values; later assignments
expand earlier effective values. UUIDs have no defaults in these new launchers.
Use `nvidia-smi --query-gpu=uuid,name,memory.total --format=csv` on the server.

A800 example (replace each UUID placeholder with a full real GPU UUID):

```bash
PYTHON_BIN=/mnt/sde/jh/envs/ucm/bin/python
MODEL_PATH=/mnt/sde/jh/ckpts/Qwen3-32B
EXPERIMENT_DIR=/mnt/sde/jh/results/ruler-corpus-v1
PREPARED_DIR=/mnt/sde/jh/inputs/ruler-corpus-v1
CACHE_ROOT=/mnt/sde/jh/cache/ruler-corpus-v1
RULER_ROOT=/mnt/sde/jh/vendor/RULER
GPU_A=GPU-<uuid0>,GPU-<uuid1>,GPU-<uuid2>,GPU-<uuid3>
GPU_B=GPU-<uuid4>,GPU-<uuid5>,GPU-<uuid6>,GPU-<uuid7>
PREPARE_WORKERS=4
```

L20 example: change `PYTHON_BIN` to `/data/jh/envs/ucm/bin/python`,
`MODEL_PATH` to `/data/jh/ckpts/Qwen3-32B`, and choose new writable result, input,
cache and source paths under `/data/jh`. Supply the L20 server's own UUIDs.
Leave `GPU_B=` for one TP4 group. Two groups must be disjoint. Do not set ordinal
CUDA device numbers. Each prompt's ordinal determines its group for every method.
Verification checks idle devices, A800/L20 identity, model/runtime compatibility,
and per-rank memory for BF16 weights, full-position KV and an 8 GiB workspace
reserve. Actual runtime allocation and initialization remain GPU acceptance gates.

## Prepare, configure, verify, collect

Configure freezes the ordered inventory and deployment settings. Preparation first
freezes **all 2,600 identities** (13 tasks × 200), then atomically publishes each
prepared sample. FWE uses alpha 2.0; QA question identities are globally unique.
Exact input size is 64,000 formatted tokens, using recorded newline padding before
user text. Complete question spans and original tokens are reconstructed; no
source truncation occurs. References are separate manifest/receipt metadata.

```bash
scripts/ruler_corpus.sh configure --limit-per-task 50
scripts/ruler_corpus.sh prepare
scripts/ruler_corpus.sh verify
# Only after a user authorizes this GPU launch with the configured UUIDs:
scripts/ruler_corpus.sh detach
scripts/ruler_corpus.sh status
```

Default `--limit-per-task` is 200. With 50, collect 650 independent probes and
2,600 answers; the full frozen plan remains 2,600 prompts. Extend it when ready:

```bash
# After the prior tranche exits; settings, code and action inventory stay fixed.
scripts/ruler_corpus.sh configure --limit-per-task 200
scripts/ruler_corpus.sh prepare
scripts/ruler_corpus.sh verify
scripts/ruler_corpus.sh resume
```

`resume` schedules only missing accepted records. The supervisor checks PID/start
identities, owned engine exit, duplicate locks and immutable receipts. It has a
900-second accepted-progress watchdog and one retry for operational errors.
Validation failures halt. Failed unaccepted attempts are preserved separately.
No command stops another experiment. Baselines run connector-free for the entire
scheduled tranche before cached collection starts. Each group then has one
persistent cached engine: construct one prompt's independent KV, wait for all
committed shards, prime, collect a fresh all64/1% one-token probe, discard its
internal token, measure sparse actions, retire, and delete KV. A resumed probe
may be reused as recorded evidence; accepted answers are never regenerated.

Qwen3-32B uses BF16 TP4, YaRN4 table 131072, KV allocation 65920/1031 blocks,
chunk and activation tiles 4096, scheduled prefill at most 16384, greedy
non-thinking generation capped at 256, and complete-question suffix at least 256.
Every sparse answer must match independent-probe scores and exact selections;
all-layer/all-rank diagnostics, native accumulation, stable ties, floor budgets,
readiness, immutable cache state and retirement are gates.

## Inventories, disk, reports and relocation

Default actions are `nocache`, `prophetkv-1`, `prophetkv-20`, `prophetkv-40`:
10,400 answers and 2,600 probes. Supply `--actions /path/actions.json` at configure
to use another ordered inventory, for example:

```json
[
  {"id":"nocache","method":"baseline","ratio":null},
  {"id":"prophetkv-10","method":"prophetkv","ratio":0.1}
]
```

Only true baseline and original all64 ProphetKV ratios in (0,1] are supported.
IDs and ratios must be unique. A changed inventory needs a new collection/output
version. Feature top50 mass remains defined even if 50% is not an action.

Retained FP32 layers alone approach **159 GiB uncompressed** for all 2,600 probes
(4 ranks × 64 layers × at most 64,000 positions × 4 bytes). NPZ compression is
lossless; do not assume a compression ratio. Scores, mappings, raw inputs, JSON
all-rank diagnostics, output records, temporary archive writes and training
snapshots require additional space. Provision several hundred GiB for results.
Per-group temporary context KV is at most about 15.6 GiB; allow double payload
for writes plus 16 GiB scratch across groups. KV is deleted after each prompt;
attention archives and measurement receipts remain reusable CPU data.

`status` refreshes `live_summary.md`, `live_summary.csv`, and `live_summary.json`
in `EXPERIMENT_DIR` and prints overall metrics and supervisor identity. Tables
include each action and task: accepted/expected answers, accuracy, answer-engine
TTFT, thinking/answer token counts. Independent probes have a separate counter.
For 120/task the scheduled totals are 6,240 answers and 1,560 probes, although the
frozen full inventory remains 200/task. Reports refresh on status invocation.
`status_same_count` writes `same_count_summary.md/.csv/.json` using the exact prompt
ID intersection across actions; ordinary live rows may cover different prompts.
Status checks accepted result hashes and protocol identity without loading token
files, replaying diagnostics/attention, or invoking model/GPU verification.

Do **not** update the code checkout used by an active corpus run: its code hashes
are frozen. To add live reporting to an older running deployment, publish/fetch
the updated code into a **separate reporting checkout**, then run its CPU reporter
against the existing results (it does not load `.env`; pass paths explicitly):

```bash
CUDA_VISIBLE_DEVICES='' /data/jh/envs/ucm/bin/python \
  /path/to/reporting-checkout/scripts/corpus_status.py \
  --root /data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20
```

Add `--same-count` for matched coverage or `--prepared /relocated/inputs` to
override the saved manifest location. This reader can run while collection
continues; it only writes report files and its own report lock. Saved results,
protocol, cache and active processes remain untouched. Live status never
certifies final completion.

`report` is available after engine
exit. `report.json/.txt/.html` label smaller tranches `partial-complete`; only the
entire inventory plus owned-engine exit produces `final-validation.json`.
Per-method paired accuracy and TTFT summaries cover only accepted data. Each
result retains output tokens, score, group, model initialization and diagnostics.

For relocation, copy the entire corpus and prepared roots preserving relative
paths, then update path fields while idle:

```bash
scripts/ruler_corpus.sh relocate --root /new/results --prepared /new/inputs \
  --model /new/model --cache-root /new/cache
scripts/ruler_corpus.sh verify --root /new/results
```

CPU readers can simply pass `Corpus('/new/results', prepared='/new/inputs')`.
Paths are excluded from corpus identity; inventory, code, model/source hashes and
hardware remain frozen. A different GPU group/hardware requires a new collection.
The separate [training guide](RULER_CORPUS_TRAINING.md) explains snapshot semantics.

For metrics over the same accepted prompts across methods, use
`status_same_count`; see [matched status](STATUS_SAME_COUNT.md).
