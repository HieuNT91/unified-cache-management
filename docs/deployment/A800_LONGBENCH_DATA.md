# TP2 collection on A800/L20 and CPU router training

This workflow supersedes the nine-action TP4 deployment in `d0483b1` for **new**
collection. Use a separate checkout and fresh prepared/result/cache directories.
Existing experiments, frozen code and accepted results stay unchanged. Only CPU
implementation/tests have been performed; no experiments, real-data training,
push, SSH or remote launch were performed for this change.

## Scope and allocation

All workers use Qwen3-32B BF16 TP2, original all64 ProphetKV and native YaRN4.
The twelve actions are no-cache and 1/5/10/20/30/40/50/60/70/80/90%.

| Server / launcher | Physical GPU pairs | Actions | Prompt assignment |
|---|---|---|---|
| A800 primary | 0–1, 2–3 | no-cache, 1/5/20/40/70/90% | even/odd source ordinal, 252/251 |
| A800 extra | 4–5, 6–7 | 10/30/50/60/80% | same 503 inputs, same even/odd assignment |
| L20 | 0–1, 2–3, 4–5, 6–7, 8–9 | all twelve | sample index modulo 5 within every task |

LongBench v2 keeps all 503 original rows: **6,036 answers + 503 feature probes**.
Primary contributes 3,521 answers and extra 2,515. Thinking, 16,384 output tokens,
114,688 input limit and official middle truncation remain as documented in the
[LongBench protocol](../protocols/longbench-v2.md).

RULER uses all 13 tasks, **100 samples/task, seed42**, generated once as a batch
per task before sharding. Each L20 pair gets 20/task, 260 prompts and all twelve
actions per prompt: **15,600 answers + 1,300 feature probes**. The 65,536-token
budget includes the output reserve; input is not padded/truncated to 64,000.
Non-thinking caps remain NIAH128 / VT30 / CWE120 / FWE50 / QA32. See the
[RULER protocol](../protocols/ruler.md). No source text, precision or answer cap
is changed to make a GPU fit.

Baseline engines run first, then cached engines reuse one temporary prompt KV
across that launcher's missing actions. Each pair has its own worker/session,
cache, lock and progress watchdog. Later feature engines start only after every
control for that dataset is committed and all control engines have exited.
A800 feature capture uses both primary pairs; RULER uses all five L20 pairs.
No training is triggered by collection completion.

## Two configuration files

Copy [`.env.a800.example`](../../.env.a800.example) to `.env.a800` on A800, and
[`.env.l20.example`](../../.env.l20.example) to `.env.l20` on L20; edit the paths.
These are the only server configuration files needed by this workflow. Launchers
load them automatically from the checkout root, including when invoked elsewhere.
No terminal `export` is needed. The trusted Bash assignment loader preserves
existing environment values; `UCM_ENV_FILE` can explicitly override the file.

A800 settings:

```bash
PYTHON_BIN=/path/to/environment/bin/python
MODEL_PATH=/path/to/Qwen3-32B
LONGBENCH_DATA=/path/to/LongBench-v2/data.json
EXPERIMENT_DIR=/path/to/new/longbench-tp2-results
PREPARED_DIR=/path/to/new/longbench-tp2-inputs
CACHE_ROOT=/path/to/new/longbench-tp2-cache
TP=2
GPU_PRIMARY=0,1,2,3
GPU_EXTRA=4,5,6,7
DATA_IMPORT_DIR=/path/to/router-imports
RULER_DATA_FILE=$DATA_IMPORT_DIR/ruler-data.json
LONGBENCH_DATA_FILE=$EXPERIMENT_DIR/longbench-data.json
TRAINING_OUTPUT_DIR=/path/to/router-training
```

L20 settings:

```bash
PYTHON_BIN=/data/jh/envs/ucm/bin/python
MODEL_PATH=/data/jh/ckpts/Qwen3-32B
RULER_PATH=/path/to/pinned/RULER
EXPERIMENT_DIR=/path/to/new/ruler-tp2-results
PREPARED_DIR=/path/to/new/ruler-tp2-inputs
CACHE_ROOT=/path/to/new/ruler-tp2-cache
TP=2
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9
SAMPLES_PER_TASK=100
SEED=42
```

Use the existing compatible environment (Python3.10, vLLM0.9.2, PyTorch2.7,
transformers4.53.2 and this repository's UCM runtime), local model and pinned
RULER sources/assets or LongBench data. Preparation runs on CPU. See
[`ruler.py`](../../scripts/ruler.py) for the existing official download command.

Configure resolves only that launcher's GPU indices to full UUIDs and freezes
its assignment. **Primary does not require extra GPUs to exist or be idle.**
Later extra configuration writes its own devices/protocol files; it leaves
primary settings, plan, protocols, prepared inputs and committed answers intact.
Primary and extra must use disjoint UUIDs and the same model/data/path settings.

At engine launch, device names, free memory and existing compute processes are
checked. A800 uses utilization0.90 and an8GiB/rank workspace estimate. The explicit
L20 TP2 RULER profile uses utilization0.95 and3GiB/rank workspace. Admission
counts model BF16 bytes/TP plus full original-position KV/TP plus workspace,
divided by utilization. It records the breakdown and refuses insufficient/busy
devices. This estimate is not remote GPU validation; initialization and runtime
checks must also pass. No automatic reduction of precision, prompt or cap occurs.

## Collection commands

Only primary is intended to start initially on A800:

```bash
bash scripts/launcher/a800_longbench_primary.sh configure
bash scripts/launcher/a800_longbench_primary.sh prepare
bash scripts/launcher/a800_longbench_primary.sh detach
bash scripts/launcher/a800_longbench_primary.sh status
bash scripts/launcher/a800_longbench_primary.sh status_same_count
```

When A800 GPUs4–7 become available:

```bash
bash scripts/launcher/a800_longbench_extra.sh configure
bash scripts/launcher/a800_longbench_extra.sh detach
```

Primary finishes its controls, releases its GPU engines, then waits on CPU for
extra. Once extra completes and exits, primary collects the503 probes and exports
the dataset. If extra failed/stopped, primary halts the handoff; resume extra,
then primary. There is no separate qualification/verify phase.

L20:

```bash
bash scripts/launcher/l20_ruler_data.sh configure
bash scripts/launcher/l20_ruler_data.sh prepare
bash scripts/launcher/l20_ruler_data.sh detach
bash scripts/launcher/l20_ruler_data.sh status
bash scripts/launcher/l20_ruler_data.sh status_same_count
```

All three collection launchers accept `stop` or `resume` in place of the final
command. Stop targets only receipt-owned sessions after PID/start-time/command
identity checks; primary stop includes its feature workers. Duplicate launch and
worker locks prevent concurrent owners. Worker failure terminates sibling workers
of that launcher and preserves committed results. Resume runs missing records
only, archives incomplete record folders, and cleans only dead owned caches.
Completed scopes refuse restart. **Do not stop as part of normal completion.**

Keep the configured checkout unchanged through completion/resume: implementation
hashes are pinned. Collection records retain per-request rank/mask/all64 checks,
full cache readiness, immutability, retirement and deletion evidence. TP2 warmup
requires four committed shards (two blocks × two ranks); TP4 remains compatible
with its eight-shard contract. Attention replay uses exact FP32 TP reduction,
not numerical tolerance.

## Reporting and portable files

Status reads committed records; it does not load the model or replay attention.
LongBench prints overall and official short/medium/long strata; RULER prints
overall and each task. Reports include accuracy, answer TTFT, paired baseline
speedup, thinking/answer token counts and output caps.

`status_same_count` intersects **exact prompt IDs across all twelve methods**.
All twelve methods remain visible with zero/N/A where missing. LongBench also
prints a separate seven-method primary matched table, including length strata,
and shows whether extra is configured and how many answers are waiting. No
unfinished method is silently dropped. UUID pairs are distinct timing cohorts;
A800 extra has no new baseline on its own pair.

Reports appear under `EXPERIMENT_DIR` as `status`, `status_same_count` and final
`.md/.csv/.json`; logs, progress and ownership receipts live under `primary/`,
`extra/` or `ruler/`. The full training export is gated on controls and probes:

- A800: `longbench-data.json` and `longbench-data.json.sha256`.
- L20: `ruler-data.json` and `ruler-data.json.sha256`.

Five features are fixed: `coverage5_median`, `head_coverage1_p10`, `coverage5_min`,
`group_agreement`, `top20_mass`. Head coverage uses all64 Q heads/layer,
**32/rank at TP2**, at zero-based layers `[7,15,23,31,39,47,55,63]`. The eligible
region excludes exact prefix and fresh suffix; softmax includes all context keys.
Coverage divides selected eligible mass by total eligible mass. Head p10 uses
NumPy linear quantiles. No task, question, reference or prediction is a feature.
Undefined/zero-mass features are null and select no-cache; malformed data halts.

Only the portable JSON and checksum need transfer from L20 to `DATA_IMPORT_DIR`
on A800. Raw attention remains local. Imports require complete503/1300-row
cohorts, all twelve actions, the exact feature definition, TP2 provenance and
valid payload/file checksums. Old nine-action or layer-mean datasets are rejected.
Exports are immutable and idempotent, including recovery after a missing-checksum
interruption. Allow sufficient local disk for temporary KV and attention archives.

## CPU training and saved-tree evaluation

After the data files are complete and RULER has been copied to A800:

```bash
bash scripts/launcher/router_training.sh train ruler
bash scripts/launcher/router_training.sh train longbench-v2
bash scripts/launcher/router_training.sh train both
```

Each command fits and evaluates on **all**1300/503/1803 prompts respectively.
Every prompt has equal weight, including the combined mode; datasets and tasks
are not reweighted to equal totals. Five deterministic folds, stratified by RULER
task or LongBench length, select settings inside the training data. The216-setting
loss/cost grid uses depths1–3 and fold-local fits/costs. Selection retains the
OOF<=2pp loss and>=4x normalized-speed target; otherwise it ranks accuracy first,
then cost. It prefers distinct OOF action vectors. Three selected settings are
refitted on the entire cohort, with `router1` primary. Fitting, setting selection
and final reported evaluation overlap; this is not held-out evidence.

For each prompt i, training uses `answer_TTFT(i,action)/baseline_TTFT(i)` and
`probe_overhead(i)/baseline_TTFT(i)`. Pooled cost is the equal-prompt mean of those
ratios, so hardware speed cannot dominate through raw seconds. Probe overhead is
included for every decision, including dense fallback. Search fields named
`mean_total_ttft` contain **dimensionless baseline multiples**, as recorded in
`settings.json`; `macro_score`/`macro_loss` are compatibility names for the
prompt-weighted score/loss (`score_weighting=prompt`).

Final evaluation reports measured action seconds and baseline seconds separately
per dataset, plus offline probe-inclusive estimates and normalized costs. The
combined report includes separate RULER/LongBench tables, all RULER tasks and
LongBench short/medium/long strata. These are lookup-based estimates from fixed
controls and later probes, **not measured live router latency**.

Outputs use a fresh directory under `TRAINING_OUTPUT_DIR/<dataset>/train-*`:
three checksummed JSON trees and readable `.txt` rules, per-tree decision CSV and
evaluation JSON, summary JSON/TXT, fold/search/settings/data snapshots and a
completion manifest. Exports enforce depth<=3 and replay training decisions,
exact threshold boundaries and missing-feature fallback against the fitted tree.

Evaluate a supplied tree without refitting:

```bash
bash scripts/launcher/router_training.sh evaluate both --tree /path/to/router1.json
# Optional explicit fresh output directory:
bash scripts/launcher/router_training.sh evaluate ruler --tree /path/to/router1.json \
  --output /path/to/new/evaluation
```

The tree checksum sidecar is required. Offline evaluation may apply a saved tree
to either dataset or their union; overlap also checks the saved input hashes.
Evaluation writes JSON/TXT and decisions
CSV/JSON, records training overlap per prompt/dataset, and never calls fitting.
All train/evaluate processes force empty CUDA visibility and single-thread BLAS.

## Git handoff and CPU tests

The local branch is `prophetkv/tp2-router-data`. Nothing was pushed automatically.
If publishing this reviewed branch, fetch it on each server and create a separate
worktree at the same final commit before configuring. Do not upgrade a configured
collection in place or reuse earlier result paths.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/environment/bin/python -m unittest discover -s tests -v
```

Tests use synthetic data/fake engines: TP2 and TP4 contracts, all64 heads, exact
FP32 reduction/masks, warmup, memory gates, two/five-way assignments, independent
primary and late extra, failure/handoff, owned stop/resume, matched reports,
checksum export/import and three-mode training/evaluation. They do not establish
A800/L20 runtime performance or memory feasibility.
