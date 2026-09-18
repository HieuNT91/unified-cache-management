# Remote Qwen3-32B / ProphetKV / RULER — four TP=2 jobs

Copy the repository, including `run_scripts/`, to the remote 8-A800 server.
Each job uses **exactly two GPUs**. Four detached jobs can run concurrently,
with separate result directories, caches, locks, logs and resume state.

## Scope and job list

All jobs use Qwen/Qwen3-32B BF16, native chat with `enable_thinking=False`,
greedy generation, **128 maximum new tokens**, and **100 samples per task/length**.
Each sample runs seven methods: **no cache, ProphetKV 5%, 10%, 20%, 30%, 40%, 50%**.

The earlier NIAH multivalue runs at 8K/16K/32K are retained. At 64K the scope
now contains **all 13 RULER tasks**. There are 1,600 distinct prompts and
**11,200 measured requests** (9,100 at 64K and 2,100 at shorter lengths).

| Job script | Physical GPUs | Shorter-length work | RULER 64K tasks | Requests |
|---|---|---|---|---:|
| `job_0.sh` | **0,1** | NIAH multivalue 8K | `niah_single_1`, `niah_single_2`, `niah_single_3` | 2,800 |
| `job_1.sh` | **2,3** | NIAH multivalue 16K | `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3` | 2,800 |
| `job_2.sh` | **4,5** | NIAH multivalue 32K | `niah_multivalue`, `niah_multiquery`, `vt` | 2,800 |
| `job_3.sh` | **6,7** | — | `cwe`, `fwe`, `qa_1`, `qa_2` | 2,800 |

Use `bash run_scripts/list_jobs.sh` for the current list, or
`bash run_scripts/job_0.sh plan` for one job's exact settings and paths.
Equal request counts do not imply equal wall-clock runtimes.

## Configure on the remote server

Edit `run_scripts/config.sh`, or export overrides:

```bash
cd /path/to/unified-cache-management
export PYTHON_BIN=/path/to/prepared/environment/bin/python
export MODEL_PATH=/path/to/Qwen3-32B
export RESULT_ROOT=/path/to/results/qwen3-32b-prophetkv-ruler-tp2
export CACHE_ROOT=/path/to/local-nvme/prophetkv-ruler-tp2
export NUM_SAMPLES=100
# If your complete RULER checkout is elsewhere:
export RULER_ROOT=/path/to/RULER
```

`RESULT_ROOT` and `CACHE_ROOT` above are **parent directories**. The numbered
job scripts automatically append `job-0`, `job-1`, `job-2`, or `job-3` and set
the corresponding physical GPU pair. You do not need to set `GPU_INDICES`.

The new default parent is `.results/qwen3-32b-prophetkv-ruler-tp2/`.
Old TP=8 results remain intact. Previously prepared protocols cannot be resumed
with these changed sources, task scopes or GPU assignments; prepare new job directories.

The runtime remains UCM 0.3.0, patched vLLM 0.9.2 with Qwen3 sparse hooks,
PyTorch 2.7.0, Transformers 4.53.2. RULER dependencies include PyYAML,
wonderwords and NLTK sentence-tokenizer data. These scripts do not install or
upgrade packages. `MODEL_PATH` must be a complete local BF16 checkpoint with its
original configuration and tokenizer. Inference and data generation work offline.

**GPU memory:** TP=2 at 64K is intended for A800 **80GB** devices. The launcher
checks a lower bound for weights plus KV against the configured memory budget;
40GB cards do not have sufficient headroom for this BF16 configuration. Smoke
checks establish whether the actual eager execution and audits fit on your server.

`RULER_ROOT` defaults to `benchmarks/vendor/RULER`. The repository carries
`PaulGrahamEssays.json`, `english_words.json`, `squad.json` (SQuAD dev-v2.0),
and compressed `hotpotqa.json.zip` (HotpotQA dev distractor) under
`scripts/data/synthetic/json/`. On the A800 server, `prepare_jobs.sh`
automatically validates and atomically extracts HotpotQA. No dataset download or
manual unzip command is required:

```bash
bash run_scripts/prepare_jobs.sh
```

The Python environment still needs the RULER package dependencies and NLTK
`punkt`/`punkt_tab` data installed ahead of time. `prepare_ruler_assets.sh`
remains a networked-machine utility for rebuilding missing source assets; do not
run it on the offline A800 server.

Allow at least 25 GiB of local cache disk space **per concurrent job** (100 GiB
for four), plus results and smoke tensors. Temporary KV storage is bounded to one
sample per job and removed only after its engine group exits.

## Prepare and launch concurrently

Prepare all four jobs (CPU work, done sequentially; includes preflight checks):

```bash
bash run_scripts/list_jobs.sh
bash run_scripts/prepare_jobs.sh
```

Then execute all four commands in the same terminal. **No trailing `&` is needed**;
each command detaches and returns while its job continues:

```bash
bash run_scripts/job_0.sh detach   # GPUs 0,1
bash run_scripts/job_1.sh detach   # GPUs 2,3
bash run_scripts/job_2.sh detach   # GPUs 4,5
bash run_scripts/job_3.sh detach   # GPUs 6,7
bash run_scripts/status_jobs.sh
```

Equivalent convenience command: `bash run_scripts/detach_jobs.sh`.
After the launchers return, verify `supervisor_live: true` for each job.
Each supervisor uses `nohup`, an independent session and `/dev/null` stdin,
so jobs survive SSH logout. GPUs are verified with `nvidia-smi -L`, pinned by
UUID, and restricted in every engine and TP rank. Busy GPUs are awaited.
No dummy or reservation workloads run.

Individual operations use the same job prefix:

```bash
bash run_scripts/job_0.sh preflight
bash run_scripts/job_0.sh prepare
bash run_scripts/job_0.sh smoke       # Qualification only
bash run_scripts/job_0.sh run         # Foreground alternative to detach
bash run_scripts/job_2.sh detach --length 65536 --task vt --method prophetkv-20
```

Do not start a second command that runs/prepares/reports the same job while its
supervisor holds the job lock. All methods for a sample use the same GPU pair.
Each task/length is gated on its longest prepared prompt: all seven measured
configurations plus 0%/100% controls and a native exact-prefix reference.
Smoke outputs have a 16-token budget and are excluded from the measurement count.
Audits cover all 64 layers and both ranks, including tensor writes, preserved
cached KV, global selection agreement and original-position causal attention.

This runner still uses **fresh engines per sample/method**, plus one population
engine per sample. Model loading adds substantial wall-clock time and shared-disk
traffic; it is excluded from TTFT. Concurrent jobs can affect one another's
CPU/disk performance, so record that concurrency when interpreting timings.

The older `run_all.sh`, `run_8k.sh`, `run_16k.sh`, `run_32k.sh` and `run_64k.sh`
remain sequential alternatives on one GPU pair and the unsuffixed result root.
`run_64k.sh` now covers all 13 tasks. For the requested concurrent layout,
use the numbered job scripts exclusively.

## Monitor, stop and resume

```bash
bash run_scripts/status_jobs.sh
bash run_scripts/job_2.sh status
bash run_scripts/job_2.sh logs       # Current/last engine log; Ctrl-C exits viewer
source run_scripts/config.sh
tail -n 80 -F "$RESULT_ROOT/job-2/supervisor.log"
nvidia-smi

bash run_scripts/job_2.sh stop       # Stop only this verified supervisor/engines
bash run_scripts/job_2.sh status     # Confirm it exited
bash run_scripts/job_2.sh detach     # Resume missing work
# Or signal all four owned supervisors:
bash run_scripts/stop_jobs.sh
```

`logs` follows the engine active when invoked; re-run it when the method changes.
A stopped or failed job preserves accepted records. The same detach command
resumes missing work and archives unaccepted attempts. Configuration/source
changes require new output directories; incompatible records are not mixed.
The watchdog stops owned engines on fatal logs or 1,800 seconds without logged
progress (`WATCHDOG_SECONDS` is configurable). GPU allocation alone is not progress.

| File/message under each `job-N/` | Meaning |
|---|---|
| `progress.json` | Accepted count and per-task/per-length/per-method progress |
| `supervisor.json` | Supervisor identity; `status` verifies whether it is live |
| `active.json` | Sample, method, engine group and current log path |
| `supervisor.log` | Scheduling, gate completion and failures |
| `logs/prepare-<length>-<task>.log` | Dataset generation |
| `cache-builds/*.log` | Offline chunk population, outside TTFT |
| `MEASURE_BEGIN` / `MEASURE_END` | Boundaries of the timed request |
| `result ... score=... ttft=... total=...` | Prediction score and request timings |
| `WORKER_COMPLETE` | Normal worker shutdown; validation follows |
| `*.validated.json` | Accepted record/log/diagnostic hashes |
| `SMOKE_GATE_PASSED <length>/<task>` | Qualification passed for that task/length |
| `state: failed` / `Traceback` | Inspect the indicated engine log |

## Read the results

Each job produces its own final artifacts after all **2,800** requests validate:

- `$RESULT_ROOT/job-N/final/comparison.csv`: **main table**, one row per task,
  context length and method; 28 rows per completed job.
- `final/REPORT.md`: scope and metric definitions.
- `final/raw_records.jsonl`: predictions, output token IDs, references, prompt
  hashes, timings and GPU provenance.
- `final/cache_build_costs.json`: offline population costs and resumed attempts.
- `final/validation.json`: that job's completion certificate.
- `records/<task>-<length>-<row>/<method>.{json,log,diagnostics.json}`:
  individual measurements, engine logs and rank diagnostics.
- `smoke/<length>/<task>/validation.json`: qualification receipt.
- `protocol.json`, `prompt_manifest.json`: pinned configuration and prompt identities.
- `cleanup.json`: owned engine exit and cache removal receipt.

There are four completion certificates; completion of one job does not imply
completion of all 11,200 requests. For an incomplete stopped job, run
`bash run_scripts/job_N.sh report` (substitute 0–3) to create `partial/` instead.

`accuracy_percent` uses local official RULER scoring: QA accepts any reference;
other tasks score the fraction of reference strings found. `mean_ttft_seconds`
measures submission to first token-bearing output, including online ProphetKV
probing, transfers and selection. `ttft_speedup` is paired mean no-cache TTFT /
paired mean method TTFT (>1 is faster); `paired_samples` records its denominator.
`mean_generation_seconds` is separate, and `length_limited` counts outputs that
reach the 128-token cap.

Dataset generation uses the supplied RULER task parameters and reserves 128 output
tokens for every task, matching the requested generation cap. All methods share
the same frozen non-thinking chat and numbered, padded 4,096-token chunks.
"64K" is a dataset target; actual prompt lengths are recorded. ProphetKV ratios
select eligible cached tokens after the exact first chunk and before the fresh
query suffix (at least 256 tokens), not fractions of the entire prompt.
64K uses the [Qwen3-32B documented YaRN configuration](https://huggingface.co/Qwen/Qwen3-32B#processing-long-texts)
with factor 4; shorter lengths retain original RoPE. This is the repository's
ProphetKV UCM/vLLM port. Cache results use buffered local warm storage. Loading,
population, readiness waits and warmup are excluded from TTFT and reported separately.

Run `bash run_scripts/test_cpu.sh` for CPU tests and shell syntax checks. Remote
TP=2 GPU inference has not been executed by this script-editing task; mandatory
smoke gates run on your server before measurements.
