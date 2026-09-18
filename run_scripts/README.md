# Remote Qwen3-32B / ProphetKV / NIAH multivalue

Copy this **repository, including `run_scripts/`**, to the remote server. Run the
commands below there. Nothing in these scripts submits work to the old server.
The historical benchmark drivers and results are not changed or resumed.

## Experiment

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-32B`, original BF16 checkpoint |
| Chat | Native Qwen3 template with `enable_thinking=False` |
| Generation | Greedy, maximum **128 new tokens**, one request at a time |
| Task | RULER `niah_multivalue` |
| Dataset targets | 8,192 / 16,384 / 32,768 / 65,536 tokens |
| Methods | No cache; ProphetKV **5%, 10%, 20%, 30%, 40%, 50%** |
| Samples | Configurable; default **100 per context length**, seed 42 |
| Measured requests | Default **400 prompts × 7 = 2,800** |
| GPUs | Remote physical GPUs 0–7, one **TP=8** engine |
| Context chunks | 4,096 tokens, including numbering and padding |
| Fresh suffix | At least 256 tokens; includes the complete query |

100 samples is a configurable assumption, not the RULER default of 500. Set
`NUM_SAMPLES=500` for 14,000 measurements, or `NUM_SAMPLES=5` for a small
140-request trial. Use a different `RESULT_ROOT` for each scope.

All methods use the same frozen token IDs, chat format, chunk markers, padding,
and output budget. Actual prompt lengths are saved; the dataset target is not an
exact input length. Percentages are the fraction of **eligible cached tokens to
recompute**, excluding the exact first chunk and fresh suffix. They are not KV
compression percentages or fractions of the whole prompt. This is the repository's
**ProphetKV UCM/vLLM port**.

The Qwen3-32B [model card](https://huggingface.co/Qwen/Qwen3-32B#processing-long-texts)
documents YaRN for extended contexts. The 64K runs use its 4× configuration
(`original_max_position_embeddings=32768`); shorter runs use original RoPE.
4× leaves room for markers and generated tokens beyond the 65,536 target.
All methods within a length use the same setting. Checkpoint files are never edited.

## Configure once on the remote server

Edit `run_scripts/config.sh`, or export overrides in your remote shell:

```bash
cd /path/to/unified-cache-management
export PYTHON_BIN=/path/to/prepared/environment/bin/python
export MODEL_PATH=/path/to/Qwen3-32B
export NUM_SAMPLES=100
export RESULT_ROOT=/path/to/results/qwen3-32b-prophetkv-niah100
export CACHE_ROOT=/path/to/local-nvme/prophetkv-niah100
# Optional, if your complete RULER checkout is elsewhere:
export RULER_ROOT=/path/to/RULER
```

`PYTHON_BIN` must refer to the prepared coupled runtime: UCM 0.3.0, patched
vLLM 0.9.2 (including Qwen3 sparse hooks), PyTorch 2.7.0, Transformers 4.53.2.
It also needs the RULER generator dependencies and NLTK sentence-tokenizer data.
An ordinary unpatched vLLM installation does not implement the required hooks.
The scripts check this and do not install or upgrade packages.

`MODEL_PATH` is a **local complete checkpoint directory**, not a Hugging Face
model ID; preparation and inference operate offline. `RULER_ROOT` defaults to
`benchmarks/vendor/RULER`. Its
`scripts/data/synthetic/json/PaulGrahamEssays.json` must already be populated.
If needed, run `bash run_scripts/prepare_ruler_assets.sh` once; this optional
command downloads RULER's essay corpus and missing NLTK data, but no model or packages.
You can instead point `RULER_ROOT` at your already prepared checkout.

Use a local SSD/NVMe for `CACHE_ROOT`. A 64K BF16 Qwen3-32B cache is roughly
16 GiB across TP shards before overhead; allow at least 25 GiB of free cache
space. Results need additional space for per-rank diagnostics and smoke tensor
artifacts. Cache storage is bounded to one sample and removed after its engines exit.

## Launch the whole experiment

```bash
bash run_scripts/plan.sh          # No GPU initialization; print exact scope
bash run_scripts/preflight.sh     # Read-only environment/model/GPU checks
bash run_scripts/prepare.sh       # CPU: generate data, freeze prompts and sources
bash run_scripts/detach.sh        # Start/resume; survives SSH logout
bash run_scripts/status.sh        # Confirm supervisor_live: true after detach exits
```

`detach.sh` uses `nohup`, `/dev/null` stdin and a separate process session.
`supervisor.log` is appended on resume. Only one supervisor may hold `run.lock`.
GPUs are identified using `nvidia-smi -L`, frozen by UUID, and checked again
before every engine launch and inside every rank. Busy selected GPUs are awaited.
There are no idle placeholders or reservation jobs.

The runner automatically gates **each length** on the longest prepared prompt:
all seven requested configurations, 0% and 100% controls, and an independent
native exact-prefix reference. Smoke uses 16 output tokens and does **not** count
as measured work. Cached smoke checks every layer and TP rank for writes,
preserved KV, selection agreement and original-position causal attention. The
100% control must match the native-prefix reference at all 64 layers.

For interactive execution use `bash run_scripts/run_all.sh` instead of detaching.
To run only qualification first: `bash run_scripts/smoke.sh`. Do not run these
simultaneously with a detached supervisor.

**Engine lifetime:** this launcher deliberately uses a fresh engine for each
sample/method plus one cache-population engine per sample, approximately 3,200
engine starts for the default measurement sweep, plus smoke/recovery. Loading a
32B model that often adds substantial wall-clock time and disk traffic. This is
not the historical persistent-engine scheduler. Loading and population are
reported separately and excluded from TTFT. All eight GPUs participate in each
engine; they are not eight simultaneous independent model replicas.

## Run an individual length or method

```bash
bash run_scripts/run_8k.sh
bash run_scripts/run_16k.sh
bash run_scripts/run_32k.sh
bash run_scripts/run_64k.sh

# One context length and one method; the length's smoke gate still applies:
bash run_scripts/run_one.sh 65536 baseline
bash run_scripts/run_one.sh 65536 prophetkv-5
bash run_scripts/run_one.sh 65536 prophetkv-10
bash run_scripts/run_one.sh 65536 prophetkv-20
bash run_scripts/run_one.sh 65536 prophetkv-30
bash run_scripts/run_one.sh 65536 prophetkv-40
bash run_scripts/run_one.sh 65536 prophetkv-50

# Same selection, detached:
bash run_scripts/detach.sh --length 65536 --method prophetkv-20
```

These share the result directory and skip validated work. Run them sequentially,
or simply use `detach.sh` for all lengths/methods. A run interrupted by an engine
error stops with its log preserved; fix the cause and invoke the same command
to resume. Unaccepted attempt files are archived. Changing prompts, checkpoint,
runtime, runner source, sample count, chunk size, or GPU assignment requires a
new result directory; incompatible results are never silently mixed.

## Monitor and read logs

```bash
bash run_scripts/status.sh
bash run_scripts/logs.sh                     # Follow current/last engine log
source run_scripts/config.sh
tail -n 80 -F "$RESULT_ROOT/supervisor.log"   # Scheduling, gates, failures
nvidia-smi
```

`logs.sh` follows the engine active when invoked. Re-run it when the method
changes. Ctrl-C stops only the log viewer.

| File or log message | Meaning |
|---|---|
| `progress.json` | Overall accepted count, target, per-length/per-method counts |
| `supervisor.json` | Supervisor PID and identity; `status.sh` checks if it is actually live |
| `active.json` | Current sample, method, engine process group and log path |
| `logs/prepare-<length>.log` | RULER data-generation output |
| `cache-builds/*.log` | Offline independent-chunk construction |
| `cache_population chunk=X/Y` | Cache building; outside TTFT |
| `MEASURE_BEGIN` / `MEASURE_END` | Boundaries of the timed request |
| `engine_progress` | Engine still stepping; not a completed measurement |
| `result ... score=... ttft=... total=...` | Request output and timing |
| `WORKER_COMPLETE` | Worker shut down normally; validation follows |
| `*.validated.json` | Record, log and diagnostics hashes after acceptance |
| `SMOKE_GATE_PASSED` | Qualification for that context length passed |
| `state: failed` or `Traceback` | Inspect indicated engine log; do not count incomplete output |

The watchdog stops an owned engine after 1,800 seconds without logged progress
or on a fatal log. Set `WATCHDOG_SECONDS` higher before launching if remote model
loading or initial kernel compilation legitimately needs longer. GPU allocation
alone does not establish progress.

## Results and interpretation

Default directory:
`.results/qwen3-32b-prophetkv-niah-multivalue/` (or `$RESULT_ROOT`).

| Artifact | Contents |
|---|---|
| `final/comparison.csv` | **Main table:** one row per length and method |
| `final/REPORT.md` | Scope and metric interpretation |
| `final/raw_records.jsonl` | All predictions, output IDs, references, hashes, timings and provenance |
| `final/cache_build_costs.json` | Offline population costs, including resumed attempts |
| `final/validation.json` | Completion certificate: all required measurements validated |
| `records/<sample-id>/<method>.json` | Individual raw measurement |
| `records/<sample-id>/<method>.log` | Individual engine log |
| `records/<sample-id>/<method>.diagnostics.json` | TP-rank scores, selected positions and layer counts |
| `smoke/<length>/validation.json` | Per-length qualification receipt |
| `protocol.json`, `prompt_manifest.json` | Frozen settings, hashes and actual input lengths |
| `cleanup.json` | Owned engine exit and temporary cache cleanup receipt |

`final/` is produced only after every requested measurement validates. For an
incomplete stopped run, use `bash run_scripts/report.sh` to produce `partial/`.
Reporting takes the same lock and must run after the supervisor exits.

In `comparison.csv`:

- `accuracy_percent`: official RULER fraction of reference strings found in
  the output, averaged over samples. Higher is better; this is not exact-match
  answer accuracy.
- `mean_ttft_seconds`: time from engine submission to its first token-bearing
  output. Includes online ProphetKV probing, transfers and selection. Lower is better.
- `ttft_speedup`: **mean paired no-cache TTFT / mean paired method TTFT**.
  Above 1 is faster. `paired_samples` tells you how many matched prompts contribute.
- `mean_generation_seconds`: total request generation latency, separate from TTFT.
- `length_limited`: count of outputs reaching the 128-token cap; inspect raw
  predictions and `finish_reason` when interpreting scores.

Offline construction/readiness waits, loading and warmup are excluded from TTFT.
The cache backend uses buffered local storage; these are warm-cache results,
not a cold-disk measurement. One timing per prompt is one accuracy sample.

## Stop, resume, and check the scripts

```bash
bash run_scripts/stop.sh          # SIGTERM only to the identity-verified supervisor
bash run_scripts/status.sh        # Confirm supervisor_live: false and engines exited
bash run_scripts/detach.sh        # Resume missing work using the same settings
bash run_scripts/test_cpu.sh      # CPU unit tests; no inference
```

If the supervisor was forcibly killed and an engine group survived, the launcher
refuses to restart or remove its cache. Inspect `active.json` and the process
identity before taking manual action. It never kills unrelated GPU processes.

Local validation covers CPU logic, TP score aggregation, YaRN delta rotation,
shell syntax and command generation. **The 32B TP=8 GPU path has not been executed
on your remote machine by this preparation task**; remote smoke gates are mandatory
and a failure leaves measured work for that length unstarted.
