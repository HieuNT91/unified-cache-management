# Agent instructions

## Permanent prohibition on dummy GPU jobs

Never launch dummy GPU jobs, idle GPU placeholders, reservation workloads, or
keep-alive workloads. Leave idle GPUs idle. This standing user instruction
supersedes every historical authorization for dummy jobs or placeholders below.
Do not resume a historical scheduler that launches them without first disabling
that behavior. Actual authorized benchmark work may continue.

## GPU usage

Before launching GPU work, identify the devices with `nvidia-smi -L` and explicitly restrict visibility to
allowed GPUs using their UUIDs, for example `CUDA_VISIBLE_DEVICES=GPU-<allowed-uuid>`.
Apply the same restriction to containers and subprocesses; never launch with all
GPUs exposed. CUDA logical indices can be remapped, so verify the physical GPU
identity rather than relying on a process-local device index. If no other GPU is
available, use the CPU or report that GPU execution is blocked.

## Network access and proxies

Codex needs `HTTPS_PROXY` for its own connection in this environment, but the
proxy substantially slows shell downloads and uploads. Prefer direct connections
for network commands when the destination is reachable without the proxy.

Remove proxy variables only for the individual command:

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy <command> <args>
```

For curl, also use `--noproxy '*'` to explicitly disable proxy use. For Python
urllib, use `urllib.request.build_opener(urllib.request.ProxyHandler({}))`.

Do not unset the proxy globally or change Codex's launch environment. If a
destination requires the proxy, retry that command with its normal proxy settings.
Removing proxy variables does not bypass sandbox network restrictions; follow
the normal approval process when the sandbox blocks a required network command.

A quick Cloudflare test on 2026-09-15 measured 676 Mbps download, 237 Mbps upload,
and 44 ms HTTPS request latency with proxies disabled. The earlier test with the
inherited environment measured 23 Mbps, 31 Mbps, and 572 ms respectively. These
are indicative measurements, not guaranteed speeds or ICMP ping times.

## Cache modes and benchmark interpretation

These definitions describe the local runner in
[`benchmarks/cacheblend_ruler.py`](benchmarks/cacheblend_ruler.py):

- **No cache (`baseline`)**: no UCM connector; vLLM prefix caching is disabled.
  The entire prompt is computed for every measured request. Normal KV caching
  during autoregressive decoding still occurs; "no cache" means no reuse from
  an earlier request.
- **Full cache (`full_cache`)**: prepopulate the entire exact request, including
  its question, then reuse its prefix KV through the UCM connector. Chunk-end
  recognition is disabled (`chunk_end_token_id=-1`). On a complete hit, vLLM
  still recomputes the final prompt token. This is an ideal exact-request reuse
  case and can reasonably have the best TTFT; it is a different reuse opportunity
  from independently cached chunks.
- **CacheBlend (`cacheblend`)**: prepopulate independent, numbered, block-padded
  context chunks; reuse the matching prefix and blend later cached chunks with
  selective recomputation. The final 256 prompt tokens are left fresh by default.
  `--blend-recompute-ratio 0.2` means 20% selection within the eligible cached
  chunk region, not 20% of every token or layer. The exact prefix is reused and
  the fresh suffix is computed regardless of ratio. A 100% ratio is a useful
  control but is not equivalent to a cache-free request.

CacheBlend trades computation for approximation. Accuracy need not improve
monotonically with the ratio. Equal task scores do not prove token-level or
tensor-level equivalence, and latency ordering alone does not establish a bug.

## RULER protocol and validation

- Local RULER source: `.downloads/RULER`. Its supplied task configuration defaults
  to 500 samples per task; do not call a smaller local dataset the full default
  evaluation. The original `.data/RULER/qwen3/65536` has only three samples/task.
- The 13 tasks are `niah_single_1`, `niah_single_2`, `niah_single_3`,
  `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`, `niah_multivalue`,
  `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, and `qa_2`.
- Use identical samples, tokenized prompt hashes, generation budgets and repeat
  counts across modes. Default `--prompt-layout chunked` adds the same markers
  and padding in all modes. This modifies the original RULER prompts. "64K" is
  the target dataset length; record actual token lengths, which vary by sample.
- **TTFT** is elapsed monotonic time from engine submission to the first engine
  output containing a token ID (`engine_step_first_token_monotonic`, schema 2).
  Record total generation time separately. Never substitute total `llm.generate()`
  latency for missing TTFT. Model loading, tokenization, cache construction,
  cache-readiness waits and warmup are excluded; these are offline engine timings.
- The NFS-named backend currently uses buffered local storage. Warm-cache results
  do not establish cold-disk performance. Report cache-building cost separately.
- Before measuring cached modes, verify every expected 64-token block and every
  tensor-parallel shard, including file size. This verifier targets UCM 0.3.0's
  NFS hash format and checks availability, not numerical tensor correctness.
  Require measured cache-hit evidence and reject cache/engine errors in logs.
- Scoring matches local official RULER substring metrics and control-character
  cleanup: QA accepts any reference; other tasks count the fraction of references
  present. Keep raw predictions, token IDs, references and `finish_reason`.
- The runner uses greedy raw completions with RULER task output limits. Thinking
  models can exhaust those limits; a matching substring does not prove a completed
  or coherent final answer. Report length-limited outputs and sample counts.
  Repeated timings of one prompt are not additional independent accuracy samples.
- Local NIAH generation puts the needle's character position in `index`, which
  can collide. Suite datasets use row ordinals as unique IDs and preserve the
  original value in `source_index`; do not merge distinct prompts by that value.
- Historical `.results/ruler64k-20260915` TTFT claims were withdrawn: those values
  were total generation times, and prompt layouts differed. The later two-sample
  ratio sweep is a smoke validation, not a broad RULER accuracy result.

## Runtime and known failures

Use `.envs/cacheblend`: installed UCM **0.3.0**, patched vLLM **0.9.2**, PyTorch
**2.7.0**, Transformers **4.53.2**, and `.tools/cuda-12.6`. Launch inference from
`/tmp` with `PYTHONPATH` removed so the newer development UCM checkout is not
imported accidentally. The suite runner supplies these settings and explicit
GPU UUID visibility. Do not independently upgrade these coupled dependencies.
Qwen model hook fixes are retained in
`ucm/integration/vllm/patch/0.9.2/vllm-adapt{,-sparse}.patch`; reinstalling vLLM
can remove installed patches. Passing this benchmark does not certify every
model or UCM code path.

- Prompts shorter than one 64-token block exposed a UCM sparse-metadata `KeyError`.
  The runner uses a 128-token warmup to avoid that path; the underlying short-prompt
  bug remains unresolved.
- Concurrent cache population exposed a readiness race: a fixed one-second delay
  was insufficient for asynchronous writes. `wait_for_cache` now polls outside TTFT (120 seconds by default; the expanded suite uses 600 seconds
  under concurrent I/O load). Do not remove this check or accept incomplete caches.
- vLLM can hang during shutdown after a traceback, retaining GPU memory at 0%
  utilization. The suite watchdog terminates the engine process group on fatal
  logs or 900 seconds without logged progress, and retries once. Check recent
  logs and validated counts, not just a stale `state: running` or allocated VRAM.

## Latest requested experiment (2026-09-16)

The latest scope supersedes the cancelled 500- and 50-sample runs:

- **Qwen3-4B-Thinking-2507**, all 13 tasks, **20 samples/task**.
- **No cache, full cache, CacheBlend 10%, 20%, 30%, 50%, 60%, 90%**.
  The user added 30% and 50% to the original six-configuration suite; preserve
  completed results and recover missing original measurements.
- One measurement/sample/configuration: **260 samples per configuration,
  2,080 measured requests total**. Data is the first 20 rows/task from the
  validated 50-sample dataset, at `.data/RULER/qwen3-20/65536`.
- Authorized physical GPUs are **1, 2, 3**; exclude **0 and 4** for this suite.
  This supersedes the earlier request to exclude the last two GPUs.
- BF16, tensor parallel size 1, eager execution. Batches contain four prompts;
  all configurations for a batch run on the same GPU. CacheBlend ratio order
  rotates across batches. Temporary KV storage is bounded per worker and cleaned
  only after its engine exits; never delete caches used by a live worker.

### Files and operation

- Suite driver: [`benchmarks/run_ruler_suite.py`](benchmarks/run_ruler_suite.py).
  Current dataset, ratios, GPUs and result directory are defined there.
- Results: [`.results/ruler64k-20-20260916/`](.results/ruler64k-20-20260916/).
  Read `protocol.json`, `dataset_manifest.json`, and `prompt_manifest.json` for
  exact model/checkpoint, source hashes, sample selection and prompt identities.
- Read `supervisor.json`, `gpu-{1,2,3}.json`, `progress.json`, and recent logs for
  live status; do not treat progress numbers in conversation history as current.
  `shards/<task>/<start>/` holds per-mode records, summaries, logs and validation
  markers. `REPORT.md` shows progress; final `comparison.csv` is written only
  after every required measurement passes validation.
- To resume, run `python -u benchmarks/run_ruler_suite.py run` from the repository
  root **only after confirming no supervisor is running**. `run.lock` prevents
  concurrent supervisors. Validated batches are skipped; incomplete work may
  repeat. The driver refuses to mix changed benchmark source snapshots on resume.
- To stop, verify the PID and command in `supervisor.json`, then send SIGTERM to
  that supervisor. It stops its engine groups and preserves validated results.
  Never kill unrelated users' GPU processes.
- The original launch was a Codex child under VS Code, without `nohup`. On
  2026-09-16 the user requested detachment: the supervisor was stopped cleanly
  and resumed through `nohup` with `start_new_session=True` (setsid semantics).
  After the launcher exited, its parent PID was 1, PID/PGID/SID matched, SIGHUP
  was ignored, stdin was `/dev/null`, and output went to `supervisor.log`.
  See `nohup-launch.json` in the result directory for launch provenance. Verify
  current process state rather than assuming this historical launch still lives.
  For future unattended launches, use a verified detached supervisor (`nohup`
  with a separate session, or `tmux`) and confirm survival after the launcher
  exits. Redirected logs alone are insufficient. Never start a duplicate supervisor.

CPU validation command (no inference):

```bash
python -m unittest discover -s benchmarks -p 'test_*ruler*.py' -v
```

Further context: [benchmark protocol](benchmarks/CACHEBLEND_VALIDATION.md),
[initial validation](.results/benchmark-validation-20260916/VALIDATION.md),
[small ratio sweep](.results/benchmark-validation-20260916/RATIO_SWEEP.md), and
[current suite notes](.results/ruler64k-20-20260916/README.md). Older notes contain
historical GPU restrictions and sample sizes; use the latest user instructions
and current suite protocol when they differ.

## Selector experiment GPU expansion (2026-09-17)

The K-only, V-only, and K+V experiment in
`.results/cacheblend-selectors-20260917/` now uses physical GPUs **1, 2, 3, 4**,
following the user's latest request. GPU 0 remains forbidden. This changes the
selector experiment's original GPU-4-only scope; the separate RULER suite has
finished and its results remain intact.

`benchmarks/run_selector_suite.py` runs one independent sample worker per GPU.
Each sample stays on one GPU across all configurations. Preserve the original
`protocol.json`, original `source/`, and existing validated records. The explicit
`execution-gpus1-4.json` amendment binds revised orchestration sources in
`execution-v2/source/`, preserved artifact hashes, and sample GPU assignments.
New records retain both protocol and execution hashes plus GPU provenance.
Inference, selector arithmetic, prompt IDs, output budgets and scoring are
unchanged. The sample in progress at the transition spans concurrency phases;
report that limitation when comparing its timings.

Read `supervisor.json`, `progress.json`, `gpu-{1,2,3,4}.json`, and recent logs for
status. `active.json` is historical single-GPU state. Resume with
`python benchmarks/run_selector_suite.py detach` only after confirming no
supervisor is running; `run.lock` prevents duplicates. Stop only the verified
supervisor PID with SIGTERM. Caches remain bounded to one sample per GPU and
must not be cleaned while their own engine group is alive.

## Selector scope reduction (2026-09-17, latest instruction)

The user requested: **stop LongBench v2 jobs; finish RULER jobs**. The selector
experiment now targets **100 RULER measurements**; 37 completed LongBench
records are preserved and no further LongBench jobs are authorized. The original
200-request study remains incomplete by user choice. Keep `protocol.json`,
`execution-gpus1-4.json`, pinned worker sources and measured records unchanged.

Use `python benchmarks/finish_selector_ruler.py detach` only if a resume is
needed and no supervisor is running. **Do not launch the old
`run_selector_suite.py` scheduler**, which would resume cancelled LongBench work.
The RULER completion runner shares `run.lock`; `scope-ruler-only/amendment.json`
pins the new scope and additional source snapshots. Final RULER tables, raw
records, five plots, report and validation certificate are in
`.results/cacheblend-selectors-20260917/ruler-final/`. Root `progress.json`
reports progress against 100 RULER requests and the preserved LongBench count.
The old supervisor required forced termination after a SIGTERM timeout;
cancelled LongBench retries and the adoption of the final in-flight RULER engine
are documented under `scope-ruler-only/`. The final RULER measurements span
cancellation of concurrent workers; retain that timing caveat in analysis.

## Selector expansion to twenty RULER prompts (2026-09-17, latest instruction)

The user requested ten additional RULER 64K samples on top of the completed ten.
The target is now **20 distinct prompts and 200 RULER measurements**, retaining
all 100 original measurements unchanged and running only 100 additional ones.
There are two prompts per each of the original ten tasks, not twenty per task.
LongBench remains cancelled and its 37 validated records are preserved.

Current driver: `benchmarks/extend_selector_ruler.py`. Current status and outputs:
`.results/cacheblend-selectors-20260917/ruler20-extension/`. Read its
`progress.json`, `supervisor.json`, `gpu-{1,2,3,4}.json` and recent logs. The
parent directory's progress/report describe the completed first ten prompts.
The new runner acquires both the parent and extension locks; do not start any
historical selector scheduler. Resume only with
`python benchmarks/extend_selector_ruler.py detach` after confirming no live
supervisor. Stop only its verified supervisor PID; the main thread polls futures
with a one-second timeout so Python stop handlers are dispatched promptly.

`extension.json` pins the original artifact hashes, additional source snapshots
and new protocol/execution hashes. New samples use one unused source row from
each original task, selected with `random.Random(20260917)`, and the identical
tokenizer/chunking function. Token IDs, boundaries, source rows and references
are saved once under `samples/`; original inputs are not rewritten. Actual new
prompt lengths are 63,104–65,600; model length remains 73,792. All other model,
selector, ratio, decoding, timing, diagnostic and cache checks are unchanged.

GPUs 1–4 remain authorized; GPU 0 is forbidden. Each new sample stays on its
assigned GPU for every configuration. Temporary caches are bounded under
`/tmp/ucm-cacheblend-selectors-ruler20-20260917/`, one per worker, and are removed
only after their engine groups exit. New inference uses the original pinned
worker and private UCM; no shared package changes. Final combined 20-prompt
tables, raw records, five plots and report are generated in `combined/` only
after all 200 RULER records validate. Cohort and per-task tables are included.

## Qwen3 Instruct selector/norm study (2026-09-17, latest request)

The new independent experiment uses **Qwen/Qwen3-4B-Instruct-2507**, non-thinking,
**128 maximum new tokens**, and physical **GPUs 1, 2, 3 only**. GPU 0 and GPU 4
are excluded. Use 50 samples each from `cwe`, `vt`, `qa_1`, `qa_2`,
`niah_multikey_3`, `niah_multivalue`: 300 distinct 64K-target prompts. There are
25 configurations: baseline plus K-only/original, K+V, V-only × L1/L2 ×
10%, 20%, 40%, 80%, totaling **7,500 measured requests**.

Driver: `benchmarks/run_instruct_norms.py`. Results and current status:
`.results/qwen3-instruct-ruler50-norms-20260917/`. Read its `protocol.json`,
`execution.json`, `supervisor.json`, `gpu-{1,2,3}.json`, `progress.json`, and logs.
Historical selector runners do not schedule this experiment. Resume only with
`CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_instruct_norms.py detach`
after verifying no live supervisor; `run.lock` prevents duplicates. Stop only its
verified supervisor PID with SIGTERM. Occupied GPUs are awaited, never cleared
by terminating unrelated processes.

The runner pins the checkpoint, datasets, tokenized native non-thinking chat
prompts, worker sources and private UCM copy. It preserves historical results and
shared packages. One sample remains on one assigned GPU across configurations;
cache cleanup requires its engine group to have exited. Cached measurements require
verified complete caches, measured hits and selector diagnostics. Final tables,
raw records, report and validation certificate are under `final/` only once all
7,500 records validate. See `benchmarks/CACHEBLEND_INSTRUCT_NORMS.md` for exact
norm formulas, prompt formatting, timing, validation and operation.

## K/V component-retention experiment (2026-09-17)

The new independent experiment uses **Qwen/Qwen3-4B-Instruct-2507**, native
non-thinking chat, greedy **128 new tokens**, physical **GPU 4 only**, and
**50 samples each** of `vt`, `cwe`, `niah_multivalue`, `niah_single_3` from the
validated 64K-target dataset. GPU 0 remains forbidden. Other experiments and
GPU 1–3 jobs are separate and must not be stopped for this run.

Driver: `benchmarks/run_kv_update.py`. Results/status:
`.results/qwen3-instruct-kv-update-ruler50-20260917/`. Target: **2,600 measured
requests** (200 prompts × 13 configurations), including the requested no-cache
baseline. K-only/V-only here refers to which tensor is updated in attention,
not the earlier selector-only ablations. The other cached component is retained
at every layer; fused projections may still compute discarded values. Two
all-token cases bypass selection; ten percentage cases select with K L1 at
10%, 20%, 50%, 70%, 90%. Exact prefix and fresh 256-token suffix retain their
established semantics.

A private UCM subclass implements retention; shared packages are unchanged.
The supervisor gates the measured suite on thirteen 64K smoke cases, with
per-layer tensor/cache-write audits for the twelve cached cases. Smoke outputs
have a 16-token cap and do not count as measurements. Read `supervisor.json`,
`progress.json`, `gpu-4.json`, recent logs and `smoke/validation.json` for status.
Resume only after verifying no live supervisor, using
`env CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_kv_update.py detach`.
Stop only its verified supervisor PID with SIGTERM. Never remove a cache while
its engine group is alive. See `benchmarks/CACHEBLEND_KV_UPDATE.md` for details.

## All benchmark jobs cancelled (2026-09-17, latest instruction)

The user cancelled every job. Both active supervisors were stopped cleanly:
`run_instruct_norms.py` (343/7,500 validated records preserved) and
`run_kv_update.py` (80/2,600 preserved). No GPU compute processes remained
after shutdown; GPUs 1–4 were released. Each result directory contains a
`cancellation.json` receipt. Do not resume either experiment or any historical
scheduler without a new user instruction. Preserve completed results.

## Instruct study cancelled (2026-09-17, latest instruction)

The user requested cancellation of all jobs. The Instruct norm-study supervisor
and its workers are stopped. Preserve existing measurements under
`.results/qwen3-instruct-ruler50-norms-20260917/`; see `cancellation.json` and
`progress.json`. Do not resume this or any historical benchmark scheduler
without a new user instruction authorizing a run.

## New five-sample NIAH K/V study (2026-09-17, latest instruction)

The user authorized a **new** independent run; the previously cancelled studies
remain cancelled. Use Qwen/Qwen3-4B-Instruct-2507, native non-thinking chat,
greedy max-new-tokens 128, **physical GPU 4 only**. Tasks: `niah_multivalue` and
`niah_single_3`, **five samples each**, using source rows 0–4 of the validated
64K-target dataset. Five configurations: no cache; K-only/all eligible tokens;
V-only/all eligible tokens; K-only/K-selection 80%; V-only/K-selection 80%.
Target: **10 prompts, 50 measured requests**. Tensor retention and all-layer
semantics match the earlier component experiment; fused QKV may still compute
discarded values. Exact prefix and fresh 256-token suffix are unchanged.

Driver: `benchmarks/run_kv_update_niah5.py`. Results/status:
`.results/qwen3-instruct-kv-update-niah5-20260917/`. Five smoke cases audit the
new scope before measured work. Read `progress.json`, `supervisor.json`,
`gpu-4.json` and recent logs. Resume only if no supervisor is live, with
`env CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_kv_update_niah5.py detach`.
Do not launch historical schedulers. Stop only this verified supervisor with
SIGTERM. Preserve prior results and never clean a cache used by a live engine.

## New 5-sample Instruct norm study (2026-09-17, latest request)

The user authorized a new experiment after cancellation, then reduced its scope
from 50 to **5 samples per task**. Use **Qwen/Qwen3-4B-Instruct-2507**, thinking
disabled, **256 max new tokens**, physical **GPUs 1, 2, 3 only**. Tasks, in requested
order: `niah_multikey_3`, `niah_multivalue`, `cwe`, `vt`. Use source row ordinals
0–4 of each existing 50-row 64K dataset: **20 prompts × 25 methods = 500 requests**.
Methods remain baseline plus original K, K+V, V × L1/L2 × 10/20/40/80%.

New driver: `benchmarks/run_instruct_norms5.py`; worker:
`benchmarks/cacheblend_instruct_norms5.py`. Results:
`.results/qwen3-instruct-ruler5-256-norms-20260917/`.
Read its protocol, execution, supervisor, per-GPU state, progress and recent logs.
Resume only this new study, if needed and no supervisor is live, with
`CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_instruct_norms5.py detach`.
The cancelled 128-token norm study and KV-update study remain cancelled; preserve
their records. No 50-sample/256-token study was launched.

This study retains fresh engines, the isolated norm implementation, native
non-thinking chat prompts, cache checks, 0.93 GPU memory allocation target,
per-sample GPU affinity and bounded caches. Its own protocol pins all 20 prompt
identities, checkpoint and source hashes. Stop only its verified supervisor PID
with SIGTERM. Final per-task and aggregate tables, raw records and validation
certificate are produced under `final/` after all 500 measurements validate.

## Random chunk-token skipping study (2026-09-17, latest GPU-4 request)

New independent driver: `benchmarks/run_random_skip.py`. Results:
`.results/qwen3-instruct-random-skip-niah5-20260917/`. Use GPU **4 only**,
Qwen/Qwen3-4B-Instruct-2507, native non-thinking chat, greedy 128 new tokens,
the same five `niah_multivalue` and five `niah_single_3` 64K-target prompts.
Run a fresh no-cache baseline and `random-skip-20`: **20 measured requests**.
Other studies, including the separately authorized norm study on GPUs 1–3,
are outside this task and must not be stopped.

The user clarified: sample exclusions equal to **20% of each whole chunk**,
chosen uniformly without replacement from its **40%–100% positional region**.
Keep CacheBlend's first-chunk and first-layer behavior unchanged. The exact first
chunk is reused; layer 0 follows the original dense scheduled-token path. Starting
before layer 1 QKV projection, omit skipped queries from QKV, attention and FFN;
their aligned cached K and cached V remain intact. All remaining tokens recompute
**both K and V**. Fresh suffix (256 tokens), cache misses and decode stay fresh.
Use ceil(40% * padded chunk length) as the start and floor(20% * chunk length)
as the exclusion count; markers/padding are included in the established chunk
length. A seed-20260917, prompt/chunk-specific mask is saved once and fixed across
subsequent layers. This is one random draw per prompt, not a multi-seed study.

The private UCM override and sources are pinned. CPU tests cover masks, first-layer
behavior, pruning before projections, cache misses, decode and cache-write audits.
The supervisor gates measurements on no-cache/random-skip 64K smoke cases, with
per-layer audits of both recomputed tensors, skipped cache values and first-chunk
cache preservation. Read protocol, progress, supervisor, gpu-4 state and logs.
Resume only this driver, after verifying no live supervisor, using
`env CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_random_skip.py detach`.
Keep all earlier result directories intact. Never clean caches owned by live engines.

## Random skip changed to 10% from the last 40% (2026-09-17, latest GPU-4 request)

The user cancelled the 20%-skip study; its supervisor exited and **14/20**
validated records are preserved under
`.results/qwen3-instruct-random-skip-niah5-20260917/`, with cancellation receipt.
Do not resume that study.

New driver: `benchmarks/run_random_skip10.py`. New results/status:
`.results/qwen3-instruct-random-skip10-tail60-niah5-20260917/`. Same ten NIAH
prompts, model, non-thinking chat, 128-token budget and physical GPU **4 only**.
Compare fresh baseline vs `random-skip-10`: **20 measured requests**. Sample
floor(10% * each whole later padded chunk length) without replacement from
ceil(60% * chunk length) through its last token. First 60% always computes.
Exact first chunk and original CacheBlend layer-0 behavior are unchanged. Starting
before layer-1 QKV, exclude sampled queries and keep both cached K and V; recompute
both tensors for all other tokens. Suffix, cache misses and decode stay fresh.
Fixed prompt/chunk-specific seed-20260917 masks; one draw per prompt.

Sources/private UCM are pinned separately; earlier records remain intact. The
new supervisor gates measured work on baseline/random-skip 64K smoke cases with
36 per-layer audits. Resume only if no supervisor is live, using
`env CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_random_skip10.py detach`.
Other GPU 1–3 studies are unrelated; do not stop them. Never remove live-engine
caches. Read protocol, progress, supervisor, gpu-4 state and recent logs for status.

## QCFuse UCM/vLLM comparison (2026-09-17, latest GPU 1–3 request)

The user authorized an independent QCFuse port and **180 measured requests**:
20 byte-identical prompts from the 5-sample/256-token norm study, baseline plus
original CacheBlend K/L1 and QCFuse at 10/20/40/80%. Model and decoding remain
Qwen3-4B-Instruct-2507, non-thinking, BF16, TP=1, greedy 256-token cap. Only
physical GPUs **1, 2, 3** are authorized; GPU 0 and GPU 4 are excluded.

Driver: `benchmarks/run_qcfuse.py`; worker: `benchmarks/cacheblend_qcfuse.py`.
Results: `.results/qwen3-instruct-qcfuse-ruler5-256-20260917/`. The supervisor
waits for the existing norm study's verified **500/500 certificate and engine
exit**. It does not cancel/restart that dependency. Preparation and CPU tests
may run before the dependency finishes; GPU calibration/smoke may not.

The pinned reference is uYanJX/QCFuse commit
`38795d91d900debb5f2df23bd7296b3ff97960c8`. Label results **QCFuse UCM/vLLM port**,
not SGLang pipeline reproduction. Offline KVzip anchors retain 15% per layer
(the paper describes 10%). Three Qwen3-4B critical layers are profiled on 128
seed-20260917 SQuAD v1.1 training examples and frozen before evaluation. Primary
TTFT includes online query probing, transfers and selection. Masks are consumed
once before layer-0 QKV; no measured query mask may be reused from warmup.

Read `protocol.json`, `execution.json`, `progress.json`, `supervisor.json`,
`gpu-{1,2,3}.json` and recent logs. The private UCM copy and sources are pinned;
shared runtime files and earlier experiments must stay unchanged. The supervisor
is locked and detached, with a 600-second no-progress watchdog and one retry.
Only resume this driver if needed, after confirming no supervisor is live:
`CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_qcfuse.py detach`.
Stop only its verified supervisor with SIGTERM. Never resume historical jobs.

All nine 64K smoke configurations, including 36-layer tensor audits for cached
cases, gate measured evaluation. `final/validation.json` requires all 180
validated records and report artifacts. `cleanup.json` records owned process/GPU
release before bounded temporary-cache removal. Never remove live-engine caches
or terminate unrelated processes. See `benchmarks/QCFUSE_RULER.md` for formulas,
port adaptations, validation and timing details.

### QCFuse reporting correction (2026-09-18)

The QCFuse dependency completed 500/500; calibration froze zero-based critical
layers `[26, 23, 21]` from all 128 examples. All nine 64K smoke configurations
passed. Read live progress rather than treating historical counts as current.

The pinned supervisor's final reporting scan incorrectly includes
`cache-builds/*.anchors.json` among timing records. Do not change its pinned
sources or measured records. `benchmarks/finalize_qcfuse.py --wait` is a separate
CPU-only finalizer, queued detached to run after the inference supervisor exits.
It validates all 180 measurements, smoke/calibration receipts and GPU cleanup,
then produces the report, paired tables, costs, five plots and certificate.
Its reporting-only amendment preserves original protocol/execution hashes and
all measured artifacts. `reporting-launch.json`, `reporting-detachment.json`
and `reporting.log` track this process; `reporting.lock` prevents duplicates.
Do not start a second finalizer while it is live. If reporting needs recovery,
run `CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/finalize_qcfuse.py`
only after verifying both supervisors have exited. An original supervisor
`KeyError: 'cache_build_seconds'` at 180/180 is this reporting issue, not an
inference failure. Final completion is established by `final/validation.json`,
`reporting-v2/completion.json`, root progress and the cleanup receipt.

## QCFuse causal correction (2026-09-18)

The original QCFuse run's 180 measurements are preserved but withdrawn due to
incorrect sparse-query causal visibility; see its semantic-validation-review.json.
Do not report those results as a validated QCFuse port. The current driver remains
benchmarks/run_qcfuse.py; corrected results are under
.results/qwen3-instruct-qcfuse-ruler5-256-causal-20260918/.
The private port now uses the authors' original-position causal attention kernel;
all 180 requests are rerun after CPU/GPU numerical tests and the nine smoke cases.
Original local CacheBlend controls remain unchanged, a documented comparison
limitation. GPUs 1–3 only; all earlier frozen sources/records remain intact.

## All jobs stopped (2026-09-18, latest user instruction)

The user requested "stop all job for now". The corrected QCFuse supervisor and
its owned engine groups are stopped; no GPU compute allocations remain. Results,
calibration and completed smoke checks are preserved under
.results/qwen3-instruct-qcfuse-ruler5-256-causal-20260918/. See cancellation.json
and cleanup.json. Do not resume QCFuse or any historical scheduler without a new
user instruction. The original QCFuse 20260917 measurements remain withdrawn.

## ProphetKV UCM/vLLM port (2026-09-18, latest authorization)

The user authorized a new independent ProphetKV experiment on physical GPUs
**1–4**, with GPU **0 forbidden**, plus owned dummy jobs on idle authorized GPUs.
Previous cancelled experiments remain cancelled. Driver:
`benchmarks/run_prophetkv.py`. Results:
`.results/prophetkv-qwen3-ruler30-20260918/`. See `benchmarks/PROPHETKV.md`.

Scope is Qwen3-4B-Instruct-2507, native non-thinking chat, greedy 128 tokens,
BF16/TP1/eager; cwe/vt/qa_1/qa_2/niah_multikey_3/niah_multivalue, 30 rows per
8K/32K/64K target, chunks 512/4096. Baseline, ProphetKV 10/20/50/90 and corrected
causal CacheBlend K/L1 10/90: **7,560 measurements** if all gates pass. Frozen
10-row pilots count toward the total. Numerical tests and 0/100% model audits
precede each pilot; accuracy/speed gates stop advancement when unmet. No claim
of full completion without `final/validation.json` and cleanup receipt.

The reusable registered method is `ucm/sparse/prophetkv/`. Runtime uses a pinned
private copy of UCM 0.3.0 and existing patched vLLM; shared packages and historical
sources/results remain unchanged. Request metadata contains exact question-token
positions, excludes answers, and is consumed once. Both methods use causal
attention based on original positions; layerwise delta-RoPE runs once/request.

Read `progress.json`, `supervisor.json`, `gpu-{1,2,3,4}.json`, recent logs and
`gates/` for current state. `detachment.json` verifies launcher survival. Resume
only this scheduler, after checking no supervisor is live, with
`CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_prophetkv.py resume`.
Stop only its verified supervisor PID with SIGTERM. Do not resume old studies.

`benchmarks/prophetkv_gpu.py` owns small CUDA placeholder jobs (~480 MiB including
context). Receipts are `placeholder-{gpu}.json`. The supervisor releases its own
placeholder before a worker, waits for unrelated users, and replaces the placeholder
while idle. Stop/completion releases placeholders and bounded caches. Never remove
caches while engine groups are alive; never terminate unrelated GPU processes.

ProphetKV validation notes: fixed 73,792-token engine allocation did not fit this
runtime's profiled KV budget. Engine limits now equal each saved prompt's length
plus 128, rounded to 64, identically across methods, with memory utilization 0.93.
All patched vLLM Python files are pinned as well as private UCM. Query compaction
uses the original-position Triton kernel; native FlashAttention is used only when
original query positions are exactly a contiguous suffix. Do not reintroduce
bottom-right attention for arbitrary compacted queries. The 100% model audit
uses an unchanged per-layer relative-RMS threshold of 0.03, not bitwise equality;
8K/512 passed with maximum 0.02814. Audit tensors are written inside workers:
vLLM callback RPC transport did not preserve nested tensor payloads. Earlier
failed development revisions are preserved and excluded from measurements.

Eleven CPU tests, GPU score/causal/RoPE checks through 64K, and all nine 8K/512
smoke cases passed before pilot launch. These checks do not certify the remaining
context/chunk phases; consult live gates and progress. The supervisor was verified
detached (parent 1, PID/PGID/SID match, SIGHUP ignored, stdin /dev/null). Current
status must still be checked rather than relying on this launch history.

## Corrected QCFuse resumed (latest user instruction)

The user requested "continue" in the QCFuse thread after the stop. Only the
corrected QCFuse comparison is reauthorized on physical GPUs 1–3. Historical
studies, including ProphetKV, must not be resumed by this instruction. Driver:
benchmarks/run_qcfuse.py. Results:
.results/qwen3-instruct-qcfuse-ruler5-256-causal-20260918/. Sources and model/input
hashes are unchanged; calibration is retained. Stop receipts and eight prior
validated smoke cases were preserved under resume-history/ before restarting
the complete smoke gate and the 180-request sweep. See resumption.json and live
supervisor/progress files. The original 20260917 QCFuse results remain withdrawn.

## QCFuse restricted to GPUs 1 and 2 (latest user instruction)

The user requested "use GPU 1 and 2, nohup if sweep". Only physical GPUs 1 and 2
are now authorized for QCFuse. The old three-GPU supervisor was stopped and its
engine allocations released. Current driver: benchmarks/run_qcfuse_gpus12.py.
Resume only this driver with CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python
benchmarks/run_qcfuse_gpus12.py detach after confirming no supervisor is live.
Do not launch the older three-GPU run_qcfuse.py scheduler.

The independent execution-gpus12.json amendment pins the new supervisor, limits
UUID visibility to GPUs 1/2, and assigns all 20 prompts across those GPUs. Original
protocol, inference sources, calibration and prior artifacts remain preserved.
Results remain in .results/qwen3-instruct-qcfuse-ruler5-256-causal-20260918/.
The new launcher uses nohup, a separate session and /dev/null stdin; verify
nohup-launch.json and detachment-gpus12.json. All earlier scope/validation rules
remain in force. Other historical studies remain stopped.

## ProphetKV 64K chunk sweep on GPUs 3 and 4 (2026-09-18)

The user authorized a new independent nohup sweep: Qwen3-4B-Instruct-2507,
non-thinking, greedy 128 new tokens; vt, cwe, niah_single_1, niah_multikey_1,
qa_1; ten source rows 0–9/task from the validated 64K dataset. Baseline plus
ProphetKV 20/30/40/50% across nominal chunk sizes 512/4096/8192/12000 gives
1000 measured requests. UCM 64-token alignment pads nominal12000 chunks to
12032; protocol records nominal and aligned lengths. Exact-prefix reuse grows
with chunk size; rates still exclude prefix and fresh256-token suffix.

Driver: benchmarks/prophetkv64k/run_prophetkv.py. Results:
.results/prophetkv-qwen3-ruler10-64k-chunks-gpu34-20260918/.
Only physical GPUs3 and4 with UUID visibility; GPU0 forbidden. QCFuse on GPUs1/2
is separate and must not be touched. Prior ProphetKV study remains preserved.
The new requested sweep is independent of the old32K performance gate; numerical
and all-layer correctness smoke gates remain mandatory for every chunk layout.

Read protocol.json, progress.json, supervisor.json, gpu-{3,4}.json and recent
logs. Resume only this driver with CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python
benchmarks/prophetkv64k/run_prophetkv.py resume, after confirming no live supervisor.
Nohup, separate session, /dev/null stdin, supervisor lock,600-second no-progress
watchdog and one retry. One sample remains on the same GPU across all layouts.
Owned idle GPU placeholders are released before inference and on completion/stop.
Caches are bounded under /tmp/ucm-prophetkv-qwen3-ruler10-64k-chunks-gpu34-20260918;
never delete before the owning engine group exits. Sources and private UCM pinned.

## QCFuse LongBench v2 ten-sample comparison (latest request)

The user authorized a new independent run on physical GPUs1 and2 only: ten
LongBench v2 samples below64K, no cache and QCFuse10/20/40%, forty measurements.
Driver: benchmarks/run_qcfuse_longbench10.py; worker:
benchmarks/cacheblend_qcfuse_longbench10.py. Results:
.results/qwen3-instruct-qcfuse-longbench10-256-20260918/.

Retain Qwen3-4B-Instruct-2507, native non-thinking chat, greedy256-token cap,
BF16/TP1, corrected original-position causal fusion,15% anchors and the frozen
SQuAD profile [26,23,21]. Deterministic seed20260918 selection from dataset short
subset covers all six domains,1–2 samples/domain. Full formatted prompt lengths
are11072–58688 tokens, no truncation. Questions/options/instructions fit the fresh
256-token suffix. Official LongBench v2 zero-shot template and exact option
extraction/scoring are pinned. Other studies and shared packages stay untouched.

Four longest-prompt smoke cases with36-layer audits gate the sweep. The nohup
supervisor has its own lock, UUID-only GPUs1/2,600-second watchdog,one retry and
bounded per-worker caches. Read protocol,selection-manifest,progress,supervisor,
gpu-{1,2} and logs. Resume only this driver with CUDA_VISIBLE_DEVICES=''
.envs/cacheblend/bin/python benchmarks/run_qcfuse_longbench10.py detach after
verifying no live supervisor. Never launch a historical scheduler for this task.
See benchmarks/QCFUSE_LONGBENCH10.md.

## ProphetKV adds niah_multivalue (latest user request, 2026-09-18)

The user added niah_multivalue to the active 64K ProphetKV sweep on GPUs3/4.
Scope is now six tasks (vt,cwe,niah_single_1,niah_multikey_1,qa_1,niah_multivalue),
ten samples/task, four nominal chunks512/4096/8192/12000 and five methods
(baseline, ProphetKV20/30/40/50):1200 requests. Other settings remain unchanged.
The forty added layouts use original validated64K source rows0–9 and the same
GPU affinity rule. New task follows the five original tasks within each phase.

Current driver: benchmarks/prophetkv64k/run_with_multivalue.py. Results stay under
.results/prophetkv-qwen3-ruler10-64k-chunks-gpu34-20260918/.
Original protocol, execution, pinned sources/private implementation and validated
records stay unchanged. scope-multivalue/amendment.json pins expanded scope and
new input hashes; preserved-records.json binds pre-transition validated artifacts.
The inference worker retains its original protocol hash; the amendment supplies
scope provenance for new records. transition.json and previous-supervisor/ retain
restart history; report concurrency-phase limitations for timing comparisons.

Only resume the expanded driver with CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python
benchmarks/prophetkv64k/run_with_multivalue.py resume after verifying no supervisor
is live. It shares the original run.lock and caches, reuses verified correctness
gates, skips accepted measurements and requires1200 records for final certificate.
Never launch the old1000-request supervisor. GPUs1/2 jobs are separate.

## KV repair CSV and missing-result queue (2026-09-18, latest request)

The user requested filling kv_repair_result.csv with available validated results,
leaving missing results blank, and scheduling missing jobs. Confirmed512-token
chunks; the last duplicated VT column at32K/64K is CWE. Ten samples/group.
TTFT is mean seconds; accuracy is percent. TTFT reduction is100*(baseline_mean−
method_mean)/baseline_mean. Accuracy improvement is relative100*(method_score−
baseline_score)/baseline_score, not percentage points. Undefined zero-baseline
relative changes remain blank. All comparisons use matching no-cache prompts.

CPU updater: benchmarks/fill_kv_repair_report.py refresh|watch. Output is the root
CSV; provenance, original template, report metadata, missing-job manifest and
updater PID/log/lock are under .results/kv-repair-report-20260918/.
The detached updater refreshes every30seconds and exits when all48 groups have
all10 valid records. Check report-supervisor.json and process identity before
starting another updater. It exposes no GPUs. Do not fill with smoke/partial data.

Supplemental driver: benchmarks/prophetkv_csv/run_prophetkv.py. Results:
.results/prophetkv-csv-backfill-20260918/.200 new requests:8K and32K,512 chunks,
NIAH-single-1 baseline/P20/P30/P40 (80 requests), and NIAH-multivalue/VT/CWE
P30/P40 (120 requests). Original baseline/P20 results are preserved and reused;
all supplemental prompts for those three tasks match original tokens exactly.
New single-key data uses local RULER generator, pinned tokenizer and seed42.
Historical vs supplemental phases and some physical GPU assignments differ;
retain this timing-comparison limitation in report provenance.

The supplemental nohup supervisor is queued CPU-only until the active1200-request
64K sweep completes, has a valid final certificate/cleanup receipt, exits and
releases its lock. Only then may it use UUID-restricted physical GPUs3/4, with
numerical/smoke gates, existing watchdog/retry/cache rules and owned placeholders.
It does not stop the active sweep or other studies. The64K multivalue cells are
already scheduled by the active expanded sweep; do not duplicate those jobs.
Resume only the supplemental driver if needed and no supervisor is live, using
CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/prophetkv_csv/run_prophetkv.py resume.

## QCFuse LongBench v2 full short subset (2026-09-18, latest request)

The user expanded the completed ten-sample LongBench v2 experiment to all dataset
`short` rows below64K, explicitly choosing171 samples and excluding nine
oversized documents without truncation. Physical GPUs1–2 only; use nohup.
Preserve the completed40 measurements and original ten inputs byte-for-byte.
The combined target is684 records (171 prompts × no-cache/QCFuse10/20/40),
with644 new requests. Qwen3-4B-Instruct-2507, non-thinking, greedy256 tokens,
BF16/TP1/eager and frozen profile [26,23,21] remain unchanged.

Driver: `benchmarks/run_qcfuse_longbench_short.py`. Results/status:
`.results/qwen3-instruct-qcfuse-longbench-short-20260918/`.
Read protocol, selection-manifest, execution, progress, supervisor and per-GPU
JSONs/logs. Resume only this runner after checking no live supervisor:
`CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/run_qcfuse_longbench_short.py detach`.
Do not restart the completed ten-sample driver or any historical scheduler.

The full query and options are preserved in a per-sample fresh suffix of
max(256,query-length);40 queries need more than256 tokens, at most871.
Eight smoke cases cover all methods on the longest prompt and longest query;
216 cached-layer audits gate new measurements. Old inputs/results stay intact;
combined reports separate retained-ten and short-expansion timing cohorts.
Sources/private UCM are pinned independently. Watchdog600 seconds, one retry,
one bounded cache per GPU. Stop only the verified supervisor, and clean caches
only after owned engines/GPU allocations exit. Other GPU3/4 jobs are unrelated.
See `benchmarks/QCFUSE_LONGBENCH_SHORT.md` for protocol and validation details.

## User's kv_repair.csv adds64K/4096 report section (2026-09-18)

The user supplied kv_repair.csv with the existing8K/512,32K/512,64K/512 sections
and a new64K/4096 section. Fill those sections; do not infer8K/4096 or32K/4096
requests. The original kv_repair_result.csv remains separately maintained at512.
New CPU-only updater: benchmarks/fill_kv_repair_chunks.py refresh|watch.
State/template/provenance/job coverage/lock/log/launch/detachment live under
.results/kv-repair-report-20260918/chunks/. Avoid duplicate updater processes.
Each section uses its own matching full-prefill baseline; require all10 accepted
rows before populating a group. Missing values remain empty. Previous numeric
values are preserved. All pending cells are covered by the active1200-request
sweep and queued200-request supplemental run, so no extra GPU jobs were added.

## CSV TTFT metric changed to speedup (latest instruction)

Both kv_repair.csv and kv_repair_result.csv now use TTFT Speedup (x), replacing
TTFT Reduction (%). Compute mean full-prefill TTFT divided by mean method TTFT
for matching task/context/chunk/source rows; baseline=1.00x. This is a ratio of
arithmetic means, not paired geometric-mean speedup. Accuracy improvement remains
the signed relative percentage against full prefill. Both CPU updaters implement
the new formula; historical reduction CSV/metadata snapshots are retained in their
report-directory history/. GPU inference and queued measurements are unchanged.


## ProphetKV RULER100 + LongBench v2 (latest user scope, 2026-09-18)

The user cancelled all experiments on GPUs1–4 and authorized a new independent
ProphetKV comparison: **100 samples per each of seven RULER tasks** (`vt`, `cwe`,
`niah_multivalue`, `niah_single_1`, `niah_multikey_3`, `qa_1`, `qa_2`), target64K,
chunk4096; no cache, ProphetKV10%, ProphetKV20%. Also all LongBench v2 documents
with actual pinned-tokenizer context length <65536, without truncation or a
length-category filter:184 qualifying samples. Total884 prompts/2652 requests.
Retain Qwen3-4B-Instruct-2507, non-thinking greedy128, BF16/TP1/eager.

Old ProphetKV expanded64K and queued CSV backfill supervisors are stopped with
cancellation and cleanup receipts;958/1200 and0/200 records preserved. The two
CPU CSV watchers were stopped too. Do not resume these or historical jobs.
New driver: `benchmarks/prophetkv_full/run_prophetkv.py`.
New results: `.results/prophetkv-qwen3-ruler100-longbenchv2-64k-c4096-20260918/`.
Read its protocol, progress, supervisor, GPU states and logs for current status.
Before resume, verify no live supervisor; CPU-only command:
`CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/prophetkv_full/run_prophetkv.py detach`.

GPUs1–4 UUID-restricted;GPU0 forbidden. Owned idle placeholders remain authorized.
RULER newly generated100/task,seed42. LongBench official zero-shot MCQ and answer
extraction;query scores include question/choices only, no answer/output instruction.
Fresh suffix is max(256,complete query tail length), privately generalized for
long MCQs. Numerical tests and full36-layer smoke audits (including0/100 controls)
on longest RULER/LongBench prompts and longest query gate measured work.
Frozen prompt/protocol/source hashes; complete caches and measured hit evidence,
fresh engines, stable GPU affinity,600-second watchdog,one retry,bounded caches,
verified nohup detachment. Primary TTFT speedup is paired arithmetic-mean full
prefill TTFT divided by paired arithmetic-mean method TTFT. Preserve prior data.

## ProphetKV matched native-reference gate amendment (latest restart)

The user renewed the full RULER100/LongBench184 nohup scope after the supervisor
stopped at0/2652. Independent native vLLM (no UCM, probe, pruning or custom
attention) reusing exactly4096 first-chunk tokens produced fresh-suffix hidden
states **bitwise equal at every one of36 layers** to ProphetKV100 on the failing
VT case. Native prefix reuse reproduces the original3.184% difference from
cache-free computation; it is not evidence of a ProphetKV-specific mismatch.

Use **benchmarks/prophetkv_full/run_verified.py detach** for the current restart,
with CUDA_VISIBLE_DEVICES=''; never restart the old run_prophetkv.py scheduler
alone, which has the superseded comparison gate. Original protocol, worker,
private UCM, prompts,10/20% inference and full-prefill baseline remain unchanged.
The execution-v2/amendment.json pins the added driver and independent dense
reference; prior failed logs/gate/state are preserved in execution-v2/prior/.
The new gate requires bitwise equality of all36 fresh-suffix tensors against
native dense computation after exact first-chunk reuse, while retaining direct
no-cache differences for reporting. It does not raise the old3% threshold.

Three audited prompts remain required (longest RULER, longest LongBench prompt,
longest LongBench query). Independent native references and all five original
smoke configurations run before measurements. Controls do not count toward2652.
Read current supervisor/progress/GPU state and recent supervisor.log. The new
supervisor is nohup-detached; wait for occupied GPUs and restrict every GPU
launch by verified UUID. Numerical tests and all prior measured safeguards remain.

## Four-GPU dataset-gated ProphetKV scheduling (latest user correction)

The user requested using all four GPUs and removing dummy jobs while benchmarks
run, and reaffirmed **LongBench v2**, not LongBench v1. Dummy holders are now
disabled. Current driver: **benchmarks/prophetkv_full/run_parallel.py**. Resume
only its `detach` command with CUDA_VISIBLE_DEVICES='' after checking for a live
supervisor. Do not launch the superseded run_verified.py or run_prophetkv.py
schedulers. Result root and2652-request scope are unchanged.

Execution-v3/amendment.json pins orchestration only. GPUs1,3,4 can start RULER
after the RULER numerical/model gates pass, while GPU2 completes the LongBench v2
model gates, then joins RULER. Each GPU proceeds to its LongBench v2 prompts only
after those gates pass. All sources/prompts/inference/timing and fixed prompt GPU
affinity are unchanged. The strongest36-layer matched-reference gate remains.
No duplicate supervisors, no unrelated process termination, no GPU0 workloads.
The prior transition encountered a transient owned-group shutdown check; the
previous engine group was verified fully exited before confirming the new run.
Check live progress/supervisor/GPU states and latest logs, not historical errors.

## Persistent ProphetKV restart (latest authorization, 2026-09-18)

The user requested restarting all 2,652 comparisons with resident engines,
preserving the original 33 accepted records separately. Current driver:
`benchmarks/prophetkv_persistent/run.py`; current results/status:
`.results/prophetkv-persistent-ruler100-longbenchv2-64k-c4096-20260918/`.
Physical GPUs 1–4 only; GPU 0 forbidden; dummy jobs disabled. Frozen 700 RULER
and 184 LongBench v2 prompts, assignments and 128-token budget are unchanged.
Fixed engine capacity is 65,792. Three configuration phases per GPU normally
use twelve engines. Qualification gates run before any restarted measurements.

Read the current supervisor/progress/GPU states and session logs. Resume only
`CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/prophetkv_persistent/run.py detach`
after verifying no supervisor or owned engines remain live. The original
`prophetkv_full/run_parallel.py` is stopped and must not be resumed. New
qualification failures are preserved in `qualification-history/`; they are not
sweep measurements. Namespaced block hashes, scheduler completion receipts and
worker transfer barriers are required before deleting individual sample files;
do not remove the open store's `.temp` directory. Keep original data and shared
installations unchanged. See the new result directory's README for operations.

Persistent rollout verification: all qualification gates passed, including
20 requests/configuration with one engine and zero live-allocation growth,
A–B–A isolation, exact persistent/isolated output/selection/count comparisons,
all three 36-layer bitwise native-prefix controls (including the 871-token query),
and injected failure/resume. The detached supervisor entered the sweep and
accepted at least one measurement on each GPU. See `gates/validation.json` and
`rollout-verification.json`; use live progress for current counts. Normal sweep
initializations are twelve; qualification/recovery starts are separate.
