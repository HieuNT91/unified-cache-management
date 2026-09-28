# Agent instructions

- Scope: Qwen3-32B BF16, YaRN 4× (32768 → 131072), TP configurable (default 4).
- Exactly three public modes: `baseline`, `prophetkv`, `selective_prophetkv`.
- All modes use chunked prefill with at most 16384 scheduled tokens per step.
  Prefix caching stays disabled. Baseline has no UCM connector.
- Cached requests reserve full original-position prompt KV, load/align once,
  and retain one global selection across steps. Never use sparse-slot remapping.
- Empty repair ranges advance without forward/sampling; sample only after the
  full prompt. Reject preemption. Clear request state on finish/cancel/retirement.
- Probe the saved full suffix in at most 16384-query/projection batches, with
  all-context normalization and query-weighted means. Keep repaired KV intact.
- ProphetKV averages all 64 layers. Selective averages explicit zero-based IDs,
  or the last N layers; sum in ascending FP32 order, then divide by layer count.
- Preserve TP averaging, all-context-key normalization, stable position ties,
  exact floor budgets, original-position causal fusion and all-layer alignment.
- Context chunks default to 4096; setup accepts multiples of 64 through 16384.
  Keep the whole question in a fresh
  suffix of at least 256 tokens; never silently truncate inputs.
- Preserve cache readiness, all-rank selection replay and request retirement.
- `setup` freezes a JSONL collection and builds independent KV once. `run --setup`
  inherits model/TP/preparation and runs all prompts, or `--prompt-ids`, sequentially.
- Keep setup hashes separate from execution UUIDs. Algorithm/ratio/layers/output
  cap do not change KV identity; model/tokenizer/tokens/layout/runtime changes do.
- Exclusive build lock, shared reader locks, every-rank committed shards and
  retirement are mandatory. Resume only incomplete chunks; preserve complete shards.
- Cached collection readers never populate, write warmup KV, delete KV or silently
  rebuild. Baseline creates no connector and never looks up KV. Legacy `--input`
  retains its temporary cache. Setup completion is published atomically.
- Persistent setup GPU validation is pending assigned devices. The opt-in matrix
  is `tests/gpu_persistent_setup.py`; CPU tests use fake engines and local fixtures.
- Collection runs publish live aggregation after validated/retired prompts and
  final aggregation only after every selected prompt succeeds and engine shutdown.
  Report per-subtask and prompt-weighted overall TTFT, accuracy, thinking/answer
  content-token counts; keep control tokens and scored/unscored denominators explicit.
- Freeze per-prompt subtask/references/scoring as evaluation metadata; references
  never enter inference. Score only decoded answer content, excluding thinking.
  Missing references mean N/A; unfinished thinking has no answer content.
- Run CPU tests with `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v`.
- GPU execution needs a user request and explicit UUIDs. Never start, stop or
  resume historical experiments from other branches/worktrees; never use dummy jobs.
- Keep data, weights, caches, logs and results out of Git. See README for commands.
- vLLM 0.9.2 UUID compatibility is installed by `apply_all_patches` before
  attention backend imports, in drivers and spawned workers. Keep UUID visibility
  unchanged; NVML resolves UUIDs to physical indices only for metadata queries.
  Do not rely on manually modified site-packages or replace UUIDs with ordinals.

- 2026-09-26 focused GPU check is complete: two exact 64000-token NIAH single-3
  prompts × three modes, six validated measurements, all UUIDs correct. Both
  ProphetKV modes used 20%; selective layers [45,48,50,56,58]. User approved a
  test-only 64256-token/1004-block allocation on GPUs1–4; YaRN4 table131072 and
  scheduled budget16384 unchanged. Greedy non-thinking256. All engines exited
  and caches retired. See `outputs/niah-single3-64000-2samples-20260926/` for
  comparison and final-validation.json. Do not resume the completed driver.
  The default full-window configuration remains unchanged and GPU-unvalidated.

- Remote all-503 LongBench v2 commands: `scripts/A800_LONGBENCH.md` and
  `scripts/a800_longbench.sh`. No local GPU launch. Two UUID-pinned TP4 groups split
  even/odd prompt rows, each running baseline then all 12 cached configurations.
  Use `run.sh sweep`: build ONE prompt's KV, wait for all TP shards, reuse across
  vanilla/selective 1/5/10/15/20/30%, retire and delete before the next prompt.
  No collection KV setup in this launcher. One baseline engine plus one cached
  engine per group; policy changes require quiescence. Read requests cannot dump.
  Thinking/output16K, prefill16K, chunk4K, selective zero-based layers11–15 and
  preserved middle truncation to formatted input<=114688 remain unchanged.
  At most55.88GiB total retained KV, plus write/scratch/results overhead.
  Live reports serialize both writers; final requires503 validations/method and
  both engines exited. CPU tests cover bounded deletion and dynamic policies;
  `tests/gpu_temporary_sweep.py` is opt-in and has not been run on remote GPUs.
  Existing persistent setups and historical experiments must remain untouched.

- Remote deployment uses Git fetch and a separate server worktree, with existing
  data at `/mnt/sde/jh/projects/unified-cache-management/.data/LongBench-v2/data.json`.
  User explicitly wants no archive transfers. Prepare tokens offline with the
  local Qwen3 model/tokenizer; see the A800 guide for exact paths and commands.

- L20 RULER commands: `scripts/L20_RULER_64000.md`, `scripts/l20_ruler.sh`.
  User already installed the environment and cloned the clean branch at
  `/data/jh/unified-cache-management/ucm`; model `/data/jh/ckpts/Qwen3-32B`,
  Python `/data/jh/envs/ucm/bin/python`. Git deployment and official downloads only.
  Eight tasks x100: cwe/fwe/vt/qa_1/qa_2/niah_multivalue/niah_multikey_2/3.
  Exactly64000 formatted input tokens (not65536), native thinking/output16384,
  BF16 YaRN4, chunk4096, scheduled prefill16384. Baseline plus vanilla/selective
  1/5/10/15/20/30/40/60/80%, selected zero-based layers11–15:15200 measurements.
  Pinned NVIDIA generators retain source tokens; documented newline padding before
  user text attains exact length. No forced assistant prefix or truncation.
  Two UUID-pinned TP4 groups on remote0–3/4–7;8–9 stay free. Build each prompt once,
  reuse across18 cached policies, then retire/delete. No local GPU launch.
  Per-method live/final and combined status/aggregate report TTFT, answer-only RULER
  accuracy, thinking/answer content-token lengths per task and overall.

- A800/L20 mid-run scope change: `resume` on the existing launchers stops scheduling
  selective and runs only missing vanilla ratios; baselines must be complete.
  See `scripts/RESUME_PROPHETKV.md`. Never delete completed records or restart
  baseline. Stop only the exact output-owned process tree before updating code.
  Resume validates original fingerprints, all-rank diagnostics, retirement and
  retained hashes; keep original reports and raw artifacts unchanged. New report
  state/session receipts live in `continuation/attempt-*`, selected by
  `continuation.json`. Final scope is3521 A800/8000 L20, selective discontinued.
  CPU-only recovery/locking/stop tests passed; remote GPU continuation is pending.

- Server-local launcher settings live in Git-ignored `.env`; `.env.example` is
  tracked. `scripts/server_env.sh` loads one trusted Bash assignment per line,
  preserving existing environment values before expanding later assignments.
  Both launchers load it before defaults. Optional `UCM_ENV_FILE` selects another
  file. Keep existing run/prepared/cache paths and UUID groups for a continuation.
  See `scripts/SERVER_ENV.md` for fetch-before-stop/update/resume commands.

- Resume first prints saved-file counts per method/shard and complete vanilla
  prompts; `counts` is read-only and does not load token files or diagnostics.
  `RESUME_VALIDATION=fast` skips saved diagnostic replay/hashing, retains result
  metadata/hash/input/config/retirement checks and requires diagnostic presence.
  Record the mode in receipts/reports; existing diagnostic pins are preserved
  but not verified in fast mode. Every NEW measurement remains fully validated.
  Full remains the absent-variable default; .env.example explicitly selects fast.
  Accept the pinned 16c1f75/2ae7f30 continuation runtime on upgrade, preserving
  accepted cohort fingerprints for repeated resumes. Unknown runtime drift fails.
- Historical remote fingerprints may include untracked `ucm/vendor/RULER` and
  `ucm/.cache/vendor/RULER` directories (both can coexist). Preserve both in the
  checkout. Resume can reconstruct a pinned release plus current vendor hashes,
  accepting only an exact original fingerprint match. Never remove vendor hashes
  or rewrite original receipts to force compatibility. Keep those files unchanged.
