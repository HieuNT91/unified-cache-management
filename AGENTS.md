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
