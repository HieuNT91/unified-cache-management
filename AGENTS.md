# Agent instructions

- Scope: Qwen3-32B BF16, YaRN 4× (32768 → 131072), TP configurable (default 4).
- Exactly three public modes: `baseline`, `prophetkv`, `selective_prophetkv`.
- Chunked prefill and prefix caching stay disabled. Baseline has no UCM connector.
- ProphetKV averages all 64 layers. Selective averages explicit zero-based IDs,
  or the last N layers; sum in ascending FP32 order, then divide by layer count.
- Preserve TP averaging, all-context-key normalization, stable position ties,
  exact floor budgets, original-position causal fusion and all-layer alignment.
- Context chunks are at most 4096 tokens. Keep the whole question in a fresh
  suffix of at least 256 tokens; never silently truncate inputs.
- Preserve cache readiness, all-rank selection replay and request retirement.
- Run CPU tests with `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v`.
- GPU execution needs a user request and explicit UUIDs. Never start, stop or
  resume historical experiments from other branches/worktrees; never use dummy jobs.
- Keep data, weights, caches, logs and results out of Git. See README for commands.
