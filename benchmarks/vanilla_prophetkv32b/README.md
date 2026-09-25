# Vanilla ProphetKV matched comparison

360 fresh measurements: baseline plus ProphetKV5/10/20/30/40/50/60/1%, with
one persistent engine per configuration in that order. Exact same 40 prompts,
references and row-major task-interleaved order as the completed selective run.
All previous results and sources are hash-pinned and preserved.

Only the scoring method differs: all64 layers contribute equally. Sum FP32
question/head means in ascending layer order, divide by64, then use the existing
TP all-reduce and divide by4. Division by64 is exact power-of-two scaling for
these scores, preserving native all-layer sum rankings. Scores normalize over
all cached keys. Exact floor budgets and ascending-position ties remain.
Original all64-layer suffix probe computation and all-layer alignment remain.

Qwen3-32B BF16 TP4 eager, YaRN2, original-frequency RoPE table extended to65920,
1031 KV blocks, memory tiling4096, chunks4096, fresh256, non-thinking greedy256.
GPUs1–4 by UUID only; GPU0 and dummy workloads forbidden. No separate model
qualification. Ordinary unmeasured priming compares all64 layer scores with
historical captures; observer removed before timed generation. Exact native
mask replay, identical all-rank scores/masks, all64-layer selected-set checks,
cache readiness/hits, TP4 warmup commit checks and retirement retained.

CPU supervisor and reporter use locks, nohup and independent sessions. Progress
watchdog900 seconds and one retry; bounded caches removed only after retirement
or owned engine exit. Final TXT/HTML/CSV/JSON reporting requires360 validated
measurements and owned-engine exit. Includes direct paired comparisons with
selective results and fresh-versus-prior baseline. Timing cohorts differ; this
is the same small cohort used to choose the selective layers.

Result root: `.results/vanilla-prophetkv32b-40samples-output256-20260924/`.
Read progress/supervisor/reporter/active, launch/detachment receipts and current
session log before any action. Never duplicate a supervisor/reporter.

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python /home/thnguyen/spark/unified-cache-management/benchmarks/vanilla_prophetkv32b/run.py detach
```

Use `status` for requested status checks. Startup verification only; no ongoing
assistant polling afterward. Final outputs: `final/index.html`, `final/results.txt`,
`final/summary.csv`, `final/records.json`, `final/versus_selective.csv`,
`final/versus_selective_paired.csv`, and `final-validation.json`.
