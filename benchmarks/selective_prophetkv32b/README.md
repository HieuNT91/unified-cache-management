# Selective ProphetKV on the 40-prompt selection cohort

Independent 360-measurement comparison: baseline then selective ProphetKV at
5/10/20/30/40/50/60/1%. Five source rows each of MK2, MK3, MK1, CWE, VT,
QA1, QA2 and single3; row-major task interleaving is identical for every method.
Exact input bytes and references are copied from the completed layer study.
This evaluates the same cohort used to choose layers, not held-out performance.

Scoring layers (zero-based): [45,48,50,56,58]. Native FP32 layer sums in ascending
order, existing TP all-reduce and head averaging, all-context-key normalization.
Suffix projection runs through layer 58; suffix forward propagation ends at 57.
All 64 cached layers are aligned before recomputation. Original ProphetKV remains
selectable in the private method implementation. Exact floor budgets and ascending
position ties; prefix reuse and fresh256 are unchanged.

Qwen3-32B BF16 TP4 eager, YaRN2, native non-thinking, greedy256, chunk4096,
allocation65920, 1031 KV blocks. The RoPE extension verifies every original entry
and the unchanged-frequency formula. Memory tiling and other cache safeguards
are retained. No separate qualification. Ordinary priming compares five-layer
attention against prior captures, then removes the observer before timing.
Saved scores and selected sets are exported after generation. Every rank/layer,
cache hit, score replay and retirement is checked before accepting a record.

Physical GPUs1–4 by UUID only; GPU0 and dummy jobs forbidden. One persistent
engine per configuration, sequentially, locked CPU supervisor, 900-second
accepted-measurement watchdog and one retry. CPU reporter requires all360
validations and engine exit before publishing final TXT/HTML/CSV/JSON reports.
Read live progress/supervisor/reporter/active and current logs before acting.
Never duplicate a supervisor or reporter or start a historical scheduler.

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python /home/thnguyen/spark/unified-cache-management/benchmarks/selective_prophetkv32b/run.py detach
```

Use `status` instead of `detach` for requested status checks. Result root:
`.results/selective-prophetkv32b-40samples-output256-20260924/`.
`implementation.json` freezes sources and inputs; `preserved-prior.json` pins
the completed study. Final output is `final/index.html`, `final/results.txt`,
`final/summary.csv`, `final/paired.csv`, `final/records.json`, and arithmetic
comparison tables. `final-validation.json` marks complete acceptance.
Startup verification only; no ongoing assistant polling afterward.
