# Oracle target recomputation, then high-budget ProphetKV

One locked detached supervisor and one CPU reporter run two experiments in order.
Root: `.results/oracle-highbudget-prophetkv32b-20260925/`.

1. `oracle/`:30 exact prompts, ten rows0–9 each MK1/MK2/MK3. Target-only once,
   then target-union-all64 and target-union-selected5 at10/30/50/70/90%:330 runs.
   The union is all target key/value tokens plus the original ProphetKV top
   floor(budget×eligible) positions. Overlaps are deduplicated; the actual budget
   can exceed the nominal budget and is always reported. CWE is excluded.
   Target-only skips attention probing and recomputes only eligible target tokens
   plus the usual fresh suffix. It is not repeated for identical layer scopes.
   One MK2 row8 target lies in the exact prefix: its10 target tokens remain reused,
   as explicitly chosen by the user, and are reported as already exact.
2. `highbudget/`:150 prompts, fifty rows0–49 each MK2/MK3/CWE. Original full-question
   all64 and selected5 scoring at90/80/95%, in that listed order:900 fresh runs.
   This experiment starts automatically only after all330 first-experiment results
   validate, owned engines exit and the first final report hashes verify.

Both use original full-question rows and selected layers [45,48,50,56,58]. All64
retains native FP32 sum/64; selected5 native FP32 sum; TP sum/4 is unchanged.
Original suffix probing/stopping, full-key softmax, exact floors/ascending ties,
all64 cache alignment and all-layer selected-set verification remain.

The first experiment is explicitly oracle-assisted: ground-truth references
identify key/value spans in the original context. These are diagnostic results,
not deployable retrieval scores. Sidecar token mappings preserve every input byte.
The second experiment never passes oracle positions to inference. All requested
prompts are copied from the original32B50-row cohort; no truncation. No new baseline
or qualification measurements. All1230 requested measurements are fresh.

Qwen3-32B BF16 TP4 eager, GPUs1–4 pinned by UUID, YaRN2 with original65536 table
entries unchanged and allocation65920,1031 KV blocks, chunk/memory tile4096,
fresh256, non-thinking greedy256. GPU0 and dummy jobs are forbidden.

Persistent engine per configuration. Retain cache readiness/hits, eight committed
warmup TP4 shards, per-rank saved-score/native-mask/union replay, per-layer selected
sets, retirement barriers, bounded caches,900-second watchdog and one retry.
Ordinary priming captures per-layer scores and compares available prior captures;
observers are removed before timed generation. New rows lack prior all-layer
captures and instead pass normal priming/measured exact-score/mask equivalence.
Target-only validates its exact oracle mask without scoring. No separate smoke.

Each stage publishes TXT/HTML/CSV/JSON and PNG/PDF only after its full validation
and owned engine exit. Reports include accuracy, paired differences, mean/median
TTFT and matched speedups, actual repair counts, oracle overlap/additions, raw
predictions/token IDs, prompt lengths, output caps and all separate durations.
Previously existing results are hash-pinned and preserved. Timing cohorts differ;
the first five rows/task overlap the cohort used to choose the selected layers.

Read root and stage progress/supervisor/reporter/active state, launch/detachment
receipts and current logs before any action. Never duplicate processes or restart
historical jobs. Previous query-key run was cancelled at87/264; cancellation and
accepted/incomplete preservation are in `.analysis/oracle-highbudget-transition-20260925/`.

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python benchmarks/oracle_highbudget_prophetkv32b/run.py detach
```

`run.py status` is read-only. Verify startup once; status checks only on request.
