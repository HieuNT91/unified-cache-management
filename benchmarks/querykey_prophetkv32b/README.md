# Query-key attention ablation

Independent, immutable 15-prompt MK2/MK3/CWE cohort (rows 0–4/task), 210 fresh
focused measurements across all64/selected5 and budgets 5/10/20/30/40/50/60%.
Reuse validated full-question controls and the latest vanilla baseline; run only
missing controls. Final comparison has 420 method records plus 15 baselines.

`prepare.py prepare` copies exact inputs and verifies tokenizer offset mappings
into all frozen token IDs. `spans/` sidecars identify only final-question spans:
MK2 full key phrase, MK3 full hyphenated UUID, CWE “10 most common words” (a task
phrase, not an entity key). Answers/evidence never enter inference selection.
Private query_scope/layer_scope configuration changes only scoring query rows;
FP32 layer sum/64 for all64, five-layer sum for selected5 and native TP reduction
are retained, with normalization over all cached keys. Sequential selected-layer
stopping, all-layer alignment, floor budgets, ascending ties and fresh256 remain.

Qwen3-32B BF16 TP4 eager, YaRN2 original65536 table entries unchanged, allocation
65920,1031 KV blocks, chunk/memory tile4096, non-thinking greedy256. Only physical
GPUs1–4 by UUID; GPU0/dummy jobs forbidden. No separate model qualification.

Run CPU tests before transition. `transition.py` verifies current ownership,
stops the authorized vanilla supervisor/reporter, verifies all owned engines
exit, preserves accepted/incomplete artifacts separately and checks hashes.
`prepare.py freeze` freezes the exact retained inventory and missing schedule;
`prepare.py pin` pins all implementation/input/control files. Never rerun a
historical scheduler. Original accepted artifacts remain at their original paths.

The locked detached supervisor processes ascending budgets: focus-all64,
focus-selected5, then any missing full-question controls, each with one persistent
engine and row-major MK2/MK3/CWE order. Cache readiness/hits, TP4 committed warmup
files, exact per-rank score replay, every-layer selected sets, retirement, bounded
caches, a900-second watchdog and one retry are mandatory. Ordinary priming captures
per-layer focused scores; observer is removed before timed generation. Full
controls match historical scores/masks; focus is expected to differ.

CPU-only final reporting waits for every required fresh validation and owned
engine exit. It checks all retained hashes and publishes TXT/HTML/CSV/JSON,
per-prompt exact masks and PNG/PDF diagnostics. Reports retain actual lengths,
output cap counts, predictions/token IDs, all separate durations, evidence mass,
key/value/harmonic recall, complete statements, CWE reference-word occurrence
coverage, paired mask overlap and answer changes. Timing cohorts and retained vs
fresh measurements are explicit. Conclusions are descriptive for five prompts per
task from the same cohort used to select five layers; CWE needs distractor counts
too, so its annotated reference-word coverage is incomplete evidence.

Result root: `.results/querykey-prophetkv32b-mk2-mk3-cwe-5samples-output256-20260924/`.
Before action read progress, supervisor, reporter, active state, launch/detachment
receipts and the active session log. Never duplicate supervisor/reporter.

```
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python benchmarks/querykey_prophetkv32b/run.py detach
```

`run.py status` is read-only. Verify startup once, then status only on request.
