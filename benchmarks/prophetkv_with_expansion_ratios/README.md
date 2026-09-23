# Expansion ratio extension

Run 600 new measurements on the parent's exact 200 frozen prompts: total ratios
30%, 40%, 50% with explicitly requested anchor ratios 22.5%, 30%, 37.5%. Other
selector parameters, probing and recomputation are unchanged. The private UCM
runtime is copied byte-for-byte from the completed parent experiment.

Retain all 400 baseline/20% measurements without rerunning or modifying them.
The combined report contains 1000 measurements across five configurations.
Preserved input bytes, reference answers, model/runtime hashes, 128-token cap,
physical GPU assignments, native RoPE, BF16/TP1/eager execution, 4096-token chunks,
256-token fresh suffix and 65792-token allocation remain unchanged. No prompt
exceeds 65536 tokens and no prompt is truncated.

GPUs 1–4 only, pinned by UUID. GPU 0 and dummy jobs are forbidden. New method
order rotates across GPUs, with one persistent engine per configuration per GPU.
No separate model qualification/smoke runs. Normal initialization, priming,
cache readiness/hit evidence, all-layer selected-set checks, saved-score reference
validation, TTFT semantics, request retirement, bounded caches, watchdog and one
retry remain enabled. Baseline and 20% timings come from the earlier cohort;
the final report explicitly states this comparison limitation.

Driver: `bash benchmarks/prophetkv_with_expansion_ratios/run.sh prepare|verify|detach|status`.
Check live process identities and current logs before launching; never duplicate
a supervisor/reporter or resume a historical driver. Selector checks are CPU-only
with `checks.py`; `checks.py --cuda` requires only physical GPU 1 visible by UUID
and performs selector parity checks, not model inference.

Results: `.results/prophetkv-with-expansion-ratios-ruler4x50-20260923/`.
The locked detached CPU reporter writes `prophetkv_with_expansion_ratios_results.txt`
and `final/` raw records, CSVs and validation after all 600 new records validate and
owned engines exit. The original report and all parent artifacts remain unchanged.
`preserved-parent.json` pins the retained artifacts; the frozen protocol records
the exact total/anchor pairs, inherited prompt identities and method phase orders.
