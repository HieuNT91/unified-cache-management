# ProphetKV with expansion

Compare baseline and `prophetkv_with_expansion` on 200 distinct prompts (50 each
of niah_multikey_2, niah_multikey_3, cwe, qa_1). The first 50 source rows whose
fully formatted prompts fit **at most 65,536 tokens** are frozen without truncation.
Both methods use identical inputs and GPU assignments, with method phase order
counterbalanced across physical GPUs 1–4. GPU 0 and dummy jobs are forbidden.

The selector retains floor(N * .15) stable top-ranked anchors and selects exactly
floor(N * .20) tokens. Consecutive anchors with at most two intervening unselected
positions group transitively. Segment scores sum only anchor scores, in ascending
position order using float64, divided by sqrt(anchor count). Segments rank by score
descending, then leftmost position. Each proposes a right window immediately after
its final anchor: clamp(round(8 * sqrt(anchor count)), 8, 64), using Python's
ties-to-even rounding. Greedy expansion deduplicates visits and stops at the total
budget. Internal gaps are left for stable score-ranked fallback. Internal chunk
boundaries do not clip windows; the eligible region's end does.

All eight parameters are configurable through `ucm_sparse_config.ProphetKV` with
`method: prophetkv_with_expansion` and the `expansion` mapping. The exact method
name also has a factory registration. Changing `total_ratio` from .20 requires an
explicit compatible `anchor_ratio`; it is never rescaled. Original `prophetkv`
keeps its existing selector. CUDA uses sequential float64 segment accumulation,
stable ranking, shared normalization/window lookup values, and integer first-visit
priorities. Scores and masks stay on the GPU until post-generation export.

Results: `.results/prophetkv-with-expansion-ruler4x50-20260923/`.

Run `bash benchmarks/prophetkv_with_expansion/run.sh status` for progress.
`prepare` freezes the runtime and protocol after CPU input preparation with
`CUDA_VISIBLE_DEVICES='' <runtime-python> benchmarks/prophetkv_with_expansion/data.py prepare`.
`verify` checks immutable artifacts. `detach` requires CPU/CUDA selector validation,
uses locks, and refuses live supervisors/reporters. Check current process identities,
state and logs before any resume. The smoke command is disabled. The supervisor
runs eight normal persistent-engine sessions, retries once, and uses a 900-second
validated-result watchdog. Only missing/invalid measurements are scheduled.

The CPU reporter writes `prophetkv_with_expansion_results.txt` and final raw records,
CSV tables and validation only after 400 accepted measurements and verified owned
engine exit. Cache readiness, measured hits, every layer's selected set, reference
mask equality, and request retirement remain mandatory. TTFT includes probing,
selection, alignment/transfer and recomputation. Model loading, cache construction,
readiness, priming and post-generation export are excluded and recorded separately.

No old scheduler is imported or resumed. Historical experiment artifacts and
installed packages are preserved. Read `transition/` for the previous job's stop
and artifact-preservation receipts once the authorized transition occurs.
