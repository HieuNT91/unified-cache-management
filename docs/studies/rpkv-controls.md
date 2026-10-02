# rpkv GPU control studies

Historical scope and evidence recorded on 2026-10-01; reorganized on 2026-10-02.
Live status must come from each package's receipts, not present-tense wording
below. These runs use their frozen implementations with independent diagnostic
probes. New fixed controls use answer diagnostics with zero independent probes;
see [experiment rules](../agents/experiment-rules.md). The user's instruction was
to leave the expanded run unchanged when removing probes from current source.

See [testing](../agents/testing.md) for the separate GPU edge-case matrix and
[RULER](../protocols/ruler.md) / [LongBench v2](../protocols/longbench-v2.md) for
current input and evaluation contracts. Paths below are worktree-root-relative.

## User-requested GPU controls (2026-10-01)


Package: `outputs/rpkv-gpu-validation-20261001/`. Source is snapshotted in
`frozen-code/`; do not mutate an active frozen runtime or duplicate supervisors.
The native control driver is `scripts/rpkv_gpu_validation.py`. There are 210
scheduled answers: original RULER13 ×5 and five LongBench v2 inputs, each with
connector-free no-cache, original all64 ProphetKV1%, and ProphetKV5%. Each sparse
pair has an independent diagnostic one-token probe; 70 probes total, separate
from the answer count and answer TTFT. Sparse controls run before baselines.

Physical GPUs1–4 are UUID-pinned RTX4500Ada24GB cards. The local allocation is
KV65920/1031 blocks with native YaRN4, BF16 TP4, chunk4096 and prefill16384.
RULER uses five-row upstream task batches, seed42, non-thinking and original task
caps. LongBench retains native thinking and16384 output tokens. The user explicitly
approved selecting from full-text inputs whose complete prompt plus output reserve
fits this allocation, after three initial random selections exceeded memory.
All503 rows were tokenized;158 fit. Random seed42 selects source rows87,25,238,220,
202, with10243–48397 input tokens and no truncation. Selection receipts preserve
the eligible pool, all lengths, source identity and the initial unused selection.
This is a constrained five-input LongBench comparison, not a full503-row result.

`results/{ruler,longbench-v2}/report.json`, `report.md` and `report.csv` update after
validated prompts. Separate initialization/construction/readiness/priming receipts,
raw predictions/token IDs, per-rank diagnostics, immutable-cache checks, retirement
and process-exit receipts are retained. The CPU sequence starts LongBench only
after the RULER-owned GPU group exits, then independently validates all210 answers
and70 probes and writes `final/report.md`, `report.json`, `measurements.csv` and
`validation.json`. RULER is rescored with the pinned original evaluator functions;
all generated thinking/answer/control counts are independently replayed.

Do not infer completion from a launch or startup receipt. Only `final/validation.json`
and the accepted records establish completed evidence. The separate100% repair,
dense fallback and synthetic GPU edge-case matrix in the testing guide remains outside these
requested three measured methods.


The initial GPU attempt halted while priming original `niah_single_1-000`:
PcStore flattened repeated file hashes into an oversized read. Its frozen code,
logs,14 accepted answers and7 probes are preserved unchanged under
`startup-history/duplicate-cache-load/` and excluded from the final comparison.
`TrackedStore` now submits batches with unique file hashes, retaining all separate
destination slots and tracking each backend task through retirement. The real
Python/native pointer-flattening regression and failed-transfer retirement tests
pass; the full CPU suite passes232 tests. The corrected GPU run starts with the
same failing input:16 context chunks, including repeated content. Both scheduled
1% and5% answers and the independent probe passed all-rank replay, immutable-cache,
YaRN alignment and retirement checks. See `results/ruler/startup-verification.json`.
The same frozen70 inputs are used in the corrected full comparison.

### Completed validation

All210 answers and70 independent probes passed the final audit, including
identical input hashes, all-rank replay, cache immutability, retirement and
independent scoring. All owned GPU engines exited. The final evidence is in
`outputs/rpkv-gpu-validation-20261001/final/validation.json`; per-task results
and raw measurement columns are in `final/report.md` and `final/measurements.csv`.

| Dataset | Method | N | Accuracy % | Mean TTFT s | Mean thinking tokens | Mean answer tokens |
|---|---|---:|---:|---:|---:|---:|
| RULER | No cache | 65 | 86.13 | 29.175 | 0 | 27.43 |
| RULER | ProphetKV 1% | 65 | 74.18 | 2.119 | 0 | 28.26 |
| RULER | ProphetKV 5% | 65 | 76.26 | 3.553 | 0 | 27.69 |
| LongBench v2 | No cache | 5 | 40.00 | 9.041 | 2623.4 | 168.2 |
| LongBench v2 | ProphetKV 1% | 5 | 40.00 | 0.909 | 1336.2 | 170.2 |
| LongBench v2 | ProphetKV 5% | 5 | 40.00 | 1.232 | 2163.6 | 99.6 |

TTFT includes request submission and TP action synchronization, but excludes
separate cache construction, readiness, ordinary priming and independent probes.
Four RULER outputs per method reached their task cap; no LongBench output reached
the16K cap. These small-cohort results do not establish general accuracy or
validate the separate full-repair/dense-fallback GPU matrix. Data, caches and
result artifacts remain local and are not included in the Git commit.

## Expanded controls: 40/task and 20 LongBench inputs

The separately authorized package is
`outputs/rpkv-ruler40-longbench20-20261001/`. It contains fresh measurements for
520 RULER inputs (13 original40-row batches, seed42) and20 seed42 random
LongBench inputs from the approved full-text fitting pool. The three methods
remain no-cache, all64 ProphetKV1% and5%:1620 answers plus540 independent probes.
The completed five-sample comparison is preserved separately; its measurements
are not counted toward this run.

`experiment.json` freezes the scope and source identity; each dataset protocol
pins it, the row inventory, prepared files and runtime. The reporter derives
expected counts from this manifest and rejects missing or duplicate rows.
`scripts/rpkv_longbench_inputs.py` records all503 full formatted lengths before
sampling, retains full text, and keeps the native16384 output cap. Reports use
the same TTFT definition, original RULER scoring and thinking/answer token counts.

One detached CPU supervisor runs RULER; a separate detached CPU sequence waits
for its owned GPU processes to exit, then runs LongBench and the independent
final audit. The package contains a read-only frozen source snapshot, launch
and detachment receipts, live per-dataset reports and automatic final reports.
Never duplicate a supervisor or restart the completed earlier scope. New
reporting/cohort checks and the existing CPU regression suite pass234 tests.
