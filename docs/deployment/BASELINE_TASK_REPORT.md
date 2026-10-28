# Task reports with copied no-cache results

Run `python scripts/experiment_task_breakdown.py EXPERIMENT_DIR` for either a
RULER or LongBench v2 experiment. The tree-prophetkv report automatically reads
`EXPERIMENT_DIR/baseline-source/` when present. It requires no env file, model,
prepared inputs or GPU, writes no reports, and may run alongside collection.

The copied directory preserves the original run structure:

```text
baseline-source/
  settings.json
  plan.json
  rows.json
  ruler/                         # primary/ for LongBench v2
    protocol.json
    records/nocache/...           # complete result folders, including receipts
    sessions/*/initialization.json
```

The baseline is validated under its original protocol, input hashes, seeds,
GPU assignments and initialization evidence. No receipt is rewritten for the
new tree run. The source rows must match each target prompt's prepared JSON hash,
task, ordinal, evaluation protocol, thinking policy, references and scoring.
Committed token hashes must agree across methods for the same prompt/seed.
Paths to the original model and prepared files may be unavailable on the reader.

For the A800 two-seed tree run, only baseline seeds0 and1 are read from the
original four-seed run. Seeds2 and3 remain untouched and do not affect metrics.
Missing requested seeds in the source schedule, mismatched inputs/policies,
corrupt results or missing initialization evidence cause a clear error.

With a copied baseline, both report scripts automatically use the exact
intersection of completed prompt/seed pairs across no-cache and every scheduled
tree percentage. The task tables show identical sample membership for each
method, with accuracy, TTFT, thinking tokens and answer tokens. Counts above the
tables show all available answers for the selected seeds; table counts show the
shared subset. Empty intersections report zero samples and N/A, not zero accuracy.
LongBench tasks use the prepared domain/subdomain labels.

If `baseline-source/` is absent, the tree report explicitly says it is reporting
only tree methods. The regular breakdown retains its all-available behavior in
that case; `experiment_same_count.py EXPERIMENT_DIR` always uses the intersection.
A copied baseline with missing metadata or corrupt committed records is an error
rather than a silent fallback; missing answers are allowed.

The copied RULER baseline comes from L40 and the tree answers from L20. Their
original hardware provenance is retained; TTFT comparisons are cross-server
measurements, not a measurement of method speedup on the same hardware.

On L20:

```bash
/data/jh/envs/ucm/bin/python \
  /data/jh/unified-cache-management/.worktrees/tree-prophetkv-l20-v1/scripts/experiment_task_breakdown.py \
  /data/jh/unified-cache-management/outputs/ruler-l20-tp2-thinking-tree-prophetkv-500-v1
```

On A800:

```bash
/mnt/sde/jh/envs/ucm/bin/python \
  /mnt/sde/jh/projects/unified-cache-management/.worktrees/tree-prophetkv-a800-v1/scripts/experiment_task_breakdown.py \
  /mnt/sde/jh/projects/unified-cache-management/.results/longbench-v2-a800-tp2-tree-prophetkv-2seed-v1
```

CPU tests cover both command-line entry points, both datasets, real accepted
receipts, L40/L20 group differences, four-to-two seed selection, exact sample
intersection, unchanged source files, input/policy/token mismatches, corrupted
results and missing initialization evidence. No remote run or GPU execution is
part of this report update.

## Export tree outcomes to NPZ

From the updated checkout, run `python scripts/export_features.py EXPERIMENT_DIR`.
For a tree-prophetkv experiment this writes
`EXPERIMENT_DIR/tree-prophetkv-outcomes.npz`, with six one-dimensional arrays:
`prompt_id` (`ID--seed-N`), `method`, `accuracy` (0–1), `ttft` (seconds),
`thinking_tokens` and `answer_tokens` (int64). Token counts are copied from
accepted result metrics for both tree and no-cache; they are not retokenized or
inferred from text. Missing or invalid token counts are rejected.
No model, prepared inputs, attention captures or feature extraction are needed.

If `baseline-source/` exists, its original accepted no-cache records are included
after the same input/policy/token checks as the report. Only the target run's
seeds are included (A800 seeds 0–1, even if the baseline has four). Without that
folder, the exporter explicitly reports that it exported tree methods only.
All currently committed answers are exported independently for each method.
The experiment may still be running, methods may have different counts, and
no intersection or complete-cohort requirement is applied. Missing answers and
in-flight answers without a commit receipt are omitted; an empty collection
produces empty arrays. Corrupt committed records or mismatched input identities
still fail before writing the NPZ. The export prints available counts per method.
Existing output files are preserved; use `--output NEW_PATH.npz` for a new export.

This tree export uses repository readers, so deploy the complete checkout,
including `scripts/export_tree_outcomes.py`; copying only `export_features.py`
is insufficient. Legacy non-tree export behavior is unchanged by this addition.
