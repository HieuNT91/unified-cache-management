# Offline deeper-router search

Run from this clean worktree:

```bash
PYTHON_BIN=/home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python
"$PYTHON_BIN" scripts/deeper_router_study.py all
"$PYTHON_BIN" scripts/verify_deeper_router_study.py
"$PYTHON_BIN" scripts/deeper_router_results.py
"$PYTHON_BIN" scripts/verify_deeper_router_results.py
```

Default output: `outputs/ruler13-deeper-router-study-20260930/`. `--output` selects
another directory. `prepare`, `search`, `report`, and `resume` stages are available.
The entry point disables CUDA and fixes BLAS/OMP/MKL threads to one. Four CPU worker
processes search independent feature subsets. An exclusive process lock prevents
duplicate writers. No inference, integration, deployment, or historical scheduler
is invoked.

The frozen 260-prompt six-action outcome matrix and all 45 previously extracted
attention features are reused. All prompts are used for fitting, adaptive selection,
and scoring. These are training-set findings. Depths 1–5 yield 4,320 settings per
subset with unchanged objectives, loss modes, weights, penalties, minimum leaf
sizes, ordered CART arithmetic, and threshold tie handling.

Controls include the original five features, the incumbent seven features, and
all 45 features. Forty single additions are ranked across all depths. The strongest
eight produce all 28 pairs; six triples extend the best pair. Winning added features
receive drop-one full-grid refits. Drop-one refits are one-round attribution diagnostics of the selected winner,
excluded from the primary-search stage as in the original study. The finalizer
then ranks ALL evaluated candidates, including these bounded diagnostic refits,
for the final per-depth and overall winners. It does not recursively eliminate
features. `finalized/drop-one-ablations.csv` explicitly names the primary-search
winner as the ablation anchor. The incumbent is retained throughout.

Reuse requires the original matrix, feature columns, fitter implementation, and
settings to match exactly. Completed expanded subsets from the preserved first pass are reused verbatim
with provenance and accounting; that pass was stopped to bound the ablation stage.
Completed depth-1–3 candidates are reindexed by setting
identity into the expanded grid. All candidate metrics are recomputed. The selected
tree for each depth/subset is refitted to recover its rules and must reproduce the
saved decisions exactly. New subsets without a matching old search run all settings.
Completed extraction and search units have immutable digest envelopes; resume
checks source and implementation hashes and preserves these files and their mtimes.

Primary TTFT remains selected answer TTFT plus the frozen traversal allowance,
charging saved probe TTFT only for dense. All exported policies receive separate
traversal benchmarks and incremental resident-array feature benchmarks (two prompts
per task, five repetitions each). Adjusted estimates replace the frozen traversal
allowance with measured traversal and add measured incremental feature work.
Feature measurements exclude archive I/O and aggregation already needed for the
original features. Live transfer, synchronization and switching remain unmeasured.

The final deliverables are `finalized/report.txt` and `finalized/report.json`;
they compare per-depth and overall winners with the
8.605654-second incumbent and 3.650979-second hindsight diagnostic. “Depth” means
the setting's maximum depth; actual depth and leaves are also reported. The overall
accuracy floor is computed from dense minus 2 percentage points (89.5769231%).
`finalized/per-task.csv` reports separate task losses; the constraint is not per-task.
`finalized/decision-changes.csv` includes newly selected 1% decisions. `finalized/trees/` contains rules
and exports with a deliberately offline-only schema. The primary-stage `trees/`
also contains all drop-one ablation exports. `finalized/complete.json` pins the
finalizer, its source study and its outputs; repeated finalization only validates
existing artifacts and does not overwrite them.

Verification independently reads all 1,560 accepted answers and 260 probes,
recomputes every candidate's score and TTFT, verifies final ranking, exported
choices/action counts/accuracy/TTFT, threshold boundaries, missing-feature dense
fallback, maximum depth/leaves, minimum leaf sizes and sample counts. A complete
resume must preserve all completed artifacts and the entire prior study. CPU tests
also construct full 16-leaf depth-4 and 32-leaf depth-5 trees and compare the expanded
fitter against the original implementation.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest tests.test_deeper_router_study tests.test_attention_study -v
```
