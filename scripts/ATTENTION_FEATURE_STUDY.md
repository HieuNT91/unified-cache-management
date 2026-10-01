# Offline attention feature study

This CPU-only study uses the completed 260-prompt corpus and six-action measured
outcome matrix. All prompts participate in fitting, selection and scoring. Results
are training-set findings, with no held-out or live-router performance claim.

Run from the clean worktree with its NumPy environment:

```bash
PYTHON_BIN=/home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python
"$PYTHON_BIN" scripts/attention_feature_study.py all
"$PYTHON_BIN" scripts/verify_attention_feature_study.py
```

The entry point disables CUDA and fixes BLAS/OMP threads to one before importing
NumPy, as required for exact original feature replay. It does not import benchmark
implementation code or launch inference. `prepare`, `extract`, `search`, `report`
and `resume` stages are available; `--output` chooses a fresh result directory.
Default: `outputs/ruler13-attention-feature-study-20260930/`.

`protocol.json` freezes source hashes, feature definitions, implementation hashes,
and all 2,592 settings before searching. Each extraction and feature-subset search
is an atomic checksummed unit. Resume validates and reuses completed units without
rewriting them; an interrupted subset is recomputed. An exclusive process lock
prevents duplicate writers. Changing the pinned implementation requires a new
study directory. Original runtime constants, exports and training sources remain
unchanged.

The original five features are exactly replayed from four per-rank archives for
each prompt. Forty additions comprise 12 concentration, 16 layer coverage, six
adjacent-band disagreement and six selection geometry features. Their precise
formulas are frozen in `runner/attention_features.py` and the study protocol.
The eligible interval excludes the first reused chunk and fresh suffix. Undefined
required features cause dense fallback; corrupted arrays halt extraction.

The variable-column fitter preserves ordered CART, stable sorting, FP64 target
arithmetic, native tie order, depths 1–3, leaf sizes, penalties, objectives and loss
weights. All original 2,592 candidates must reproduce saved decisions. Then each
single addition receives the full grid; the top eight produce 28 pairs, and the
best pair produces six triples. Drop-one ablations refit the full grid. Final
selection retains the original router. Ties prefer accuracy, fewer additions,
fewer leaves, lower depth, then frozen feature and setting order.

Primary TTFT is selected answer TTFT plus the original measured traversal allowance,
with saved probe TTFT charged only for dense. A common frozen traversal allowance
avoids timing noise selecting among identical choices. Actual final-tree traversal
is also measured and reported. Resident-array feature overhead is measured on two
prompts per task, five repetitions per prompt, separately from primary estimates.
The reported improvement is the maximum mean additional overhead that could be
spent before losing the gain over the original router. Live transfer,
synchronization and switching remain unmeasured; measurement sessions are mixed.

Outputs include `report.txt`, `report.json`, `ranked-features.csv`, `per-task.csv`,
`decision-changes.csv`, `cpu-overhead.json`, and interpretable `trees/*.txt`.
Study trees use a deliberately separate offline-only schema and digest envelope;
they are not deployable runtime policies. `validation.json` records candidate and
export checks; the independent verifier checks accepted outcome records, every
exported decision/boundary/fallback, and a full resume with unchanged hashes and
modification times for every completed extraction/search unit.

Tests:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest tests.test_attention_study -v
```
