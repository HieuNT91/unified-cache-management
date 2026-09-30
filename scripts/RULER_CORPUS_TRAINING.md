# Dataset reading and shared-router training

Collection produces an unsplit reusable RULER corpus. Training is CPU-only and
never launches an engine. The loader works while collection continues:

```python
from runner.corpus import Corpus
corpus = Corpus('/data/results/ruler-corpus-v1', prepared='/data/inputs/ruler-corpus-v1')
snapshot = corpus.snapshot()
print(snapshot['counts'])
pid = snapshot['samples'][0]['id']
inputs = corpus.inputs(pid)          # token/layout input, without references
attention = corpus.attention(pid)    # validates/decompresses one prompt on demand
outcome = corpus.outcome(pid, 'prophetkv-20')
features = corpus.features(pid)
```

Opening the loader reads only the frozen plan and per-sample preparation receipts.
Snapshots freeze visible acceptance receipts under a publication lock. A complete
sample has one accepted independent probe and every outcome in the action
inventory. Unprepared/missing/in-progress records do not become training examples;
counts disclose exclusion. Corrupt accepted artifacts halt processing. A snapshot
pins receipt hashes, and each receipt pins its artifacts. New arrivals cannot
change an existing run. Save a standalone snapshot with
`scripts/ruler_corpus.sh snapshot --output /new/snapshot.json`.

Attention NPZ files retain exact per-rank/per-layer FP32 arrays, local ascending
FP32 sum/64, native TP-reduced scores, context/question positions, eligible
boundaries, original-to-formatted mappings, and selections for every sparse action.
The reader checks every rank, lossless values, exact FP32 accumulation and a valid
four-rank FP32 reduction order; it never uses a tolerance to conceal disagreement.
Selections exclude the first chunk, take floor(eligible length × ratio), break
equal scores by ascending original position, and return sorted positions.

## Train and evaluate on exactly the same samples

Set the updated checkout's `.env` to the original collection directory:

```dotenv
PYTHON_BIN=/data/jh/envs/ucm/bin/python
EXPERIMENT_DIR=/data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20
```

Train the original four actions now, and all six once the 5%/10% follow-up finishes:

```bash
scripts/ruler_corpus.sh train --action-scope original --evaluation training \
  --seed 42 --policy-count 3 --output outputs/router-original-training

# After 5%/10% collection and final reporting finish:
scripts/ruler_corpus.sh train --action-scope all --evaluation training \
  --seed 42 --policy-count 3 --output outputs/router-six-actions-training
```

`--evaluation training` uses every complete sample for fitting and evaluates the
final trees on **exactly the same IDs**, with no held-out set. For the completed
120/task L20 collection, each run fits and evaluates 1,560 prompts. Do not supply
`--train-samples` unless it equals the full complete count. `split.json` records
identical `train_ids` and `evaluation_ids` and an empty `heldout_ids` list. Reports
explicitly identify training-set resubstitution. Five-fold OOF results still guide
setting selection; final refitted-tree performance is reported separately.

`--action-scope original` reads only the original action inventory and works while
the follow-up is incomplete. `--action-scope all` (the default) joins a completed
`extensions/add5-10/` when present: nocache plus ProphetKV 1/5/10/20/40% for the
L20 run. An unfinished extension raises an error. If there is no extension, `all`
uses the base actions; wait for follow-up completion before the six-action run.
Original and added records retain their respective protocols and are unchanged.
No GPU inference is needed. Timings come from different measurement sessions.

Use a new output directory each time. `summary.txt` and `summary.json` give each
policy's accuracy, baseline accuracy, loss, estimated TTFT/speedup, action counts
and separate OOF metrics. Detailed results are in `routerN-training/report.txt`,
`report.json` and `decisions.csv`. Portable policies are `routerN.json` with
checksum sidecars and readable rules. Re-evaluate a training tree explicitly with:

```bash
scripts/ruler_corpus.sh test --tree outputs/router-original-training/router1.json \
  --evaluation training --output outputs/router-original-training-repeat
```

Offline commands use the saved prepared path when `PREPARED_DIR`/`--prepared` is
absent. An extension requires its original prepared directory. The follow-up
runs the current checkout directly; no frozen runtime copy is required. Leave
any checkout used by an active collector unchanged.

`snapshot`, `replay`, and offline `test` also support the combined reader.
Existing four-action trees automatically select their original corpus. Python
callers use `runner.corpus_training_data.open_corpus(root, action_scope='original')`
or the default combined view; pass the same `action_scope` to `train`. The raw
`Corpus` reader continues to describe the original collection.

## Optional held-out split

The default `--evaluation heldout` preserves the original split behavior. For
example, fit on 80/task and evaluate the other 40/task in the 120/task collection:

```bash
scripts/ruler_corpus.sh train --action-scope all --evaluation heldout \
  --train-samples 1040 --seed 42 --policy-count 3 \
  --output outputs/router-six-actions-heldout
```

`N` is the **total** across all 13 tasks. Each task needs at least two complete
samples. Reserve one training and one test sample/task, allocate remaining
training slots proportionally to remaining per-task capacity using deterministic
largest remainders, then use seeded hash ordering within tasks. Exactly N enter
training when `13 <= N <= complete_samples - 13`; all others enter held-out testing.
For example, **50 complete samples/task gives 650 usable samples; 390 train leaves
260 held out** (30 train and 20 test/task). A seed affects membership and folds,
not the inventory or collected outcomes.

## Shared trainer and artifacts

One shared pooled policy is fitted per selected setting. Task labels enter only
sampling, task-stratified prompt folds and reporting. Router input is exactly five
numeric features: top1/top20/top50 attention mass, cosine agreement, and top20
Jaccard between layer groups 0–31 and 32–63. Questions, references, task IDs, gold
evidence and output predictions never enter policy features. Missing features
select `nocache`; corrupt data raises an error.

`ruler13-v1` searches all 216 settings:

- 90 positive-loss trees: depth 1/2/3, minimum leaf 10/20, shrinkage 0/5/20,
  decision threshold 0/.02/.05/.10/.20.
- 126 cost trees: depth 1/2/3, minimum leaf 10/20, loss weight
  1/2/5/10/20/50/100, leaf penalty 0/.005/.02.

Positive losses are max(0, dense score minus action score). Cost objectives add
loss-weighted penalties to probe-inclusive action time normalized by the fitting
fold's dense median. Each of five folds derives its own fitting data, shrinkage
prior, mean action costs and normalizer. Small task groups spread across five
folds; a fold need not contain every task when a task has fewer than five training
samples. No validation/held-out statistics enter a fold's fitting constants.

Feasibility requires OOF macro accuracy loss <=2 percentage points and estimated
TTFT speedup >=4×. Feasible settings rank by time, accuracy, depth, leaf count and
grid order; remaining ranks prioritize accuracy then time. Distinct OOF decisions
are preferred. Three policies are refitted by default; `--policy-count` changes
policy count independently of action count. Single-leaf trees, duplicates and
unmet targets are explicitly recorded; router1 remains primary. The held-out set
is evaluated only after ranking/refitting and cannot recalibrate costs or select
another primary. Other trainer names fail explicitly until implemented as a new
version; existing runs are immutable.

Run artifacts include snapshot, split, matrices, all settings and OOF results,
fold membership/statistics, fitted trees, and per-policy evaluation reports (`routerN-training` or
`routerN-heldout`, depending on the evaluation mode). Each
`routerN.json` is an individual interpretable tree with full-precision thresholds,
action/feature definitions, training/held-out membership, evidence hashes and
hardware provenance. Transfer its `.sha256` sidecar too. Readable `.txt` rules
accompany every tree. Compiler decisions are checked against the fitted policy
on training feature values, threshold boundaries and missing-feature fallback.

## Replay and offline tree testing

```bash
scripts/ruler_corpus.sh replay --decisions decisions.jsonl --output /new/replay
scripts/ruler_corpus.sh test --tree /data/training/ruler13-v1-run1/router1.json \
  --mode offline --output /new/offline-test
```

A decision line is `{"sample_id":"cwe-000","action":"prophetkv-20"}`. Unknown,
duplicate or unmeasured decisions fail. Replay joins only accepted outcomes; it
cannot synthesize a result for an uncollected ratio. Tree testing uses saved
attention, the unchanged tree and the same measured-outcome join. Default tests
use frozen held-out IDs and hashes. `--evaluation training` instead uses the
exact training IDs and evidence recorded by a training-evaluation run; it cannot
be combined with `--evaluation-snapshot`. `--evaluation-snapshot /path/snapshot.json`
evaluates an explicit later complete snapshot and rejects any training overlap;
select only non-training samples in that snapshot. The original evidence pins
must remain present and valid.

JSON/TXT/HTML/CSV reports include per-task and overall accuracy, paired dense
accuracy differences, answer TTFT, action frequencies, task coverage and speedup.
**Estimated router TTFT** adds the independently measured probe, required export,
feature extraction and retirement to the selected fixed-action answer TTFT. It
is not a live router measurement. Probe archive writing is currently required for
replay validation and included in this conservative estimate, with its duration
also recorded separately. No hardware cost refit occurs at evaluation.

For measured live TTFT and cross-dataset LongBench v2 evaluation, use the
[tree inference guide](TREE_INFERENCE.md).

To freeze later complete samples while automatically excluding training overlap:

```bash
scripts/ruler_corpus.sh snapshot --tree /trees/router1.json --output /new/later-evaluation.json
scripts/ruler_corpus.sh test --tree /trees/router1.json --mode offline \
  --evaluation-snapshot /new/later-evaluation.json --output /new/later-test
```

`summary.json` and `summary.txt` disclose each selected policy's feasibility,
OOF macro loss/speedup, leaf count and duplicate ranks. They retain router1 as
primary even when targets are unmet.
