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

## Foreground progress and skipping dataset validation

Offline `train`, `test`, `replay`, and `snapshot` print flushed progress to stderr
from command startup. Stage changes appear immediately; five-second heartbeats
show elapsed time, completed/total counts and the current file, sample, setting,
or fold. An unchanged count is explicitly labeled as still processing, including
while waiting on a publication lock. Final JSON remains on stdout. Logging applies
to new invocations; an already-running process keeps its original code.

For a completed collection whose saved results/features you want to trust, add
`--skip-validation`. This reads the small saved probe/result JSON files directly:
no source checksum scans, attention archive loading/replay, diagnostic revalidation,
or feature reconstruction. Presence of result/acceptance files determines the
available sample inventory; missing records are not invented. The mode is recorded
as `dataset_validation: skipped` in the snapshot, tree, summary and offline report.
It does not provide checked artifact evidence. Schema/shape checks needed to fit a
tree still apply. Timing uses the original recorded values in either mode.

Run in the foreground; the wrapper continues to load `.env`:

```bash
export PYTHON_BIN=/data/jh/envs/ucm/bin/python
export EXPERIMENT_DIR=/data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20
export PREPARED_DIR="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prepared"])' "$EXPERIMENT_DIR/protocol.json")"
export OUT="/data/jh/unified-cache-management/router-six-actions-train120-eval120-$(date +%Y%m%d-%H%M%S)"
bash scripts/ruler_corpus.sh train --skip-validation \
  --action-scope all --evaluation training --train-samples 1560 \
  --trainer ruler13-v1 --seed 42 --policy-count 3 --output "$OUT"
cat "$OUT/summary.txt"
```

Training evaluates all three trees automatically on the same 120 samples/task.
A later standalone evaluation of these unchecked trees also needs the flag:

```bash
bash scripts/ruler_corpus.sh test --skip-validation --evaluation training \
  --tree "$OUT/router1.json" --output "${OUT}-router1-repeat"
```

No GPU process is launched by these commands. Use a new training output directory;
this command does not resume a partially completed CPU training run.

## Tree depths and accuracy versus TTFT

`--depths 1 2 3 4 5` searches those maximum tree depths. Use `--depths 5`
for just depth 5; trees may stop earlier when splits do not help or minimum leaf
sizes prevent them. Accepted depths are 1–32, default 1/2/3. Exported trees retain
their depth bound for offline testing and shared-tree inference. Existing frozen
router policies are unchanged. Deeper trees can overfit and take longer to fit.

`--accuracy-weight W` with the default `--selection-objective legacy` switches
fitting and OOF ranking to a weighted cost:

```text
estimated router TTFT / fitting-fold median dense answer TTFT
  + W * max(0, dense score - selected action score)
```

Scores use the 0–1 scale. `0` optimizes time, `10` assigns a cost of 0.1 to a
one-percentage-point loss, and `100` assigns a cost of 1.0 to that same loss.
Larger weights favor preserving accuracy; they do not guarantee an accuracy
floor. Improvements over dense do not offset losses on other samples. TTFT here
includes saved probe overhead and remains an offline estimate.

With a supplied weight, the search uses cost trees only: two minimum leaf sizes
(10/20) × three leaf penalties (0/.005/.02), or six settings per requested depth.
Each setting is fitted in five folds, then selected policies are refitted on the
whole training set. Ranking minimizes the task-macro mean of the weighted OOF
cost, using each held-out row's fitting-fold timing normalizer. The old <=2pp/4×
feasibility indicator is still reported but does not override weighted ranking.
Distinct OOF decision vectors are preferred for additional policies.
`--policy-count 3` means three exported policies, not three available actions.

Omit `--accuracy-weight` to retain the legacy mixed loss/cost search and ranking,
with 72 settings per depth. Omit both options for the original 216 settings.
Depths, objective and weight are saved in `settings.json`, each tree, and the
summary. Changing these flags requires a new output directory and invocation;
it does not modify an already-running trainer.

For all 120 samples/task, with depths 1–5 and weight 10, run from your clean
branch checkout (or use the absolute `$CODE/scripts/ruler_corpus.sh` path):

```bash
export PYTHON_BIN=/data/jh/envs/ucm/bin/python
export EXPERIMENT_DIR=/data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20
export PREPARED_DIR="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prepared"])' "$EXPERIMENT_DIR/protocol.json")"
export OUT="/data/jh/unified-cache-management/router-depth5-weight10-$(date +%Y%m%d-%H%M%S)"
bash scripts/ruler_corpus.sh train --skip-validation \
  --action-scope all --evaluation training --train-samples 1560 \
  --trainer ruler13-v1 --seed 42 --policy-count 3 \
  --depths 1 2 3 4 5 --accuracy-weight 10 --output "$OUT"
cat "$OUT/summary.txt"
```

This loads `.env` with exported-variable precedence, prints foreground progress,
uses all available actions (including 5% and 10%), and reports offline results.
All 13 tasks train one shared router per selected policy. Evaluation reuses the
same 1560 training samples; it is not an independent held-out test.

## Prefer lower budgets within an accuracy-loss limit

Use `--selection-objective min-budget --max-accuracy-loss-pp 2` to favor sparse
routing directly. It removes the 4× speedup gate for this selection mode:

1. Keep only settings whose OOF task-macro accuracy loss versus nocache is at
   most the specified number of percentage points.
2. Minimize the sample-average selected budget, assigning nocache 100% and each
   sparse action its actual ratio (1%, 5%, 10%, 20%, 40%, or another collected ratio).
3. Break ties by estimated probe-inclusive TTFT, then higher macro accuracy,
   shallower maximum depth, fewer leaves and setting order.

Fitting uses cost trees with `budget_fraction + lambda * positive_accuracy_loss`.
By default this mode searches lambda 0/.1/.25/.5/1/2/5/10/20/50/100, minimum leaf
10/20, and penalties 0/.005/.02: 66 settings per depth. This gives the search
aggressive sparse candidates without relying on measured latency being ordered
by budget. `--accuracy-weight W` restricts lambda to W (six settings per depth),
but the accuracy constraint and budget-first ranking still apply. Omit the weight
for a broader search. All 13 tasks train shared policies; task IDs are not inputs.

`--selection-objective min-ttft` instead minimizes estimated TTFT among settings
within the same accuracy-loss limit. It also has no 4× speedup gate. Its default
fitting grid remains the mixed loss/cost grid; an explicit accuracy weight selects
cost trees only. `legacy` retains the earlier ranking, including its speedup gate
when no explicit weight is provided.

Nocache remains available, including the missing-feature fallback. No tree leaves
are edited after fitting. If no candidate meets the accuracy constraint, training
stops with an error instead of silently exporting an ineligible router. If fewer
settings qualify than `--policy-count`, only the eligible settings are exported;
the summary records requested and exported counts. Different OOF action vectors
are preferred before filling additional policy slots. The constraint is an OOF
selection criterion, not a guarantee for the final refit or future inputs.

Compare 2pp and 5pp on your existing 120/task corpus, in the foreground:

```bash
cd "$CODE"
git pull --ff-only origin prophetkv/clean-qwen3-32b-yarn4
export PYTHON_BIN=/data/jh/envs/ucm/bin/python
export EXPERIMENT_DIR=/data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20
export PREPARED_DIR="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prepared"])' "$EXPERIMENT_DIR/protocol.json")"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
for LOSS_PP in 2 5; do
  OUT="/data/jh/unified-cache-management/router-low-budget-${LOSS_PP}pp-${RUN_TAG}"
  bash "$CODE/scripts/ruler_corpus.sh" train --skip-validation \
    --action-scope all --evaluation training --train-samples 1560 \
    --trainer ruler13-v1 --seed 42 --policy-count 3 \
    --depths 1 2 3 4 5 \
    --selection-objective min-budget --max-accuracy-loss-pp "$LOSS_PP" \
    --output "$OUT"
  if [ -f "$OUT/summary.txt" ]; then cat "$OUT/summary.txt"; fi
done
```

Each run searches 330 settings. The wrapper loads `.env`; exports take precedence.
There is no GPU inference or dataset revalidation with these commands. Progress
and five-second heartbeats remain enabled. Compare `accuracy_loss_pp`, `actions`,
`mean_selected_budget_percent`, `answer_ttft_seconds` and
`estimated_router_ttft_seconds` in each summary. OOF action counts and average
budget are also included. A 5pp limit allows more loss but does not guarantee that
the final refitted tree will choose a lower budget. Both final reports evaluate
on the training samples; they are not independent held-out performance results.

## Choose the available routing actions

`--action-scope` accepts `all` (default), `original`, or individual actions:

```bash
# All collected actions, including completed 5%/10% additions:
--action-scope all

# Restrict sparse choices to 1%, 5%, and 10%:
--action-scope nocache prophetkv-1 prophetkv-5 prophetkv-10

# Equivalent shorthand (commas and percentage suffixes also work):
--action-scope 1 5 10
--action-scope 1,5,10
```

Nocache is always retained for baseline comparisons and missing-feature fallback.
Choose at least one sparse action. Unknown/uncollected action IDs fail explicitly;
do not mix `all` or `original` with individual choices. Actions are ordered by
the source inventory, so argument order does not alter tie handling. This changes
the router's available actions, independently of `--policy-count`.

Only the selected actions, nocache and a probe must be present for a sample to
enter the subset snapshot. With `--skip-validation`, this also works while other
budgets are unfinished. Fully validated readers retain the extension completion
requirements when selecting added actions. Source protocols and accepted records
remain unchanged; subset identities are saved in training artifacts. Standalone
`test --tree ...` inherits that tree's selected inventory when scope is left at
its default `all`; an explicit incompatible scope is rejected.

For the same 120/task training evaluation, favor only the three smallest budgets:

```bash
export OUT="/data/jh/unified-cache-management/router-actions-1-5-10-$(date +%Y%m%d-%H%M%S)"
bash "$CODE/scripts/ruler_corpus.sh" train --skip-validation \
  --action-scope 1 5 10 \
  --evaluation training --train-samples 1560 \
  --trainer ruler13-v1 --seed 42 --policy-count 3 \
  --depths 1 2 3 4 5 \
  --selection-objective min-budget --max-accuracy-loss-pp 2 \
  --output "$OUT"
cat "$OUT/summary.txt"
```

This retains nocache, so it encourages smaller sparse choices without forbidding
dense fallback. If no candidate meets the accuracy limit, no ineligible policy
is exported. The foreground launcher still loads `.env` and prints progress.

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

`ruler13-v1` defaults to these 216 settings:

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

Under the default selection rules, feasibility requires OOF macro accuracy loss <=2 percentage points and estimated
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
