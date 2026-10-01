# Add 50%, 70%, and 90% to the existing L20 corpus

Use the clean branch and `scripts/ruler_corpus_add_ratios.sh --percentages`.
Omitting percentages retains the old 5%/10% behavior. Each sorted percentage list
has independent state: `extensions/add50-70-90/` for this batch. Accepted old
answers, probes, protocol and `extensions/add5-10/` are preserved. Do not configure
or prepare a new base corpus and do not restart the original collector.

For 120 samples × 13 tasks, this schedules 4,680 new answers (1,560 per budget).
It reuses the original saved probes; it does not schedule baseline answers or new
independent probes. Each prompt's temporary KV is built once for the new batch,
reused across its missing actions, then retired/deleted. All-layer attention,
mask equivalence, readiness, GPU identity and new-record acceptance checks remain
in force. The 900-second watchdog and one operational retry are unchanged.

Run from the existing clean checkout on L20, after previous collection engines
have exited. `$CODE` must point to that checkout; no extra worktree is required.
The commands below are remote instructions, not an authorization for local GPUs.

```bash
cd "$CODE"
git pull --ff-only origin prophetkv/clean-qwen3-32b-yarn4
export PYTHON_BIN=/data/jh/envs/ucm/bin/python
export EXPERIMENT_DIR=/data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20

# Display the exact GPU UUID groups this extension will inherit.
"$PYTHON_BIN" - "$EXPERIMENT_DIR/protocol.json" <<'PYCODE'
import json,sys
p=json.load(open(sys.argv[1]))
for i,group in enumerate(p['groups']):
    print('group',i,','.join(group))
PYCODE

# Launch only the new batch. No external nohup is needed.
bash "$CODE/scripts/ruler_corpus_add_ratios.sh" detach --percentages 50 70 90

# On-demand progress; same percentage list selects the same batch.
bash "$CODE/scripts/ruler_corpus_add_ratios.sh" status --percentages 50 70 90
tail -n 60 "$EXPERIMENT_DIR/extensions/add50-70-90/supervisor.log"
```

The wrapper loads `$CODE/.env` (or `UCM_ENV_FILE`); exported variables win. Python
and the corpus root use those settings. GPU UUIDs, model, prepared paths, cache
root and sample count are inherited from the original saved `protocol.json` and
tranche; changing `GPU_A`/`GPU_B` in `.env` does not move this extension to another
group. Keep code fixed while collection runs. Collection uses built-in detached
supervision; offline router training can still run in the foreground.

CPU preparation and source replay happen inside the supervisor before GPU work;
first answers may take time. A separate `verify --percentages 50 70 90` is optional
and performs expensive preparation/replay without launching GPUs. The collection launcher also supports `--skip-validation` as described below.

For an interrupted batch, after its owned processes have exited:

```bash
bash "$CODE/scripts/ruler_corpus_add_ratios.sh" resume --percentages 50 70 90
```

Resume measures only missing records. Do not launch a second copy or change the
percentage list midway. Overlapping budgets registered to another batch are
rejected. Every batch holds the shared corpus run lock during collection.

New accepted answers appear at:

```text
records/prophetkv-50/<sample>/
records/prophetkv-70/<sample>/
records/prophetkv-90/<sample>/
```

After successful completion and engine exit, reports are published automatically:

```bash
cat "$EXPERIMENT_DIR/extensions/add50-70-90/complete.json"
cat "$EXPERIMENT_DIR/extensions/add50-70-90/report.md"
```

The batch report compares original actions with 50/70/90; it does not include
the separate 5/10 batch. The offline dataset reader joins all registered batches:
`--action-scope all` gives nocache and 1/5/10/20/40/50/70/90 (nine actions).
Individual subsets work too, for example `--action-scope 1 5 10 50 70 90`.
Old saved-tree testing keeps that tree's original action inventory.

With `--skip-validation`, only rows with saved probe and every selected action
are used, even during unfinished collection. For `--evaluation training`, omit
`--train-samples` to use the currently complete set; the shared trainer still
requires representation from all 13 tasks. Once all 120/task are complete,
`--train-samples 1560` is valid. The fully validated all-action reader requires
completion and engine-exit evidence for every selected extension batch.

No tests or local/remote GPU execution were performed for this launcher update.

## Skip saved-data validation

Add `--skip-validation` to skip old input/answer/diagnostic checksum scans and
full per-layer/per-rank attention replay. Preparation reads only the native FP32
score vector from one original NPZ archive per sample, derives the new masks,
and saves them. It does not decompress the large layer arrays. Saved record
commit markers are trusted on resume, status and final reporting in this mode.
Newly generated answers still undergo diagnostic/score/mask, readiness,
retirement and publication checks. Runtime/model compatibility, original cohort
completion, duplicate locks and exact GPU UUID ownership checks remain enabled.
The current inference input is still checked when its new request begins.

```bash
cd "$CODE"
git pull --ff-only origin prophetkv/clean-qwen3-32b-yarn4
export PYTHON_BIN=/data/jh/envs/ucm/bin/python
export EXPERIMENT_DIR=/data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20
bash "$CODE/scripts/ruler_corpus_add_ratios.sh" detach \
  --percentages 50 70 90 --skip-validation
bash "$CODE/scripts/ruler_corpus_add_ratios.sh" status \
  --percentages 50 70 90 --skip-validation
```

After interruption and owned-process exit, use `resume` instead of `detach`:

```bash
bash "$CODE/scripts/ruler_corpus_add_ratios.sh" resume \
  --percentages 50 70 90 --skip-validation
```

The flag is forwarded to the supervisor and workers. It can also be passed to
`verify` (preparation only), `report`, and `status_same_count`. Newly prepared
extensions store `source_validation: skipped` and inherit it on later commands.
For an extension originally prepared with full validation, passing the flag on
resume does not rewrite its protocol or accepted records; the new session,
new answers and completion report carry the skipped-source label. A separate
`source-validation.json` remembers this choice without changing accepted records,
so subsequent commands also inherit the skipped-source mode.

An already-running process cannot pick up the flag or changed code. This update
does not stop or replace it. Use a new invocation after the owned processes exit;
do not launch a duplicate. Fast collection still spends time deriving masks and
building KV caches; skipping replay does not eliminate these operations.

Use `--skip-validation` for subsequent offline training/reading of a batch that
trusted its saved sources. The fully validated reader rejects such batches
rather than silently presenting them as fully source-validated.
