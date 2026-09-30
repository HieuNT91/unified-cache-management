# Add 5% and 10% to the completed L20 RULER collection

`ruler_corpus_add_ratios.sh` is a **manual follow-up** to `ruler_corpus.sh`.
Run it after the original collection finishes successfully and all of its
supervisor/worker/engine processes exit. It refuses an active or incomplete run.
Do not update the checkout used by the currently running collector: that job
pins its source hashes.

The follow-up reads the original protocol and tranche, reuses the exact saved
prepared inputs, and adds only original all64 ProphetKV **5%, then 10%** answers.
For 120 prompts/task across 13 tasks this means **3,120 new answers** and **9,360
combined answers**. The original 1,560 probes are reused; no new independent
probe, baseline, original sparse answer, or router training is scheduled.

## Deployment and commands

Use the updated checkout directly. Put these assignments in its `.env` file
(adjust paths to your server):

```dotenv
PYTHON_BIN=/data/jh/envs/ucm/bin/python
EXPERIMENT_DIR=/data/jh/unified-cache-management/ucm-ruler120-l20/outputs/ruler13-120-l20
```

After the original collection completes and exits:

```bash
scripts/ruler_corpus_add_ratios.sh detach
scripts/ruler_corpus_add_ratios.sh status
scripts/ruler_corpus_add_ratios.sh status_same_count
```

A separate `verify` command is **optional**. `detach` prepares and validates the
saved inputs and attention in the background before collecting the new answers.
For a CPU-only preflight, run `scripts/ruler_corpus_add_ratios.sh verify` first.
No runtime copy, original source checkout, or `--runtime-code` argument is needed.
The supervisor and workers run from the current checkout; keep that checkout
unchanged while the follow-up is running. Existing older extension results remain
readable without their old `frozen-code/` directory; nothing deletes that directory.

If the original collection is still running, put updated code in a separate
worktree and give it the `.env` above. Its absolute result path shares the existing
records. Do not update the original running collector's checkout.

`--root` overrides `EXPERIMENT_DIR`. Without either, the default is
`outputs/ruler13-120-l20` below the launcher's checkout. The existing trusted Bash
`.env` loader and `PYTHON_BIN` conventions apply. Model, prepared input, cache
location, and GPU UUIDs come from the saved protocol; new `MODEL_PATH`,
`PREPARED_DIR`, `CACHE_ROOT`, or `GPU_A/GPU_B` values do not override those saved
assignments. `detach` checks the saved UUIDs are available and unoccupied, then
launches a detached CPU supervisor. No automatic waiter or launch is installed.

## Results and validation

New results are real directories under the requested existing records root:

```text
outputs/ruler13-120-l20/
  records/prophetkv-5/<original-prompt-id>/{result,diagnostics,validated}.json
  records/prophetkv-10/<original-prompt-id>/{result,diagnostics,validated}.json
  extensions/add5-10/
    protocol.json                 # Separate extension protocol
    sources.json                  # Pins original records, inputs and receipts
    derived/                      # Replayed scores and new exact budget masks
    sessions/                     # New initialization/ownership evidence
    supervisor.log
    detachment.json
    startup-group0.json           # First validated prompt in each used group
    live_summary.{md,csv,json}
    same_count_summary.{md,csv,json}
    report.{md,csv,json}           # Final combined report
    complete.json                 # Published after full validation and engine exit
```

Original settings, protocol, result files, archives, reports, and completion
receipts remain unchanged. The original protocol still describes its original
four actions. **Use the new launcher's reporting commands for all six methods**;
old readers that assume one protocol cannot validate the added records. Each
original record is checked under its original protocol, each new record under
the extension protocol, and initialization paths remain relative to the main
result directory.

The selected cohort is the original saved tranche, in original prompt order,
even if more of the reserved 200/task have been prepared. The ordinary
`partial-1560-validation.json` completion receipt is sufficient for 120/task.
Original one/two TP4 UUID groups and ordinal-to-group assignments are retained.
Each group uses one cached engine, builds each prompt's temporary KV once,
performs ordinary priming, and measures only missing 5%/10% answers. Independent
saved probes are replayed under their original inventory; added masks use native
FP32 scores, exact floor budgets and stable ascending-position ties. Existing
all-rank/all-layer selection, cache readiness, immutability, retirement, warmup,
and runtime initialization checks remain in force. Temporary KV is deleted
before answers are committed.

The supervisor shares the original launch/run locks to prevent overlapping
collection. It has a 900-second no-accepted-progress watchdog and one operational
retry per group. Validation mismatches halt. Failed unaccepted new payloads are
preserved under `incomplete/`; accepted records cannot be replaced. No command
stops or restarts the original collector or another experiment.

## Status and recovery

With `EXPERIMENT_DIR` pointing at the existing output root:

```bash
scripts/ruler_corpus_add_ratios.sh status
scripts/ruler_corpus_add_ratios.sh status_same_count
# After inspecting a failed attempt and confirming its owned engines exited:
scripts/ruler_corpus_add_ratios.sh resume
# Final reports are automatic. Recover interrupted final publication with:
scripts/ruler_corpus_add_ratios.sh report
```

Status checks saved result hashes and the appropriate protocol without loading
the model or attention arrays. Matched status uses the exact prompt-ID
intersection across all six methods. Reports contain answer-only accuracy,
answer-engine TTFT, token counts and caps, per task and overall. Original and
added actions come from different timing sessions. Live reports are not final
acceptance receipts; final publication requires all new diagnostics, source pins,
and owned engine exit to validate.

Verify each group's startup receipt once, then request status as needed. These
commands have CPU coverage; real L20 GPU execution remains a server-side check.

## Train both routers on the collected samples

With the `.env` above, run the original four-action training now:

```bash
scripts/ruler_corpus.sh train --action-scope original --evaluation training \
  --seed 42 --policy-count 3 --output outputs/router-original-training
```

After 5%/10% collection completes, train with all six actions:

```bash
scripts/ruler_corpus.sh train --action-scope all --evaluation training \
  --seed 42 --policy-count 3 --output outputs/router-six-actions-training
```

Both commands fit on **all complete samples and evaluate those exact same sample
IDs**: 1,560 prompts for the completed 120/task collection. There is no held-out
set. Omit `--train-samples`; the command uses the entire complete cohort.
`original` ignores an unfinished extension; `all` requires the extension to finish
if it is present. Use the second command only after the extension finishes to get
six actions. Each output directory must be new.

Read `summary.txt` / `summary.json` in each output directory for accuracy, dense
accuracy, loss, estimated TTFT, speedup and action counts. Per-policy details are
in `routerN-training/report.txt`, `report.json` and `decisions.csv`; individual
portable trees are `routerN.json`. These are CPU-only training-set evaluations
from saved outcomes, not new inference or held-out performance. TTFT includes the
saved independent probe overhead and selected answer time; it is an estimate,
not measured live-router latency. The existing five-fold OOF search still selects
settings before each final tree is refitted on all samples. See the
[training guide](RULER_CORPUS_TRAINING.md) for details.
