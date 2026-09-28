# Continue A800 or L20 without selective runs

Use the existing launcher’s `resume` command. It keeps every completed baseline
and ProphetKV result, skips selective, and runs only missing ProphetKV answers.
It does not regenerate prepared inputs. Baseline must already be complete on both
shards, with its engine shutdown receipts. Final totals, including the retained
baseline, are **3521 for A800** and **8000 for L20**.

No remote processes are stopped by installing this change. Follow the transition
below on each server. Keep the **existing** `EXPERIMENT_DIR`, `PREPARED_DIR`, model,
cache directory and GPU assignments. Do not use the new-output-directory commands
from the initial launch guide for this continuation.

## Publish the code from the development machine

```bash
git -C /home/thnguyen/spark/unified-cache-management/.worktrees/prophetkv-clean \
  push origin prophetkv/clean-qwen3-32b-yarn4
```

## Choose the launcher on the server

For L20, from the actual checkout (without an extra `/ucm`):

```bash
cd /data/jh/unified-cache-management
export PYTHON_BIN=/data/jh/envs/ucm/bin/python
export MODEL_PATH=/data/jh/ckpts/Qwen3-32B
export PREPARED_DIR="$PWD/.cache/ruler-64000-thinking-prepared"
export CACHE_ROOT="$PWD/.cache/ruler-temporary"
export LAUNCHER=scripts/l20_ruler.sh
# Keep EXPERIMENT_DIR set to your existing timestamped run directory.
```

For A800, use the same code worktree and prepared directory as the original run.
The documented original paths are:

```bash
export PROJECT=/mnt/sde/jh/projects/unified-cache-management
cd "$PROJECT/.worktrees/longbench-a800"
export PYTHON_BIN=/mnt/sde/jh/envs/ucm/bin/python
export MODEL_PATH=/mnt/sde/jh/ckpts/Qwen3-32B
export PREPARED_DIR="$PROJECT/.data/LongBench-v2/prepared-qwen3-yarn4"
export CACHE_ROOT="$PROJECT/.cache/longbench-temporary"
export EXPERIMENT_DIR="$PROJECT/outputs/longbench-v2-503-yarn4-thinking16k"
export LAUNCHER=scripts/a800_longbench.sh
```

If your run used different directories, keep those values. Read the original
`group-0/plan.json` to check its cache path (`CACHE_ROOT` is its parent).

## Stop the old code before updating the checkout

Fetching and extracting the standalone stop helper do not replace files used by
the live processes. The helper uses only the standard library and matches this
exact output directory. It identifies the coordinator, driver and descendant or
reparented workers, checks PID start times, stops the coordinator, interrupts the
drivers, and escalates to TERM/KILL if required. It waits for owned process exit
and records `stop-receipt.json`. In-flight answers can be discarded; completed
result files, old logs and caches are left intact at this stage. Unrelated jobs
are not targeted.

```bash
: "${EXPERIMENT_DIR:?Set the EXISTING run directory}"
: "${PREPARED_DIR:?Set the EXISTING prepared directory}"
test -f "$EXPERIMENT_DIR/group-0/plan.json"
test -f "$PREPARED_DIR/manifest.jsonl"

git fetch origin
STOP_HELPER=$(mktemp /tmp/ucm-sweep-stop-XXXXXX.py)
git show origin/prophetkv/clean-qwen3-32b-yarn4:scripts/sweep_control.py > "$STOP_HELPER" &&
"$PYTHON_BIN" "$STOP_HELPER" stop --output "$EXPERIMENT_DIR" &&
git merge --ff-only origin/prophetkv/clean-qwen3-32b-yarn4
```

Proceed after stop and merge succeed. The helper does not delete caches. The
resume validator removes only old uniquely owned `sweep-<uuid>` caches after
verifying the stop receipt, process exit, input compatibility and retained answers.

## Resume only missing ProphetKV answers

```bash
RESUME_LOG="$EXPERIMENT_DIR/launcher-resume-$(date +%Y%m%d-%H%M%S).log"
nohup setsid bash "$LAUNCHER" resume >"$RESUME_LOG" 2>&1 </dev/null &
echo "Coordinator PID: $!"
tail -f "$RESUME_LOG"
```

CPU validation runs first and prints the number of missing measurements. It
checks both original shard fingerprints, model/engine settings, token hashes,
all-rank diagnostic replay, retirement and output accounting before starting GPUs.
Compatibility includes the released sweep runtimes at c45e92e, 966b5cf, bcc132c
and 501fc70. Unrecognized code/input changes fail without rerunning anything.
Subsequent resumes require the same continuation runtime.

Completed `result.json` plus diagnostics are the commit boundary. This recovers a
crash after result publication but before reporting. An incomplete prompt directory
without a result is moved to the new attempt’s `incomplete/` directory, then retried.
The continuation uses fresh temporary KV for each prompt that still needs work,
reuses it across only the missing ratios, retires requests and deletes the KV.
Each policy is warmed on its first actual use in the new engine.

Original baseline reports, selective reports and completed per-prompt artifacts
stay unchanged. New ProphetKV results use the existing per-method directories.
New session/engine receipts and combined report state live under
`continuation/attempt-.../`, referenced by `continuation.json`; retained artifact
hashes and the original/new runtime fingerprints are recorded there. Timings span
separate sessions. No selective result is counted as missing or required to finish.

## Live and final aggregation

These commands work for both launchers and automatically detect the continuation:

```bash
bash "$LAUNCHER" status
cat "$EXPERIMENT_DIR/live_summary.md"

# After both workers finish:
bash "$LAUNCHER" aggregate
cat "$EXPERIMENT_DIR/final_summary.md"
```

JSON/CSV/Markdown include baseline and all requested vanilla ratios, per subtask
and overall. Selective is explicitly marked discontinued. Final reports require
all retained/new answers and both resumed worker shutdown/cache-deletion receipts.

If interrupted again, use `bash "$LAUNCHER" stop`, then `resume` with the same
paths and committed code. A new attempt preserves earlier attempt metadata,
verifies retained hashes, and skips any newly completed answers as well.

CPU regression coverage includes compatibility rejection, exact diagnostic replay,
missing baseline rejection, partial-ratio skipping, first-use warmup, repeated
resume, two-shard finalization, locks and stopping scoped CPU subprocesses while
leaving an unrelated process alive. No GPU resume has been run locally.
