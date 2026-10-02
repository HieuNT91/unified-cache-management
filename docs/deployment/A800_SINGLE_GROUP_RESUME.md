# Continue LongBench on GPUs 0–3 without ProphetKV 15%

> Migration note (2026-10-02): this guide was inherited from earlier
> `prophetkv-clean` deployments/studies. Protocol settings, branch names,
> paths and run status below describe that scope; they are not authorization
> to launch, stop or migrate an experiment. For new `rpkv` work, follow the
> [root rules](../../AGENTS.md), [RULER protocol](../protocols/ruler.md) and
> [LongBench v2 protocol](../protocols/longbench-v2.md). In particular, old
> exact-64000, padding and RULER thinking/16K-output settings are obsolete.
> Shell commands remain relative to the worktree root, not this folder.

Use the existing `a800_longbench.sh resume` command with the settings below.
It preserves the original even/odd sample assignments, but runs those two logical
shards **sequentially on one TP4 group**. GPUs 4–7 are not used. A shared lock also
prevents the two resumed workers from running concurrently if invoked directly.

The active scope becomes baseline plus ProphetKV **1/5/10/20/30%**:
**503 × 6 = 3018 answers**. Completed 15% and selective results stay in their
original directories, but are excluded from scheduling, progress totals and
matched comparisons. The revised completed count is the previous completed count
minus the completed 15% answers; it cannot be inferred from `1726/3521` alone.

Every accepted active-method result is validated and reused, including answers
previously produced on GPUs 4–7. Only missing answers run. Baselines must already
be complete, as required by the existing continuation workflow. An interrupted
answer without a committed result may run again. Prepared inputs and model
settings stay unchanged. Runtime compatibility includes releases `8c2685f` and
`55d634c`, in addition to the earlier pinned releases.

## Server commands

First commit and push the implementation from the development checkout. Then,
in the existing server checkout with its existing `.env`, stop the output-owned
processes before updating code:

```bash
bash scripts/launcher/a800_longbench.sh stop &&
git pull --ff-only origin prophetkv/clean-qwen3-32b-yarn4
```

Keep `EXPERIMENT_DIR`, `PREPARED_DIR`, `CACHE_ROOT`, `MODEL_PATH` and `PYTHON_BIN`
pointing to the same existing run. Resolve physical devices 0–3 to UUIDs on this
server; do not copy UUIDs from another machine:

```bash
nvidia-smi -i 0,1,2,3 --query-gpu=index,uuid,name --format=csv,noheader
export GPU_A="$(nvidia-smi -i 0,1,2,3 --query-gpu=uuid --format=csv,noheader | paste -sd, -)"
export RESUME_SINGLE_GROUP=1
export RESUME_EXCLUDE_PERCENTAGES="15"

RESUME_LOG="a800-single-group-resume-$(date +%Y%m%d-%H%M%S).log"
nohup setsid bash scripts/launcher/a800_longbench.sh resume >"$RESUME_LOG" 2>&1 </dev/null &
echo "Coordinator PID: $!; log: $RESUME_LOG"
```

Alternatively save these three values as `KEY=VALUE` assignments in `.env`, using
the actual comma-separated UUID list for `GPU_A`. `GPU_B` is ignored in this mode.
These options apply to `resume` only; `run` rejects them.

The CPU validator checks compatibility and saved results before publishing the
new continuation and launching a worker. Follow its progress in `RESUME_LOG`.
`RESUME_VALIDATION=fast` retains the existing fast-resume semantics: result hashes,
metadata and retirement are checked; prior attention diagnostics are not replayed
or rehashed. Every new answer receives full validation.

After the new continuation is published:

```bash
bash scripts/launcher/a800_longbench.sh counts
bash scripts/launcher/a800_longbench.sh status
bash scripts/launcher/a800_longbench.sh status_same_count

# After both logical shards have finished:
bash scripts/launcher/a800_longbench.sh aggregate
```

Both status commands automatically use the saved active scope, including from a
fresh terminal without the new environment exports. `status_same_count` uses the
exact prompt-ID intersection of baseline and all five active ratios. It does not
wait for the excluded 15% results. Until validation publishes the new continuation,
status still reflects the previous scope.

Subsequent resumes inherit the exclusions and sequential GPU assignment, even
without the new environment variables. Each result's GPU provenance stays intact;
results from different continuation device assignments remain reusable. Final
aggregation still requires completion and engine shutdown for both logical shards.
Reports note that retained and resumed timings may come from different GPU groups
and sessions; matching samples does not remove that timing difference.

No remote GPU run is performed by installing these changes. The migration path
is covered by CPU tests; execution on the server remains to be verified there.
