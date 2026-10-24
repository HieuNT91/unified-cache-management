# tree-prophetkv on L20 and A800

Run only `tree-prophetkv`, using complete prepared inputs manually copied from
the completed no-cache run. The launcher never copies inputs, tokenizes,
prepares, reruns no-cache, extracts features or runs independent feature probes.
Existing no-cache results remain in their original experiment directory.

| Server | Inputs | TP2 pairs | Ratios | Answers/prompt | Total new answers |
|---|---|---:|---|---|---:|
| L20 | RULER thinking, 13 × 500 prompts | 5 | 5, 20, 40, 60% | 1, seed 0 | 26,000 |
| A800 | LongBench v2 thinking, all 503 prompts | 2 | 5, 20, 40, 60% | 2, seeds 0–1 | 4,024 |

Both use Qwen3-32B BF16, TP2, automatic KV allocation at 95% memory, and the exact
UUID pairs supplied in the env templates. The L20 window remains 82,304 tokens;
A800 remains 131,072. Thinking policy and original token IDs are preserved from
preparation (15,000 thinking-content / 16,384 total output budget). LongBench's
prepared four-seed receipt stays unchanged; the new schedule selects seeds 0–1.

## Configuration and manual input copy

From the new source checkout, create the appropriate local env file:

```bash
# On L20
cp config/.env.l20.tree-prophetkv.example config/.env.l20.tree-prophetkv
# On A800
cp config/.env.a800.tree-prophetkv.example config/.env.a800.tree-prophetkv
```

Review the model/Python/project paths in these templates:
[L20](../../config/.env.l20.tree-prophetkv.example),
[A800](../../config/.env.a800.tree-prophetkv.example).
Each points to a **new** experiment and cache directory. Do not point
`EXPERIMENT_DIR` at the completed no-cache run.

Manually copy the **contents of the complete prepared folder** to
`$EXPERIMENT_DIR/prepared` before configuring. In particular, these files must
exist directly under `PREPARED_DIR`: `preparation.json`, `manifest.jsonl`,
`samples/`, and any other files from the original prepared directory. Do not copy
the old result/settings/cache directories into the new experiment.

| Server | Existing prepared source | New default destination |
|---|---|---|
| L20 | `/data/jh/unified-cache-management/inputs/ruler-l40-tp2-thinking-500-v3` | `/data/jh/unified-cache-management/outputs/ruler-l20-tp2-thinking-tree-prophetkv-500-v1/prepared` |
| A800 | `/mnt/sde/jh/projects/unified-cache-management/.data/prepared/longbench-v2-a800-tp2-4seed-v3` | `/mnt/sde/jh/projects/unified-cache-management/.results/longbench-v2-a800-tp2-tree-prophetkv-2seed-v1/prepared` |

A800 no-cache results remain at
`.results/longbench-v2-a800-tp2-4seed-v3`. The old worktree's root `.env.a800`
is not loaded automatically; these launchers use the new files under `config/`.
Exported variables override env file values; `UCM_ENV_FILE` can explicitly select
another file. Always review exported `EXPERIMENT_DIR`, `PREPARED_DIR`,
`CACHE_ROOT`, `PERCENTAGES` and UUID settings when switching runs.

## Commands

After the manual input copy, optional CPU-only `configure` adopts the existing
inputs and records their identities. `start` also configures if necessary, then
starts a detached CPU supervisor and one GPU worker per supplied TP2 pair.
No command named `prepare` or `features` exists in this launcher.

```bash
# L20: RULER thinking
bash scripts/launcher/l20_ruler_thinking_tree_prophetkv.sh configure
bash scripts/launcher/l20_ruler_thinking_tree_prophetkv.sh start
bash scripts/launcher/l20_ruler_thinking_tree_prophetkv.sh status

# A800: LongBench v2 thinking
bash scripts/launcher/a800_longbench_tree_prophetkv.sh configure
bash scripts/launcher/a800_longbench_tree_prophetkv.sh start
bash scripts/launcher/a800_longbench_tree_prophetkv.sh status
```

Both launchers also accept `detach` (same as `start`), `resume`, `stop` and
`report`. Resume keeps committed answers and runs only missing method/seed
records. It uses saved receipt identities without repeating the initial full
sample scan; workers still validate each consumed input. Source updates do not
rewrite configure-time provenance; each engine records its actual source.
Changing ratios, seeds, paths or UUID assignments requires a new experiment.
A completed experiment cannot be restarted.

Logs live at `$EXPERIMENT_DIR/tree/supervisor.log` and
`$EXPERIMENT_DIR/tree/cached-*-group*.log`. `status.json`/`status.md` show committed
answer counts, accuracy and TTFT by ratio. Final reports require all scheduled
answers and owned engine exit; they do not replay diagnostics or import the old
no-cache measurements.

## Cache and method behavior

Each worker constructs fresh KV and F3 for one prompt, reuses them across that
prompt's ratios and seeds, then deletes them after retirement. A resumed attempt
reconstructs cache for prompts with missing answers. F3 is method metadata,
recorded during chunk construction; it does not require a separate feature
extraction job. Its values/checksums and KV immutability are checked before reuse
and after answers. Native answer diagnostics retain the method evidence.

Construction and warmup are outside answer TTFT. Internal S4 probing, F3 loading,
tree selection and masked repair remain inside the method's timed answer path.
See [the method guide](../TREE_RMASK.md) for exact semantics. S4 is included in
the percentage budget; adoption fails with the prompt/ratio identified if a
budget cannot hold all required S4 tokens. Inputs are never changed to make a
budget fit.

Initial adoption validates the complete manifest, consumed sample hashes,
model/tokenizer hashes, task/source membership, thinking policy and all selected
budgets. It accepts an experiment containing only its manually copied
`prepared/` folder, and rejects old result directories or nonempty cache roots.
Prepared contents and preparation receipts remain read-only.

These are server launch instructions. Local TP4 evidence is separate from
L20/A800 TP2 capacity and stability; see the validation limits below.

Initial launcher validation on 2026-10-24: 87 CPU tests passed, covering both new launchers,
read-only input adoption, two-seed scheduling from four-seed preparation,
method dispatch, F3 warmup/readiness/immutability/retirement, resume and seeded
reporting, plus existing TP2, thinking, v3, naive-reuse and tree-method regressions.
Shell/Python syntax, CLI help and diff checks passed at that stage.

## Explicit local GPU validation

The opt-in arguments `--local-validation --validation-prompt-ids ID ID` allow a
small, declared selection from existing prepared inputs on one TP4 group of
RTX 4500 Ada GPUs. Set `TP=4` and exactly four full UUIDs in a separate validation
env. This uses the real launcher/controller/worker, BF16, 4 GiB CPU offload per
rank and the unchanged 82,304/131,072 execution windows and answer budgets.
All output/cache paths must be new. No input copying or preparation is performed.
The selection and local profile become frozen run parameters; use the same flags
on resume. This is not the server's default configuration and does not establish
L20/A800 TP2 capacity, I/O contention or full-cohort stability.

### Local GPU test limits (2026-10-24)

The authorized TP4 test used the L20 launcher with original RULER thinking
inputs, 4 GiB/rank CPU offload and a fresh cache. It passed engine initialization,
chunk KV/F3 construction, full cache loading and masked multi-step prefill for
the first 65K-token prompt at 5%. Duplicate launch was rejected. The owned
supervisor/workers were stopped when the user requested finalization; zero
answer records had committed. Existing layer-mean results were preserved.

The A800 launcher adopted two original LongBench inputs for the declared
two-seed test, but its GPU run had not started. The complete ratio/seed matrix,
GPU resume and sustained multi-pair server I/O remain unverified. This partial
test neither reproduced nor established a fix for the reported L20 cache-load
failure. Runtime errors now include the backend error, rank, request and failed
block count while retaining the fatal incomplete-load check. No automatic retry
or timeout increase was introduced to hide failures.

Final commit validation: 181 focused CPU tests passed in an isolated checkout of
the staged tree, without the unrelated uncommitted study modules. This includes
method math/native evidence, persistent setup, temporary cache retirement,
chunked prefill, transfer failure diagnostics, TP2/server launchers, thinking
policies and local validation-profile guards. Staged Python 3.10/shell syntax,
document links and diff checks passed.
