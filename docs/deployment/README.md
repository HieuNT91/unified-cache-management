# Deployment guides

Run shell commands from the worktree root. Read [AGENTS.md](../../AGENTS.md) and
[experiment protection](../agents/experiment-rules.md) before process operations.
This documentation move does not authorize a launch, update, stop or resume.

## TP2 collection and portable training

For `noah` with eight L40 GPUs, use [thinking RULER on L40 TP4](L40_RULER_THINKING_DATA.md).
It has a separate launcher/env and two TP4 groups; GPU availability is user-managed.

Use [TP2 A800/L20 collection and training](A800_LONGBENCH_DATA.md) for the new
503 LongBench / 2600 RULER twelve-action scope (200/task; older100/task exports remain supported). Launchers are
`a800_longbench_primary.sh`, `a800_longbench_extra.sh`, `l20_ruler_data.sh` and
`router_training.sh`; they read `.env.a800` or `.env.l20` automatically. The older
TP4 workflow commands below remain separate and compatible.

## Current rpkv setup

All shell launchers live in `scripts/launcher/`. Invoke them from the worktree
root, for example `bash scripts/launcher/l20_ruler.sh status`, or use an absolute
path from another directory. They resolve the worktree root two levels above
their own location; `.env`, data/cache defaults and Python paths stay rooted at
the checkout.

The workflow is configure → prepare → detach (or resume) → status/report.
The separate `verify` command has been removed. Metadata/protocol checks happen
at launch; input hashes/layouts are checked where inputs are consumed. Normal
resume uses `fast`, and reports use committed results without diagnostic replay.

| Launcher | Workflow |
|---|---|
| `a800_longbench.sh` | LongBench baseline/ProphetKV sweep on A800 |
| `a800_router.sh` | A800 router comparison |
| `l20_ruler.sh` | RULER baseline/ProphetKV sweep on L20 |
| `router_infer.sh` | Supplied-tree inference |
| `ruler_corpus.sh` | Router corpus collection, training and replay |
| `ruler_corpus_add_ratios.sh` | Added ProphetKV action collection |

All six launchers accept `stop`, using the same `.env` and `EXPERIMENT_DIR`
as their launch command. For example:

```bash
EXPERIMENT_DIR=/absolute/path/to/run bash scripts/launcher/ruler_corpus.sh stop
EXPERIMENT_DIR=/absolute/path/to/run bash scripts/launcher/router_infer.sh stop
EXPERIMENT_DIR=/absolute/path/to/run bash scripts/launcher/a800_router.sh stop
EXPERIMENT_DIR=/absolute/path/to/run bash scripts/launcher/ruler_corpus_add_ratios.sh stop
```

Router/corpus stop takes the launch lock, checks saved PID/start-time/command
identities and experiment paths, stops scheduling, then terminates the owned
worker/engine sessions. It allows 30 seconds for workers to exit after TERM,
then uses KILL if needed. A dead worker leader's surviving ranks are included;
changed process identities cause an error. No model, prepared inputs, policy,
GPU query or full source validation is needed to stop a run.

Results, caches and original process receipts are preserved. Successful stop
writes `stop-receipt.json` only after owned processes have exited. The add-ratios
launcher targets only `extensions/add5-10/` and stores its stop receipt there;
use the matching launcher for the workflow being stopped. Repeating `stop` on
an idle configured run is harmless. Stop does not update a frozen implementation
or bypass the existing compatibility checks for a later explicit `resume`.

`server_env.sh` is their shared helper in the same directory. Dataset adapters
and Python controllers remain directly under `scripts/`; the public `run.sh`
entry point remains at the worktree root. Archived agent/README snapshots retain
their historical paths; use this directory layout for current commands.

Use the existing environment loader in `scripts/launcher/server_env.sh`. Trusted Bash
`KEY=VALUE` settings support quotes, comments, optional `export` and earlier
variable references; existing environment values take precedence over `.env`,
then launcher defaults. Do not put arbitrary shell scripts or multiline values
in `.env`. The default file is in the checkout root; `UCM_ENV_FILE` selects an
explicit file (relative to that root), and a missing explicit file is an error.

Typical settings are `PYTHON_BIN`, `MODEL_PATH`, `EXPERIMENT_DIR`, `PREPARED_DIR`,
`CACHE_ROOT`, and assigned UUID groups `GPU_A`/`GPU_B`. Router deployments also
need a compatible `ROUTER_POLICY_FILE`. Use fresh paths for new `rpkv` protocols;
for a genuine continuation, retain original manifest paths/configuration and
validate compatibility before resume. Old policies are not automatically valid.

Prepare and validate [RULER](../protocols/ruler.md) or
[LongBench v2](../protocols/longbench-v2.md) with the current adapter. Verify scope,
source/input hashes, output reserve, allocation and UUIDs before an authorized
launch. A server's memory/YaRN allocation must fit the full prompt plus output;
do not truncate or alter a protocol silently to fit hardware.

## Inherited server and recovery guides

These guides preserve earlier operational detail. Their banners distinguish old
protocol examples from current `rpkv`. In particular, `L20_RULER_64000.md` describes
a historical exact-64000 thinking protocol, not new RULER preparation. Branch
names, cohort sizes and paths in inherited commands must not be blindly reused.

| Guide | Purpose |
|---|---|
| [SERVER_ENV.md](SERVER_ENV.md) | `.env` syntax, precedence and historical server update workflow |
| [A800_LONGBENCH.md](A800_LONGBENCH.md) | Two-group A800 LongBench preparation, sweep, monitoring and reporting |
| [A800_ROUTER.md](A800_ROUTER.md) | Native router deployment and policy/cohort verification |
| [L20_RULER_64000.md](L20_RULER_64000.md) | Historical L20 launch configuration; use current RULER protocol for new inputs |
| [RESUME_PROPHETKV.md](RESUME_PROPHETKV.md) | Owned-process recovery, immutable completed results and fast/full validation |
| [A800_SINGLE_GROUP_RESUME.md](A800_SINGLE_GROUP_RESUME.md) | Historical one-group A800 continuation |
| [STATUS_SAME_COUNT.md](STATUS_SAME_COUNT.md) | Matched-count status/report interpretation |
| [RULER_CORPUS_COLLECTION.md](RULER_CORPUS_COLLECTION.md) | Corpus collection deployment, ownership and progress |
| [TREE_INFERENCE.md](TREE_INFERENCE.md) | Explicit tree selection and inference command guide |

Resume only the declared missing work under the original frozen protocol. Do not
update a running package from current source or weaken hashes to make it resume.
The expanded local `rpkv` control experiment was explicitly left on its frozen
probe-based implementation; the stop/update examples do not apply to it.
