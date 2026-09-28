# Server-local settings for A800 and L20

Both launchers automatically read `.env` from the checkout containing `scripts/`,
regardless of the current working directory. Git ignores `.env` and `.env.*`;
`.env.example` remains tracked. Copy the template once or write your own `.env`.
Each setting is a trusted local Bash `KEY=VALUE` assignment on one line. Quotes,
comments, optional `export`, and references to earlier variables are supported.
Do not put multiline assignments or general shell scripts in this file.

Precedence is **existing environment > .env > launcher defaults**. Existing values
also win when later assignments expand them. If a previous shell still contains
old exports, start a fresh terminal or unset those variables once before using
`.env`. All loaded settings are exported to workers. Optional `UCM_ENV_FILE` selects
another file; relative paths are resolved against the checkout root. A missing
explicit file is an error, while no default `.env` preserves the old behavior.

For a continuation, copy the exact original output, prepared-input, model and
cache paths. Do not create a new result directory. Resume verifies these settings
and keeps the original GPU assignments. Optional `GPU_A`/`GPU_B` contain four UUIDs
each; omit them to keep the launchers' existing server-specific defaults.

## L20 `.env`

Save as `/data/jh/unified-cache-management/.env`:

```bash
PROJECT=/data/jh/unified-cache-management
PYTHON_BIN=/data/jh/envs/ucm/bin/python
MODEL_PATH=/data/jh/ckpts/Qwen3-32B
RULER_SOURCE="$PROJECT/.cache/vendor/RULER"
PREPARED_DIR="$PROJECT/.cache/ruler-64000-thinking-prepared"
CACHE_ROOT="$PROJECT/.cache/ruler-temporary"
EXPERIMENT_DIR="$PROJECT/outputs/REPLACE_WITH_YOUR_EXISTING_RUN_DIRECTORY"
RESUME_VALIDATION=fast
```

Replace the last value with the real timestamped directory used at launch.

## A800 `.env`

Save `.env` in the **actual code checkout containing `scripts/a800_longbench.sh`**.
The earlier guide used
`/mnt/sde/jh/projects/unified-cache-management/.worktrees/longbench-a800/.env`.
If you run from the main checkout, save it there instead. With the earlier guide's
paths, its contents are:

```bash
PROJECT=/mnt/sde/jh/projects/unified-cache-management
PYTHON_BIN=/mnt/sde/jh/envs/ucm/bin/python
MODEL_PATH=/mnt/sde/jh/ckpts/Qwen3-32B
LONGBENCH_DATA="$PROJECT/.data/LongBench-v2/data.json"
PREPARED_DIR="$PROJECT/.data/LongBench-v2/prepared-qwen3-yarn4"
CACHE_ROOT="$PROJECT/.cache/longbench-temporary"
EXPERIMENT_DIR="$PROJECT/outputs/longbench-v2-503-yarn4-thinking16k"
RESUME_VALIDATION=fast
```

Keep your actual original paths if they differ. The cache path in
`EXPERIMENT_DIR/group-0/plan.json` is a child of `CACHE_ROOT`.

## Publish, fetch, stop, update, resume

First push from the development machine:

```bash
git -C /home/thnguyen/spark/unified-cache-management/.worktrees/prophetkv-clean \
  push origin prophetkv/clean-qwen3-32b-yarn4
```

On each server, enter the **existing code checkout** and write `.env` as above.
Keep live code in place until the owned workers exit. Fetching and extracting
helpers into `/tmp` does not modify the live checkout. This bootstrap works even
when its old launcher has no `.env` or `stop` support:

```bash
git fetch origin
UPDATE_TOOLS=$(mktemp -d /tmp/ucm-update-XXXXXX)
git show origin/prophetkv/clean-qwen3-32b-yarn4:scripts/server_env.sh \
  > "$UPDATE_TOOLS/server_env.sh" &&
git show origin/prophetkv/clean-qwen3-32b-yarn4:scripts/sweep_control.py \
  > "$UPDATE_TOOLS/sweep_control.py" &&
source "$UPDATE_TOOLS/server_env.sh" &&
ucm_load_server_env "$PWD" &&
"${PYTHON_BIN:?Set PYTHON_BIN in .env}" "$UPDATE_TOOLS/sweep_control.py" \
  stop --output "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR in .env}" &&
git merge --ff-only origin/prophetkv/clean-qwen3-32b-yarn4
```

Proceed after the entire command succeeds. The helper targets only processes
owned by this exact output directory, verifies PID identities and waits for exit.
It preserves completed answers and writes a stop receipt. Resume validates prior
results and removes only stopped, owned temporary KV before rebuilding missing
work. Completed baseline/ProphetKV answers are reused; selective is discontinued.
An in-flight answer may need to be rerun. See [recovery details](RESUME_PROPHETKV.md).

On **L20**:

```bash
nohup setsid bash scripts/l20_ruler.sh resume \
  >"l20-resume-$(date +%Y%m%d-%H%M%S).log" 2>&1 </dev/null &

bash scripts/l20_ruler.sh counts  # quick saved-result counts; also printed at resume startup
bash scripts/l20_ruler.sh status
# After completion:
bash scripts/l20_ruler.sh aggregate
```

On **A800**:

```bash
nohup setsid bash scripts/a800_longbench.sh resume \
  >"a800-resume-$(date +%Y%m%d-%H%M%S).log" 2>&1 </dev/null &

bash scripts/a800_longbench.sh counts  # quick saved-result counts; also printed at resume startup
bash scripts/a800_longbench.sh status
# After completion:
bash scripts/a800_longbench.sh aggregate
```

The launcher logs above are in the code checkout. Worker logs and live/final
summaries remain under `EXPERIMENT_DIR` from `.env`; `status` prints the summary
path. No further exports or manual sourcing are needed for launcher commands.
Do not change `.env` to another output directory while managing a live run.

For later updates, the installed launchers support `stop` directly, so the
bootstrap is no longer necessary. Example for L20 (substitute `a800_longbench.sh`
on A800):

```bash
bash scripts/l20_ruler.sh stop &&
git pull --ff-only origin prophetkv/clean-qwen3-32b-yarn4
# Then use the resume command above, keeping the existing .env paths.
```

Fast resume (`RESUME_VALIDATION=fast`) skips repeated parsing/replay/hashing of
saved attention diagnostics, while checking compatible settings, saved answers,
retirement and diagnostic-file presence. New measurements retain full validation.
Use `full` for diagnostic replay and hashing as well. The chosen mode is recorded
in receipts and reports. Existing exports override the `.env` setting.

The fast-resume update recognizes the previous 16c1f75/2ae7f30 continuation runtime
and preserves results from it. Arbitrary future inference changes are not
implicitly approved by pulling code.
