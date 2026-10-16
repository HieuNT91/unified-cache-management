# L20 thinking RULER: probe features + last_traj_4f

Launcher: `scripts/launcher/l20_ruler_thinking_traj4f.sh`. It automatically loads
`.env.l20.thinking.traj4f`; edit that file directly. No shell `source`/`export`,
`configure`, RULER generator or `prepare` command is needed.

The supplied env uses ten L20 UUIDs in five TP2 pairs, BF16, automatic KV at95%,
the82304-token thinking window and the unchanged prepared output policy:
15000 thinking tokens /16384 total generated tokens. The copied500/task cohort
has6500 original prompts. The schedule is6500 independent feature probes followed
by52000 `last_traj_4f` answers at1/5/10/20/30/40/60/80%. Each pair gets1300 prompts.
There is no no-cache answer phase and no per-prompt `read:prime` request.

`last_traj_4f` computes its own last-token/four-draft selection inside each answer;
the independent feature archive is not used to choose its tokens. Readiness,
original-position attention, all-layer alignment, immutable-cache checks and
retirement remain required. First-request TTFT can include cold compilation.
Removing prime does not fix a failing external KV transfer; the first real read
still loads KV and may surface the same storage error.

## Copy prepared inputs once

On L20, create the destination parent:

```bash
mkdir -p /data/jh/unified-cache-management/inputs
```

On L40, replace `USER@L20_HOST` with your L20 SSH destination and copy **the whole
prepared directory**, including `preparation.json`, `spec.json`, `manifest.jsonl`,
`samples/` and `raw/`:

```bash
scp -r /home/zhufangzhou/jh/projects/unified-cache-management/inputs/ruler-l40-tp2-thinking-500-v3 \
  USER@L20_HOST:/data/jh/unified-cache-management/inputs/
```

Do not run the launcher until SCP finishes. Copying L40 result/cache folders is
unnecessary. The kernel is shipped as source; do not SCP the tuning output folder.
The existing `/data/jh/ckpts/Qwen3-32B` checkpoint must match the prepared model and
tokenizer. Paths on L20 live in `.env.l20.thinking.traj4f`.

The first `start` reads and validates the copied samples once and writes new run
metadata; it never rewrites or retokenizes them. `resume` uses the saved row index
and small receipts; each worker still validates a sample when consuming it.
Source hashes are provenance, not resume guards. Result/cache paths must be fresh
and separate from prepared inputs and model weights.

## Commit and push locally

The working tree also contains trajectory support needed by this launcher. Stage
these explicit dependencies, inspect the diff, then commit/push yourself. Local
env files, prepared data and tuning outputs stay out of Git. Other study scripts
and datasets are not included by these commands.

```bash
cd /home/thnguyen/spark/unified-cache-management
git add .gitignore requirements.txt .env.l20.thinking.traj4f.example \
  runner/config.py runner/corpus_records.py runner/corpus_runtime.py \
  runner/corpus_worker.py runner/generation.py runner/naive_reuse.py runner/last_traj_4.py \
  scripts/longbench_a800_control.py scripts/l20_ruler_traj4f_data.py \
  scripts/launcher/l20_ruler_thinking_traj4f.sh \
  ucm/sparse/prophetkv/attention.py ucm/sparse/prophetkv/causal_fa2_kernel.py \
  ucm/sparse/prophetkv/layers.py ucm/sparse/prophetkv/prophetkv.py \
  ucm/sparse/prophetkv/runtime.py ucm/sparse/prophetkv/draft_readers.py \
  ucm/sparse/prophetkv/trajectory.py \
  tests/test_collection_no_priming.py tests/test_l20_traj4f.py tests/test_fa2_attention.py \
  docs/agents/architecture.md docs/agents/testing.md \
  docs/deployment/L40_COLLECTION_V3.md docs/deployment/L20_RULER_THINKING_TRAJ4F.md
git diff --cached --stat
git diff --cached
git commit -m "Run L20 thinking trajectory collection without per-prompt priming"
env -u GIT_ASKPASS -u SSH_ASKPASS git push origin HEAD:prophetkv/tp2-router-data
```

## Pull and run on L20

```bash
cd /data/jh/unified-cache-management
git pull --ff-only origin prophetkv/tp2-router-data
test -f .env.l20.thinking.traj4f || cp .env.l20.thinking.traj4f.example .env.l20.thinking.traj4f
```

Edit `.env.l20.thinking.traj4f` if paths or device assignments differ. The deployed
vLLM0.9.2 environment also needs `numba` for the trajectory selector; install it
only if absent:

```bash
/data/jh/envs/ucm/bin/python -c 'import numba' || /data/jh/envs/ucm/bin/python -m pip install numba
bash scripts/launcher/l20_ruler_thinking_traj4f.sh start
```

`start` returns after detaching one CPU supervisor, which launches five TP2 workers
for each phase. Existing hardware, worker identity, capacity and owned-process
checks apply. No local or remote GPU was used to create this launcher.

```bash
bash scripts/launcher/l20_ruler_thinking_traj4f.sh status
tail -f outputs/ruler-l20-tp2-thinking-traj4f-500-v1/ruler/supervisor.log
# After a failed/stopped attempt, schedule only missing accepted records:
bash scripts/launcher/l20_ruler_thinking_traj4f.sh resume
```

An already-live supervisor prevents a duplicate launch. If intentionally stopping
this run, use `bash scripts/launcher/l20_ruler_thinking_traj4f.sh stop`.
Probes are under `features/records/probe/`; answers under
`ruler/records/last_traj_4f_*pct/`. `status.json`/`status.md` show partial progress;
`final.json`/`final.md` require all probes and answers. No baseline-dependent
portable router dataset or automatic fit is produced.

## Kernel identity

`ucm/sparse/prophetkv/causal_fa2_kernel.py` contains the supplied
`_ragged_positions_fa2_kernel` from
`outputs/ragged-fa2-softmax-tuning-20261014-v4/candidate_kernel.py` (source SHA256
`cf8db77d5530153b536a9960b9ad07be675018ff4d08e9919713787b937a0cd6`).
Arithmetic and wrapper defaults are preserved: tile64x128,8 warps,2 stages,
`use_exp2=True`. The L20 protocol selects this explicit kernel for sparse fusion
and fresh-suffix attention. The four greedy draft reads retain their existing
paged FlashAttention implementation; ordinary autoregressive decode retains its
installed backend. This launcher does not claim measured L20 speed or GPU
numerical equivalence from CPU checks.
