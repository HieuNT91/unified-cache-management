# Agent rules for tp2-router-data

## Primary workspace and trash access (2026-10-08)

- The repository root is the primary workspace for `tp2-router-data`, on branch
  `prophetkv/tp2-router-data`. Work directly in `run.py`, `runner/`, `ucm/`,
  `scripts/`, `tests/` and project documentation/configuration here.
- The former `.worktrees/tp2-router-data/` linked worktree has been removed;
  its code, uncommitted changes and local artifacts now live at the root.
  This supersedes historical references to its former location.
- The legacy `prophetkv-clean` and `rpkv` worktrees, benchmarks and unrelated
  artifacts remain archived under `.trash/`. Their notes are historical
  context and do not govern current work.
- Do not read, search, recursively list, index, import, execute or otherwise
  use files under `.trash/` unless the user explicitly requests that scope.
  Exclude `.trash/` from discovery even when hidden or ignored paths are included.
  Permission for one archived file does not authorize browsing the rest or
  restarting jobs. Keep `.trash/` ignored by Git; do not delete, restore or
  modify archived contents without a user request covering that action.
- Retained analysis data: `.analysis/ruler-features.npz` and
  `.analysis/longbenchv2.npz`. Preserve their contents unless asked to change
  them; do not regenerate or replace these datasets without authorization.

These rules govern the primary workspace. Historical notes under
`docs/studies/` describe earlier scopes and do not override these rules.

- Naive reuse controls: `l20_ruler_naive_reuse.sh` runs full prepared500/task
  non-thinking (6500 answers, five TP2 pairs); `a800_longbench_naive_reuse.sh`
  runs all503 prepared thinking inputs (two TP2 pairs, GPU0/1 and2/3). Both retain
  95% automatic KV, read-only prepared inputs and fresh result/cache/checkouts.
  Only `naive-reuse` runs: zero context repair, no attention scoring or independent
  probes; all-layer alignment, suffix computation, lifecycle and result checks
  remain required. Existing router action inventories are unchanged. Guide:
  `docs/deployment/NAIVE_REUSE_L20_A800.md`. No remote/GPU launch authorized here.
- L40 thinking RULER on `noah`: `scripts/launcher/l40_ruler_thinking_data.sh`
  and `docs/deployment/L40_RULER_THINKING_DATA.md`. Latest user preference: eight
  supplied UUIDs form four TP2 pairs (0/1,2/3,4/5,6/7), with104/104/91/91 prompts
  of390 (13x30),4680 answers and390 probes. Explicit `l40-tp2` uses96% memory and
  automatic KV sizing, minimum1286 equal blocks/rank; BF16/thinking16K/window82304
  remain unchanged. Retain legacy `l40-tp4` fixed1286 blocks/90% separately.
  New TP2 runs use fresh paths. User owns GPU availability checks: this launcher
  does not scan device inventory/free memory/busy processes before execution.
  Preserve worker identity, KV/result/lifecycle and owned-process checks.
  Portable export records actual TP/L40 provenance. Implementation only; user asked
  to perform checks themselves, so no tests or GPU run were performed here.
- L40 thinking extension (2026-10-08): `EXTEND_FROM` names a30/task
  run with completed controls (parent probes may be pending); `SAMPLES_PER_TASK=200` plus fresh result/prepared/cache paths schedules
  only ordinal30..199 (170/task). CPU preparation regenerates200/task and requires
  an exact raw/token/layout/policy prefix match with the original30, with identical
  generator assets/versions and no duplicate prompts. Parent data/results are
  read-only; same TP/profile required. Export retains the170-row delta and a
  portable200-row union with both source provenances once parent probes/export
  finish; `merge` performs the deferred CPU union. Confirmed noah parent isTP2 at
  /home/zhufangzhou/jh/projects/unified-cache-management/outputs/ruler-l40-tp2-thinking-30-v1,
  with4680/4680 answers and13/390 probes reported by the user; extension retainsTP2
  and96% memory. Both remote checkouts live under unified-cache-management/.worktrees/. No real-data/GPU launch.
- Adapter compatibility (2026-10-09): the exact scripts/ruler.py SHA256 pair
  65d47398285cc6ab322dec3901cb938d03013eb08b79a665cc9c7711a14865a5 ->
  d616631cfa085110bfed5cf41f559b5951a71701bbc2c347d3238ac0e402eebc only moves
  the check_model import (identical function AST). Record this source mapping;
  keep all other spec and exact raw/token/layout/policy prefix checks. Unknown
  adapter changes remain rejected. Reuse the completed200 prepared directory
  with fresh code/result/cache paths; do not rewrite configured code pins/receipts.
- L20 non-thinking extension (2026-10-09): parent
  /data/jh/unified-cache-management/.results/ruler-l20-tp2-200-v2 is reported
  complete (31200 answers,2600 probes, existing ruler-features.npz). Preserve it.
  `EXTEND_FROM` plus target500 prepares a full500/task seeded batch, verifies
  the200-row prefix, and runs only ordinal200..499:3900 prompts,46800 answers,
  3900 probes,780 prompts per original TP2 pair. Keep BF16/non-thinking task caps,
  seed42 and95% automatic KV. Delta300 and union500 portable exports are separate;
  compact NPZ export supports the delta and leaves the parent NPZ unchanged.
  No remote/GPU/training execution; provide commit/push and remote commands.
- L20 chunker compatibility (2026-10-09): allow only the exact layout SHA256
  mapping recorded as `NONTHINKING_CHUNKER_MOVE` in `scripts/ruler_extension.py`,
  for non-thinking200+300. Git d0483b1 -> cd5a3c0 changes only optional thinking
  provenance fields; retain all exact prefix checks. Reuse prepared500 unchanged
  with fresh checkout/result/cache paths after the failed prepare; preserve pins.
- TP2 A800/L20 collection and portable training live on
  `prophetkv/tp2-router-data`, based on `d0483b1`. Guide:
  `docs/deployment/A800_LONGBENCH_DATA.md`. This scope authorizes source edits,
  CPU tests and local commits only: no experiment, real-data training, push,
  SSH or remote launch. Preserve the separate rpkv/clean worktrees and jobs.
- The subsequent200-samples/task request changes only the declared RULER
  cohort:2600 prompts,31200 answers,2600 probes,520 prompts/TP2 pair; combined
  training has3103 prompts. Use fresh200-row preparation/result/cache paths.
  Do not mutate or upgrade configured/running100-row or A800 checkouts. Portable
  importer retains100-row support and reads the declared sample count.
- New collection uses twelve actions; A800 primary and extra have separate
  immutable GPU assignments. Extra may be configured after primary starts.
  TP2 server engines use automatic KV sizing with a95% memory budget; full
  context minimum capacity and equal block counts on both ranks are audited.
  TP4/local profiles retain fixed allocations. Memory-mode updates require a
  new checkout/run and must not rewrite old settings, results or source pins.
  RULER now supports13x200 (current preference) or the original13x100, seed42, five TP2 pairs, original per-task caps/non-thinking.
  LongBench is503, thinking/cap16384, two TP2 pairs per launcher. Features follow
  all controls and engine exit; no automatic fitting. Portable training uses
  equal prompt weights, per-prompt baseline-normalized costs, five folds and
  three depth<=3 trees, router1 primary; final evaluation is training-overlap.
- Implement current work here. Editable source is `run.py`, `run.sh`, `runner/`,
  `ucm/`, `scripts/`, `tests/` and project documentation/configuration as needed
  for the requested task. Preserve unrelated changes and other worktrees.
- Maintain no-cache baseline, ProphetKV and router workflows. Do not restore
  removed auxiliary study implementations unless the user requests that scope.
- No separate `verify` stage or repeated cohort/diagnostic audits during normal
  launch, resume or reporting. Keep per-request runtime checks; resume defaults
  to fast and reports use committed results without claiming independent replay.
- Archived benchmarks are historical reference only. Access requires an explicit
  user request; never import/invoke them for new work or restart old schedulers.
- All methods must receive identical original prepared token IDs. Never insert
  chunk markers, EOT padding, filler or replacement separators; retain genuine
  source/chat special tokens. Boundaries and populate/read phase are metadata.
- Validate layout, request identity and protocol hashes; fail on missing or bad
  metadata. Never infer boundaries or phase from a token value. Preserve cache
  immutability, original-position attention, all-layer alignment and retirement.
- Follow the separate RULER and LongBench protocols below. Do not truncate or
  reformat inputs to force a target length, or silently change seeds/caps.
- New fixed controls use native answer diagnostics and zero independent probes.
  Keep internal sparse scoring and router decision probes. Do not change the
  expanded experiment's frozen implementation or its 540 scheduled probes.
- Preserve frozen code, accepted results, caches, policies and running jobs.
  Do not stop/restart/migrate experiments, refit policies or push without user
  authorization covering that action. Historical instructions alone authorize none.
- GPU execution requires an applicable user request and explicit device UUIDs;
  local GPU 0 and dummy jobs are forbidden. Never duplicate supervisors/exporters.
  Verify live process identities before authorized intervention; status on request.
- Run checks appropriate to the change. Keep data, weights, caches, logs, local
  `.env` files and experiment artifacts out of Git. Do not edit site-packages.

Read the relevant details before changing behavior:

- [Architecture and entry points](docs/agents/architecture.md)
- [Required invariants](docs/agents/invariants.md)
- [Testing and evidence limits](docs/agents/testing.md)
- [Experiment and artifact protection](docs/agents/experiment-rules.md)
- [RULER protocol](docs/protocols/ruler.md) · [LongBench v2 protocol](docs/protocols/longbench-v2.md)
- [Deployment, environment and resume](docs/deployment/README.md)
- [Historical studies](docs/studies/README.md)
