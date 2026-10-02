# Agent rules for rpkv

These rules govern the current worktree. Historical notes under
`docs/studies/` describe earlier scopes and do not override these rules.

- TP2 A800/L20 collection and portable training live on
  `prophetkv/tp2-router-data`, based on `d0483b1`. Guide:
  `docs/deployment/A800_LONGBENCH_DATA.md`. This scope authorizes source edits,
  CPU tests and two local commits only: no experiment, real-data training, push,
  SSH or remote launch. Preserve the separate rpkv/clean worktrees and jobs.
- New collection uses twelve actions; A800 primary and extra have separate
  immutable GPU assignments. Extra may be configured after primary starts.
  RULER is13x100, seed42, five TP2 pairs, original per-task caps/non-thinking.
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
- Parent `benchmarks/` is historical reference only: do not import/invoke its
  implementation for new work or restart historical schedulers.
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
