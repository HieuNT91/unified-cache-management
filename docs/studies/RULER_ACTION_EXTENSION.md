# Extend a completed RULER corpus with additional actions

> Migration note (2026-10-02): this guide was inherited from earlier
> `prophetkv-clean` deployments/studies. Protocol settings, branch names,
> paths and run status below describe that scope; they are not authorization
> to launch, stop or migrate an experiment. For new `rpkv` work, follow the
> [root rules](../../AGENTS.md), [RULER protocol](../protocols/ruler.md) and
> [LongBench v2 protocol](../protocols/longbench-v2.md). In particular, old
> exact-64000, padding and RULER thinking/16K-output settings are obsolete.
> Shell commands remain relative to the worktree root, not this folder.

`scripts/corpus_extend.py` links a completed clean 260-prompt corpus to a new
append-only collection. The source must contain 20 prompts for each of 13 tasks,
all original answers and probes, and a successful engine-exit receipt. Source
inputs, runtime, receipts, outcomes, attention and the previous tree are pinned.
Old receipts are always validated under their original protocol.

For the September 30 run, the added actions are 5% and 10%, giving dense,
1%, 5%, 10%, 20% and 50%. Exactly 520 new answers are collected in original
prompt order, 5% then 10%. The existing 260 probes/features are reused. Archives
are replayed under their original inventory; new floor-budget/stable-tie masks
and their provenance live under `derived/`. No baseline or independent probe is
scheduled. Ordinary priming remains part of the existing cache lifecycle.

The worker keeps one TP4 cached engine on the original UUID group. It builds each
prompt's KV once, validates readiness, primes, measures missing added actions,
checks exact all-rank scores/masks and all-layer selection, retires requests,
verifies YaRN normalization and immutable KV, then deletes that prompt's cache.
Predictions, output tokens, scores, cap flags, timings and diagnostics are kept.
A 900-second no-progress watchdog and one operational retry apply; validation
mismatches halt. Accepted records cannot be replaced. Resume collects only missing
extension records. Source collection processes are never restarted.

Use a private copy of the clean runtime for detached execution. Control commands pin BLAS/OMP to one thread, matching exact feature replay. Set
`CUDA_VISIBLE_DEVICES=''` for control commands and use the original Python
runtime. Preparation and CPU tests perform no GPU inference. For example:

```bash
python scripts/corpus_extend.py prepare --root /new/extension/results \
  --base-corpus /existing/corpus/results --added-actions 5 10 \
  --previous-tree /existing/ruler13-router-more1pct-20260930/min-ttft.json
python scripts/corpus_extend.py detach --root /new/extension/results
python scripts/corpus_extend.py status --root /new/extension/results
# Only after an interrupted attempt and ownership inspection:
python scripts/corpus_extend.py resume --root /new/extension/results
```

The supervisor is detached with `/dev/null` stdin, its own session, ignored
SIGHUP and empty CUDA visibility. Only the worker sees the original four UUIDs.
`startup-verification.json` is published after the first prompt's two answers
pass validation and its cache is deleted. Verify that receipt once, then leave
the job running; later status checks are on request.

After all 520 answers validate and owned engines exit, `matrix.json` pins the
260 × 6 joined matrix. `training/` runs the same 2,592 settings: two objectives,
signed/positive loss, depth 1–3, minleaf 5/10/20, three penalties and 24 weights.
The fastest candidate satisfying at most 2 pp actual training accuracy loss is
primary. Maximum 1% usage is also reported. The previous router is retained as
a candidate, preventing regression among the reported feasible choices.

Corrected cost is selected answer-engine TTFT plus measured tree traversal,
plus saved one-token probe TTFT only when dense is selected. Every export is
checked on training rows, split boundaries and missing-feature dense fallback.
Scores, latency, method counts and candidate constraints are independently
recomputed. Reports, decisions and rules are published atomically, and a
completed training publication is immutable and safely reusable after a crash.

These are **training-set simulations with assumed integrated probe reuse**.
All 260 rows are used for fitting, tuning and scoring. New-budget timings come
from a later measurement session. Extra feature, synchronization and switching
costs remain unmeasured. No deeper-tree search or live integrated router is
implemented. The finite greedy search does not prove global optimality.
