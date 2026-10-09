# Testing and evidence limits

## L40 thinking 30 + 170 extension (2026-10-08)

The 2026-10-09 adapter import compatibility fix passed all6 extension CPU tests
on local Python3.12. Git blobs confirmed the exact old/new SHA256 pair differs
only in the check_model import; the moved function has identical AST. Fixtures
accept this pair, preserve existing prepared bytes/mtimes, publish the source
mapping and2210-row plan, and reject unknown adapters, other spec differences
and token mismatches. Real noah prefix equivalence remains to be checked by its
prepare command; no remote/GPU execution or real-data regeneration was performed.

The follow-up for the actual TP4 parent with failed/incomplete probes passed all12
extension/A800 CPU tests. A full synthetic TP4 controls-only parent configures
without final/probe/export receipts, schedules1105 prompts per group, leaves
partial probes unchanged, and completes the170-row delta while deferring the
200-row union. After parent probes/export finish, the CPU `merge` command produces
2600 rows and is idempotent; reconfiguring keeps the same source pins. Legacy
complete-parent metadata and TP2 remain covered. No remote/GPU execution.

49 focused CPU tests passed using the available local Python3.12/NumPy environment:
`test_ruler_extension`, `test_tp2_collection`, `test_a800_longbench`,
`test_export_features`, and the thinking preparation/launcher regression cases.
Fixtures cover the full390-row parent plus2210-row delta, immutable parent files,
prefix/token/policy/version mismatches, duplicate rejection, original ordinals,
546/546/559/559 TP2 sharding, frozen resume metadata, compact delta membership,
and portable2600-row union retaining original measurements and source provenance.
Shell/Python syntax and diff checks passed. No real source generation, training,
GPU inference, SSH, push or remote launch was performed. The deployment remains
Python3.10; these checks do not establish GPU/runtime compatibility.

## General requirements

Follow [invariants](invariants.md) and [experiment protection](experiment-rules.md).
Run commands from the worktree root. Use the compatible installed environment:
Python 3.10, uc-manager 0.3.0, vLLM 0.9.2, PyTorch 2.7.0 and transformers 4.53.2
were the recorded runtime versions. Confirm actual versions in each new run.
Runtime patches are process-local; do not edit site-packages.

## CPU checks

```bash
export PYTHON_BIN=/path/to/environment/bin/python
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest discover -s tests -v
```

The standalone upstream verification script and launcher `verify` stage were
removed at the user's request. CPU regression tests remain the development gate;
preparation still enforces token budgets/prefixes and preserves overflow errors.
Per-request checks remain in the inference runtime. Routine launch/report paths
do not reopen the complete prompt/raw corpus or replay saved attention archives.

| Area | Relevant tests/checks |
|---|---|
| Original-token layout, phase, hashes, real EOT, duplicates, partial chunks and suffix | `tests/test_rpkv.py`, `tests/test_ruler.py`, `tests/test_longbench.py` |
| Local/global KV positions, causal mask, 0/100% repair and multi-step prefill | `tests/test_chunked_prefill.py`, `tests/test_cpu.py`, `tests/test_rpkv.py` |
| Missing shards, readiness, immutable KV, setup and retirement | `tests/test_setups.py`, `tests/test_sweep.py`, `tests/test_rpkv.py` |
| No independent fixed-control probes, historical compatibility and result counts | `tests/test_rpkv_no_probe.py`, `tests/test_rpkv_reporting.py` |
| Manifest-only loading, removed verify CLI, fast resume and report without a model/replay | `tests/test_lightweight_loading.py`, `tests/test_rpkv_no_probe.py` |
| Router training, resume, reporting and server environment | Other `tests/test_*.py` suites relevant to the changed path |

For documentation-only changes, check links, `git diff --check` and preservation
of runtime/artifact files; model inference is unnecessary.

Tests dedicated to removed attention-feature, GPU-feature, deeper-router and
linked action-extension studies were removed with those implementations. Keep
the retained baseline/ProphetKV/router suites, including corpus training and
the measured 5/10% action view; historical test totals are not current suite size.

## Opt-in GPU matrix

GPU tests are separate from CPU discovery. Execute only within the user's
applicable scope, on explicit device UUIDs, with fresh output paths. Existing
authorization is not permission to restart another historical scope.

| Script | Checks |
|---|---|
| `tests/gpu_chunked_prefill.py` | Same prepared input across baseline/sparse modes, 0/20/100% repair, multi-step prefill and empty repair ranges |
| `tests/gpu_persistent_setup.py` | Persistent reuse, missing shards, immutable KV and baseline/control comparisons on at least two inputs |
| `tests/gpu_temporary_sweep.py` | Prompt-major reuse, retirement/deletion and paired single-input/sweep output |
| `tests/gpu_rpkv.py` | TP4 diagnostic probe replay, 1/100% controls, dense bypass versus baseline, readiness/retirement, YaRN alignment and caps |

`gpu_rpkv.py` is an explicit diagnostic test, so its independent probes remain
valid even though new fixed-control experiments no longer issue those probes.
It accepts `--model`, `--input`, `--output`, `--gpu-uuids UUID UUID UUID UUID`;
use an original prepared RULER sample longer than 32K. Full GPU acceptance also
needs source-authored duplicate chunks, real EOT inside/end-of-block, partial
chunks and long suffixes. CPU coverage alone does not establish GPU correctness.
Investigate dense/full-repair discrepancies without changing prompts or thresholds
to manufacture agreement.

## Recorded evidence, not a new test claim

After removing standalone/repeated verification, all 210 CPU tests passed.
Coverage includes manifest loading without prompt/raw payload access, rejection
of the removed CLI command, fast resume without replay, reporting without a
tokenizer/model, and runtime rejection of changed input or model identity.
No GPU inference or frozen experiment mutation was performed for this change.

After moving shell launchers into `scripts/launcher/`, all 206 CPU tests passed,
including all six launchers resolving a relocated checkout's root `.env` and
controller paths when called from `/tmp` with spaces in the checkout path.

After the 2026-10-02 source cleanup, all 205 retained CPU tests passed. Python
and shell syntax checks and 13 retained CLI `--help` checks also passed; the
322 snapshotted files under existing `outputs/*/frozen-code/` stayed unchanged.
This cleanup did not run GPU inference or change experiment acceptance rules.

The October 1 no-independent-probe source update recorded 240 passing CPU tests
in `outputs/rpkv-validation/cpu-tests-no-independent-probe-final.log`. It had no
new GPU validation at that update. The earlier five-sample GPU controls completed
210 answers and 70 probes with their own frozen source; the expanded scope keeps
its original 540 probes. Neither establishes every opt-in edge case or validates
later source changes. Read [study receipts](../studies/rpkv-controls.md) and current
package validation files before making a stronger claim.

## TP2 collection and portable trainer (2026-10-02)

The complete CPU regression suite passed258 tests after TP2/training integration.
After final checksum recovery, input-overlap accounting and boundary/report edits,
all8 focused portable-training tests passed (including full synthetic1300/503/1803
cohorts with a reduced hyperparameter grid). The TP2-focused9 tests cover warmup,
exact reduction/head coverage, resource checks, sharding, late extra configuration
and failure cleanup. Shell syntax, Python compilation, CLI help and diff checks
passed. No GPU experiment, real-data fit, push, SSH or remote deployment ran.
See [the workflow guide](../deployment/A800_LONGBENCH_DATA.md).

The subsequent200-samples/task RULER update passed all262 CPU tests. New coverage
checks the2600-row preparation,520 prompts per TP2 pair, immutable configured
sample count, declared-count export/import rejection and synthetic3103-row
combined fitting/evaluation. The original100-row and LongBench paths remain
covered. No source dataset generation, real-data fitting or GPU launch ran.

The TP2 automatic KV allocation update passed all266 CPU tests. Coverage accepts
larger caches with matching per-rank capacity, rejects insufficient/mismatched
caches and invalid YaRN/context receipts, checks the95% free-memory budget on
A800/L20, and preserves fixed TP4/local allocations. No GPU inference or
remote experiment ran; actual memory occupancy and performance remain unverified.

## Standalone compact feature export (2026-10-07)

The13 tests in `tests/test_export_features.py` passed, as did the20 current
TP2 collection regression tests. Coverage includes the100-name/18-group schema,
uniform/one-hot/zero/random attention, independent five-feature arithmetic and
pairwise head-mask checks, TP4 all-head archive reconstruction, missing/corrupt
records, incomplete cohort/engine-exit receipts, source locks, atomic publication
failure, no overwrite and portable payload checksums.

A copied single-file script exported a full synthetic503-prompt LongBench cohort
outside the repo with no model/prepared path or repo imports. A full synthetic
1300-prompt RULER export checked every record/archive and role/action join while
reusing computed features for its identical synthetic captures. Membership checks
also covered2600 non-thinking and390 thinking RULER prompts. These are CPU
fixtures, not GPU experiments or real-data training/export results.

Use discovery so the existing TP2 tests can resolve their sibling test helpers:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest discover -s tests -p test_export_features.py -v
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest discover -s tests -p test_tp2_collection.py -v
```

See [compact export](../deployment/COMPACT_FEATURE_EXPORT.md). The existing
JSON five-feature trainer and all running/frozen experiment protocols are unchanged.

The subsequent parallel-export update passed all16 exporter CPU tests. The CLI
default of64 processes, positive worker-count validation and one-thread worker
environment/restoration are covered. A relocated standalone script with2 spawned
workers exported the full synthetic503-prompt cohort, including distinct attention
captures; its features, labels, identities, GPU metadata, missing masks and source
receipt digest matched serial export exactly. Only offline extraction timings and
execution metadata are allowed to differ. An actual worker checksum failure
published no output and left no owned child processes. The1300-row RULER fixture,
TP4 captures and prior integrity checks still passed. No real-data export, GPU
job or64-worker performance benchmark was run for this update.
