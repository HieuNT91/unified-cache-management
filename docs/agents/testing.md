# Testing and evidence limits

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
