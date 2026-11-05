# Runner packages

`run.py`, `run.sh` and `scripts/launcher/` remain the public entry points.
The 41 implementation modules are grouped by responsibility:

| Package | Responsibility |
|---|---|
| `inputs/` | Configuration, original-token layouts, identities and preparation |
| `runtime/` | Shared generation, single-input/sweep execution, vLLM worker, TP, activation tiles and thinking/output budgets |
| `cache/` | Cache construction, persistent setups, temporary stores and retirement |
| `corpus/` | Prepared datasets, collection workers, accepted records, runtime, training data, fitting and evaluation |
| `router/` | Router policy schemas, features, decisions, measurement, process ownership, sweeps, export and reports |
| `tree/` | Tree-ProphetKV policy validation and fixed-control collection |
| `reporting/` | Scoring/aggregation, matched status and sweep resume |

`router/tree_policy.py` describes learned routing decisions and action inventories;
`tree/prophetkv.py` describes the tree-based KV method. They serve different
purposes. The separate `runtime/worker.py` and `corpus/worker.py` likewise implement
the vLLM worker and collection-specific worker callbacks respectively.

Package initializers are deliberately lightweight. Importing CPU configuration,
preparation, policy and reporting modules must not initialize a GPU runtime.

## Direct module commands

`run.sh` arguments are unchanged. Direct Python module commands now use:

| Previous module | Current module |
|---|---|
| `runner.sweep` | `runner.runtime.sweep` |
| `runner.resume` | `runner.reporting.resume` |
| `runner.corpus_collect` | `runner.corpus.collect` |
| `runner.router_sweep` | `runner.router.sweep` |
| `runner.router_export` | `runner.router.export` |

Internal imports, worker class names, controller subprocess commands and launcher
ownership checks use the new paths. There are no forwarding modules at the old
paths. Historical study documents and the hash entries inside
`reporting/legacy_sweep_runtimes.json` describe their original snapshots.

Moving source changes current source identities. Existing hash receipts and
accepted artifacts are not rewritten; recursive source discovery records the new
paths. Historical adapter/chunker compatibility mappings remain exact, so a new
preparation is not automatically compatible with an old extension run.

## Dependency audit (2026-11-05)

The audit covers imports, dynamic worker/subprocess names, shell entry points,
tests and active documentation. It does not claim to identify external users.
The unused historical-output comparison helper and its dedicated tests were
removed after the audit; no current runtime or CLI depended on them.

| File or group | Finding |
|---|---|
| `reporting/comparison.py` (formerly `result_comparison.py`) | Removed together with `tests/test_result_comparison.py`; the dedicated tests were its only consumers. |
| `router/export.py` | Optional one-shot policy exporter, invoked directly as a module in the A800 router guide and tested in `test_router`. No runtime import is expected; it is not unused code. |
| `reporting/resume.py` and `reporting/legacy_sweep_runtimes.json` | Required by standalone sweep continuation and the A800/L20 launchers. The JSON is a compatibility resource, not an experiment output. |
| `corpus/train.py`, `corpus/eval.py`, `corpus/training_data.py` | Training and inference configuration share these modules. `scripts/router/corpus_control.py` uses evaluation/training-data readers to validate the RULER inference cohort. Removing them requires splitting those dependencies first. |
| `runtime/sweep.py` | Also provides `PromptCache` to `tree/collection.py`; removing the sweep CLI alone would not make this module removable. |
| `router/guard.py`, `runtime/tiles.py`, `runtime/rope_window.py`, `runtime/tensor_parallel.py` | Small shared helpers used by workers/connectors/runtime. Their size is not evidence that they are redundant. |

All other implementation modules have current production import or command
consumers. The baseline, router and Tree-ProphetKV paths share cache, generation,
record-validation and reporting code.
