# Scripts by function

The 28 Python tools are grouped into seven directories. Filenames and the shell
launcher paths are unchanged; direct Python script paths and import paths now
include their functional directory.

| Directory | Files |
|---|---|
| `data/` | `ruler.py`, `longbench_v2.py`, `corpus_inputs.py`, `router_inputs.py`, `rpkv_longbench_inputs.py`; resources `corpus_sources.json`, `longbench_0shot.txt` |
| `collection/` | `longbench_a800_control.py`, `longbench_a800_data.py`, `collection_v3.py`, `ruler_extension.py`, `tree_prophetkv_data.py` |
| `router/` | `router_control.py`, `corpus_control.py`, `corpus_add_ratios.py`, `train_router.py`; preset `router_actions_1_5_10_20_30_40_50_60.json` |
| `reporting/` | `corpus_status.py`, `longbench_a800_report.py`, `collection_v3_report.py`, `experiment_task_breakdown.py`, `experiment_same_count.py` |
| `export/` | `export_features.py`, `export_tree_outcomes.py`, `export_decode_analysis.py`, `collection_v3_export.py`, `router_dataset.py` |
| `sweep/` | `sweep_control.py`, `sweep_counts.py`, `sweep_report.py` |
| `launcher/` | The 15 existing shell launchers/helpers and `launcher_stop.py` |

For example, `python scripts/ruler.py ...` becomes
`python scripts/data/ruler.py ...`, and Python imports use `scripts.data.ruler`.
`bash scripts/launcher/l20_ruler.sh ...` keeps the same path and arguments.
Package initializers do not load models or GPU runtimes.

## Shared dependencies

The `longbench_a800_*` filenames are retained even though these modules also serve
L20/L40 collection and the Tree-ProphetKV controller. The router and corpus
controllers share process/environment utilities; corpus training-data readers also
use `corpus_add_ratios.py`. Sweep continuation depends on `sweep_control.py` and
`sweep_counts.py` in addition to their standalone CLI uses.

`export_features.py` dispatches collection-v3 and Tree-ProphetKV exports to
repository readers. Its legacy twelve-action export remains standalone with
NumPy. `export_decode_analysis.py` remains a standalone standard-library tool.
Direct controller and reporting CLIs resolve the checkout independently of the
caller's current directory. Controller filenames remain unchanged for process
ownership checks; live process identities and saved receipts are not rewritten.

The two data resources remain beside their readers. All three resource files
retain their original bytes. The action JSON preset is optional and differs from
the collection-v3 action inventory; it is not substituted for that inventory.

## Retained cleanup candidates

This reorganization removes no implementation, launcher or resource:

- `router/router_actions_1_5_10_20_30_40_50_60.json` has no discovered direct
  code/config consumer, but can still be supplied explicitly through `--actions`.
- `data/rpkv_longbench_inputs.py` preserves the documented local full-text subset
  preparation protocol; it is separate from full 503-row preparation.
- `reporting/experiment_same_count.py` retains its documented CLI wrapper and
  calls the shared task report with matching enabled.

Historical study documents and frozen hash manifests retain their original paths.
Current source discovery is recursive and records the relocated files. Existing
source receipts and exact adapter/chunker compatibility mappings are preserved;
moving code does not make a new preparation automatically compatible with an old
extension run.
