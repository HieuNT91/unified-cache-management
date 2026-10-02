# Studies and historical records

These records preserve designs, evidence and original scope. They do not grant
permission to run commands or establish current process status. Old exact-length,
chunk delimiter, output-cap and policy assumptions do not override
[AGENTS.md](../../AGENTS.md), [RULER](../protocols/ruler.md), or
[LongBench v2](../protocols/longbench-v2.md).

The 2026-10-02 cleanup removed the attention-feature, GPU-feature, deeper-router
and linked action-extension implementations and their dedicated tests from the
current checkout. Their command examples below are historical and must be read
with the original frozen code, not executed from current `rpkv`. Baseline,
ProphetKV, router inference and router corpus training remain maintained.

| Record | Contents |
|---|---|
| [rpkv controls](rpkv-controls.md) | Completed five-sample GPU comparison and expanded 40-per-task/20-LongBench design; original probe and TTFT accounting |
| [Archived agent notes](agent-history.md) | Previous root AGENTS snapshot with dated experiment ownership, constraints and evidence |
| [Archived runtime guide](legacy-runtime-guide.md) | Previous root README: historical setup, router and CLI context |
| [Attention features](ATTENTION_FEATURE_STUDY.md) | Attention-feature study design and analysis |
| [GPU feature study](GPU_FEATURE_STUDY.md) | GPU feature collection and validation workflow |
| [Deeper router study](DEEPER_ROUTER_STUDY.md) | Extended-depth router study design |
| [RULER action extension](RULER_ACTION_EXTENSION.md) | Additional corpus actions and acceptance scope |
| [RULER added ratios](RULER_ADD_RATIOS.md) | Historical 5/10% action collection and comparisons |
| [RULER corpus training](RULER_CORPUS_TRAINING.md) | Corpus training/replay and policy selection rules |

Artifacts remain in their original local experiment directories. This folder
contains documentation, not copies of predictions, frozen code, caches or policy
files. Consult original manifests and final-validation receipts for evidence;
never infer completion from the historical prose or a launch receipt.
