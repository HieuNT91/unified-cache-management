# Compare the same evaluated samples

> Migration note (2026-10-02): this guide was inherited from earlier
> `prophetkv-clean` deployments/studies. Protocol settings, branch names,
> paths and run status below describe that scope; they are not authorization
> to launch, stop or migrate an experiment. For new `rpkv` work, follow the
> [root rules](../../AGENTS.md), [RULER protocol](../protocols/ruler.md) and
> [LongBench v2 protocol](../protocols/longbench-v2.md). In particular, old
> exact-64000, padding and RULER thinking/16K-output settings are obsolete.
> Shell commands remain relative to the worktree root, not this folder.

Each launcher accepts `status_same_count` alongside `status`, using its existing
`.env`, `EXPERIMENT_DIR` and prepared-input settings:

```bash
bash scripts/launcher/a800_longbench.sh status_same_count
bash scripts/launcher/l20_ruler.sh status_same_count
bash scripts/launcher/a800_router.sh status_same_count
bash scripts/launcher/ruler_corpus.sh status_same_count
bash scripts/launcher/router_infer.sh status_same_count
```

Metrics use the **intersection of accepted prompt IDs across all active methods**.
Every method therefore has the same evaluated samples, per task and overall, even
when the two GPU groups advance at different speeds. Equal counts alone would
allow different samples and misleading comparisons. A continuation excludes its
discontinued selective configurations.

The reports recompute the usual accuracy, TTFT and token statistics from those
samples. Corpus/inference reports also recompute paired speedups, differences and
router action frequencies. JSON includes the matched IDs, per-task matched counts
and available counts before matching. Expected/planned totals still describe the
full run; operational progress fields still describe the running workers.

If a method has no accepted samples, the intersection is empty: counts are zero
and metrics are N/A (null or absent), never fabricated from unmatched results.
Task coverage may be incomplete during a run.

Outputs are separate from ordinary live/final reports:

| Launcher | Matched output in `EXPERIMENT_DIR` |
|---|---|
| `a800_longbench.sh`, `l20_ruler.sh` | `same_count_summary.md`, `.csv`, `.json` |
| `a800_router.sh` | `status_same_count.json` and printed JSON |
| `ruler_corpus.sh`, `router_infer.sh` | `same_count_summary.md`, `.csv`, `.json`; paired details in `report_same_count.json`, `.txt`, `.html`, `.csv` |

These are CPU status snapshots. They use accepted sweep ledgers or committed
router/corpus result files with checked result/protocol hashes. They do not replay
saved attention diagnostics, load models, start inference, refit trees, or publish
completion receipts. Corpus/inference `status` now refreshes `live_summary.md`,
`.csv`, and `.json` and prints a compact summary. See the
[collection guide](RULER_CORPUS_COLLECTION.md) for running the standalone CPU
reporter from another checkout without changing a running experiment's code.
`server_env.sh` is a
sourced environment helper, not a launcher.
