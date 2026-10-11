# Collection v3: nine actions and answer seeds

The three data launchers now default to a **new run** using `tp2-data-v3`.
Existing checkouts, prepared inputs, results, policies and caches retain their original
protocol. Run old experiments from their original pinned checkout. Extension requests
(`EXTEND_FROM`) retain the legacy twelve-action path; naive-reuse and router launchers
retain their own protocols. Do not use `EXTEND_FROM` for these new v3 runs.

| Server | Physical GPUs / TP2 pairs | Sequences | Answers per method | Total answers | Probes |
|---|---|---:|---:|---:|---:|
| A800 | 0–3 / 2 | 503 | 2,012 (4 seeds) | 18,108 | 503 |
| L20 | 0–9 / 5 | 6,500 | 6,500 | 58,500 | 6,500 |
| L40 | 0–7 / 4 | 6,500 | 6,500 | 58,500 | 6,500 |

Action order: `no cache`, `1`, `5`, `10`, `20`, `30`, `40`, `60`, `80` percent
ProphetKV repair. A800 primary owns all nine; do not configure an extra launcher.
BF16, TP2 and automatic KV remain 95% on A800/L20 and 96% on L40. Memory capacity,
equal rank blocks, ownership, immutable cache, native answer diagnostics and retirement
checks remain in place. L40 device availability remains user-managed.

The phase barrier is **all baseline answers → all feature probes → ProphetKV**.
Each phase waits for owned workers/engines to exit. Within each shard, subtasks are
visited round-robin in fixed order (RULER task order; sorted LongBench subtask names).
Each sequence runs all methods, each method runs seeds sequentially, and cached reads
share one immutable context until all requests retire. Sharding remains source ordinal
modulo group count, independently within each RULER task. L20 has 1,300 sequences per
pair, L40 1,625 per pair; A800 has 252/251.

`ANSWERS_PER_QUESTION` is a positive integer, default 4 on A800, and declares seeds
`0..N-1` for every method including baseline. RULER uses one answer (seed 0), with
500 source rows per task and dataset generation seed 42. The plan and each stage freeze
actions, seed schedule, generation policy, traversal and shard assignment. Changing
these settings requires a new run. Resume checks committed answers individually and
runs missing seeds only. Answer receipts live at
`ROLE/records/METHOD/SEQUENCE/seed-N/validated.json`; probes remain sequence keyed.

Thinking LongBench and RULER use `qwen3-thinking-15000-total16384-v2`: at most 15,000
thinking content tokens, then a forced native `</think>` if needed. The full output
budget is 16,384, including delimiters, transition text and the original RULER prefix.
Natural early closure leaves the remaining budget for answer content; valid answer EOS
ends early. RULER's prefix is emitted once after thinking and excluded from scoring.
There is no separate per-task thinking-answer cap. Non-thinking RULER retains its old
greedy sampling, task caps, formatting and scoring. Probes/populate remain one token
and do not receive the answer policy. All methods receive unchanged prepared token IDs.

`status` and `status_same_count` print only the nine-column overall table. The run root
receives Markdown, JSON and CSV reports. RULER has overall plus 13 task tables;
LongBench has overall, three official length groups, domains and domain/sub_domain
pairs, repeated per seed. There are no category × length cross tables. Done counts,
accuracy, first-output-token TTFT and token averages count answers. Speedup is baseline
mean TTFT divided by method mean TTFT on shared `(sequence, seed)` pairs. Same-count
uses the intersection across all nine methods while retaining full planned denominators.
Empty measurements are `N/A`. Capped thinking counts forced closures; capped answers
counts output-cap stops. Answer tokens include transition/prefix text. Forced and
control-token counts remain separately recorded.

`compute feature` (alias `compute-feature`) runs on CPU using saved probes, even during
collection. It snapshots committed receipt names/hashes and holds only its own exporter
lock; later commits wait for the next export. Only sequences with a probe and every
method/seed outcome enter the file. No probes prints “chưa có data feature probe”; no
complete sequence prints “chưa đủ data”, preserving an existing NPZ. A successful
export atomically updates a matching run/schema; failures preserve the prior file.
The protected `.analysis/ruler-features.npz` and `.analysis/longbenchv2.npz` cannot be
export destinations.

Default files are `longbenchv2.npz` (up to 2,012 rows) and `ruler-features.npz` (up to
6,500 rows): `X` has 100 features, `y_accuracy` and `y_ttft` have nine columns. Each seed
has a unique ID, original sequence/source ID, seed, task/category, metrics and provenance.
Features are calculated once per sequence and repeated across seeds; `group_ids` and
`folds` keep those seeds together. The compact schema is `attention-router-compact-v2`;
portable final JSON uses `attention-router-dataset-v3`, likewise one row per seed.
The legacy readers retain all twelve actions and old schema semantics. Final portable
JSON is exported after completion/engine-exit gates; no training/refit runs automatically.

Deployment: [A800](A800_COLLECTION_V3.md), [L20](L20_COLLECTION_V3.md),
[L40](L40_COLLECTION_V3.md). UUIDs/paths in the examples are saved server information,
not evidence of a live availability check. The v3 exporter uses repository modules;
the legacy compact exporter remains usable as a standalone copied file.

## Review, commit and push from the primary workspace

Run after reviewing the diff and the [CPU evidence](../agents/testing.md). The
explicit list excludes local `.env` files, datasets, artifacts, `.gitignore` and
unrelated study changes. No commit or push was performed during implementation.

```bash
git add -- \
  .env.a800.example .env.l20.example \
  .env.l40.thinking.example docs/agents/testing.md \
  docs/deployment/README.md docs/deployment/COLLECTION_V3.md \
  docs/deployment/A800_COLLECTION_V3.md docs/deployment/L20_COLLECTION_V3.md \
  docs/deployment/L40_COLLECTION_V3.md docs/protocols/ruler.md \
  docs/protocols/longbench-v2.md runner/config.py \
  runner/corpus_records.py runner/corpus_runtime.py \
  runner/generation.py runner/reporting.py \
  runner/thinking_budget.py runner/output_budget.py \
  scripts/collection_v3.py scripts/collection_v3_export.py \
  scripts/collection_v3_report.py scripts/export_features.py \
  scripts/longbench_a800_control.py scripts/longbench_a800_data.py \
  scripts/longbench_a800_report.py scripts/longbench_v2.py \
  scripts/router_dataset.py scripts/ruler.py \
  scripts/launcher/a800_longbench_primary.sh scripts/launcher/l20_ruler_data.sh \
  scripts/launcher/l40_ruler_thinking_data.sh tests/test_collection_v3.py
git diff --cached --stat
git commit -m "Add v3 seeded nine-action data collection"
git push origin prophetkv/tp2-router-data
```
