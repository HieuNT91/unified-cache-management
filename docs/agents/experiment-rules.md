# Experiment and artifact protection

These rules supplement [AGENTS.md](../../AGENTS.md). Existing user authorization
persists for its stated scope; historical documentation is not new authorization.

## Ownership and scope

- Work in this `rpkv` checkout. Preserve other worktrees, unrelated local edits,
  historical `benchmarks/`, frozen runtimes, accepted results, raw inputs, caches,
  source locks, policies and training/export processes.
- Do not launch, stop, restart, migrate or duplicate an experiment merely because
  an old guide contains a command. Before an authorized intervention, read its
  manifest, progress, supervisor/reporter/active state, launch/detachment receipts
  and active log; verify live PID identities and ownership rather than trusting
  recorded PIDs. Completed historical schedulers must remain completed.
- Local GPU work uses explicitly assigned UUIDs; GPU 0 and dummy workloads are
  forbidden. Keep UUID visibility through drivers and spawned workers. NVML may
  resolve physical indices for metadata only. Do not silently replace UUIDs with
  ordinal visibility. Remote groups require their own verified UUID assignment.
- After startup verification, leave detached jobs running. Check status when
  requested; do not add continuous assistant polling or duplicate exporters.
- No policy refit, timing recalibration, remote launch or push unless covered by
  the user's request. Never invent a compatible tree or substitute synthetic
  policies for a pending learned artifact.

## Current source versus frozen controls

The user requested removal of independent probes from current implementation
while explicitly leaving the expanded experiment running unchanged. Preserve
that separation regardless of later progress or completion.

| Package/source | Contract |
|---|---|
| New fixed controls via `scripts/rpkv_gpu_validation.py` | `answer_validation: native-answer-diagnostics-v1`; `expected_probes: 0`; new paths and source snapshot |
| `outputs/rpkv-gpu-validation-20261001/` | Completed 210 answers and 70 independent probes; preserve final evidence and startup history |
| `outputs/rpkv-ruler40-longbench20-20261001/` | Frozen scope of 1620 answers and 540 probes; preserve runtime, processes and original acceptance rules |
| Router inference / explicit GPU diagnostic tests | Keep probes required by their declared algorithm or validation scope |

The table records contracts, not live process status. New fixed controls use
native answer diagnostics for score/mask replay, all-rank/all-layer checks,
immutable cache and retirement. They do not compare against a separate inference
request. Construction, readiness and ordinary priming remain. Do not relabel old
probe-based records as no-probe results or weaken old validation requirements.

## Identity, resume and reporting

Use fresh prepared/cache/result paths when protocols or runtime identities change.
Keep prompt, template, token, generator and model/runtime hashes in receipts.
Old prepared inputs require source reconstruction or verified mapping/hash;
historical policies/results/caches are not automatically compatible.

Resume must validate the original manifest, ownership and accepted records.
Preserve complete shards and answers; resume only missing work under the original
protocol. Fast resume receipts are not the same as full replay/revalidation.
Fast is now the default: routine readers check committed result identity and
result hashes without hashing/replaying large diagnostic payloads. Explicit
offline evidence readers and an explicitly selected full resume audit retain
their stricter checks; normal launch/report has no separate `verify` command.
Stop/update/resume instructions in inherited guides require separate applicable
scope and must never be used to migrate the expanded frozen control run.

Report actual input lengths and task output caps. Keep raw predictions/token IDs,
null predictions, scoring denominators and thinking/answer/control token counts.
Publish live reports only after validation and retirement; publish final evidence
only after all expected records validate and owned engines exit. Launch/startup
receipts alone do not establish completion.
New fixed-control reports reuse runtime scores and token counts and publish
`final/completion.json` with `report_validation: committed-results-only-v1`.
They do not reload the model/tokenizer or claim independent diagnostic/scoring
replay. Historical frozen reporters and `final/validation.json` remain unchanged.

Fixed-control answer TTFT excludes separate construction, readiness, ordinary
priming and historical independent diagnostic probes; report those costs
separately. It includes native sparse scoring/repair and request/action submission.
Router total TTFT includes online feature/probe/retirement/decision overhead and
the answer's first token. Do not compare answer-only router time as total latency.

Keep data, weights, logs, caches, local secrets and outputs out of Git. Preserve
historical pinned source manifests; a documentation change in this checkout is
not permission to rewrite a frozen package's hashes or artifacts.

See [historical records](../studies/README.md) and
[deployment/resume](../deployment/README.md) for scope-specific context.
