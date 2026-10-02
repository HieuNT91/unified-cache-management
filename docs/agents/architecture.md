# Architecture

Start with [root rules](../../AGENTS.md). Commands and data paths in this guide
are relative to the worktree root.

## Entry points and modules

| Area | Files | Responsibility |
|---|---|---|
| Public CLI | `run.py`, `run.sh` | Prepare, run, setup and sweep dispatch |
| Input identity | `runner/layout.py`, `runner/config.py`, `runner/identity.py` | Token-preserving layouts, request metadata, configuration and hashes |
| Execution | `runner/worker.py`, `runner/rope_window.py` | Model/worker integration, bounded prefill and RoPE window |
| Cache lifecycle | `runner/cache.py`, `runner/setup_store.py`, `runner/setups.py` | Construction, persistent setup, readiness and reader ownership |
| Temporary reuse | `runner/temporary.py`, `runner/sweep.py` | Prompt-major construction, reuse across methods and retirement |
| Connector/storage | `ucm/`, especially `ucm/store/` | Scheduler/worker metadata transport and external KV operations |
| Sparse attention | `ucm/sparse/prophetkv/` | Scoring, token selection, alignment, causal repair and lifecycle |
| Router | `runner/router_*.py`, `runner/tree_policy.py`, `runner/tree_profiles.py` | Feature collection, compatible policy decisions, dense/sparse dispatch and reporting |
| Reporting/recovery | `runner/reporting.py`, `runner/resume.py`, `runner/matched_status.py` | Accepted records, aggregation and resume validation |
| Dataset adapters | `scripts/ruler.py`, `scripts/longbench_v2.py` | Source preparation and protocol-specific formatting/scoring |
| Fixed controls | `scripts/rpkv_gpu_validation.py`, `scripts/rpkv_gpu_sequence.py`, `scripts/rpkv_gpu_results.py`, `scripts/rpkv_longbench_inputs.py` | Cohort execution, sequencing, reporting and full-text fitting selection |
| Launchers | `scripts/launcher/*.sh` | Server and corpus/router shell entry points; shared `.env` loader |
| Router training and evidence | `runner/corpus*.py`, `scripts/corpus_*.py`, `scripts/launcher/ruler_corpus*.sh` | Collect baseline/ProphetKV outcomes and router probe features, train/export policies, replay and validate evidence |

`scripts/ruler.py` contains both the RULER implementation and CLI; use
`scripts.ruler` for Python imports. The complete prompt plus output reserve must
fit 65,536 tokens; input length is not fixed at 64,000. Parent `benchmarks/` is
historical implementation, not a dependency for new work.

## Cleanup scope (2026-10-02)

The supported experiments are no-cache baseline, ProphetKV and router. Removed
files belonged to attention-feature studies, GPU-feature studies, deeper-router
search and linked action-extension studies: `attention_features.py`,
`attention_study.py`, `gpu_study*.py`, `deeper_router_study.py`,
`corpus_extension.py`, `extension_train.py`, and their script/test entry points.
Their historical designs and frozen experiment runtimes remain separate.

The obsolete `scripts/router_validate_local.py` and bundled
`scripts/router_cohort.json` were also removed: their legacy per-sample/padded
cohort is incompatible with original-token RULER. `scripts/router_inputs.py`
now contains only current batch preparation and validation. The historical
one-shot policy exporter remains a data reader and requires an explicit
`--cohort /path/to/original/frozen/cohort.json`; it does not make old policies
compatible with current inference.

`corpus_*` modules remain because fixed controls share their runtime/record
validation and router workflows need collection, training and replay. The
5/10% collection helpers remain part of ProphetKV action collection and the
router's measured-action training view. Setup/sweep, status/resume, dataset
adapters and CPU/GPU validation are dependencies of the retained workflows.
Selective-layer support shares these runtime modules; this cleanup does not
change the selection algorithm or remove shared code paths.

## Request and cache flow

1. Dataset preparation produces one token sequence plus evaluation metadata.
   The formatter records prompt/template/token identities and question positions.
2. `runner/layout.py` derives context ranges and a fresh suffix without editing
   tokens. All answer methods consume the same prepared sequence.
3. Construction submits each context chunk with `phase=populate`, local positions
   starting at zero, and explicit request metadata. Readiness waits for committed
   shards on every participating rank.
4. A sparse read sends the original complete prompt with `phase=read`. Stored KV
   is loaded into original global slots and aligned once; selection/repair keeps
   those positions through all prefill steps. Identical chunks can share storage
   while loading into different destination slots.
5. Finish/cancel/retirement clears request state before another policy or prompt.
   Temporary caches can be removed only after owned operations retire. Persistent
   readers retain the shared cache and never rebuild it implicitly.

Baseline has no connector and no cross-request prefix reuse. It still uses the
model's ordinary autoregressive KV within its own request. Dense router fallback
must bypass UCM and verify zero reused KV and zero UCM operations.

## Runtime and diagnostics

The inherited runtime targets Qwen3-32B BF16, TP configurable (default four),
YaRN4 from 32768 to 131072 positions, and vLLM 0.9.2. A run's allocation, hardware
and version receipts determine its actual configuration; the YaRN table size
does not guarantee that the full window fits available GPU memory.

Cache construction, committed-shard readiness and ordinary priming are separate
stages. Priming exercises the prepared execution path before the measured answer;
it is not the independent diagnostic probe removed from current fixed controls.
New fixed controls validate scores/masks recorded by the answer itself. Sparse
attention scoring inside that answer remains part of its algorithm and TTFT.
Router decision probes remain necessary when a selected policy needs attention
features; their online overhead belongs to router timing.

See [invariants](invariants.md), [testing](testing.md) and
[experiment rules](experiment-rules.md) before changing these paths.

## Routine loading and reporting

There is no separate verify stage. Protocol loaders read the manifest and small
identity receipts without scanning every source/prompt or saved diagnostic.
Workers retain input hash/layout checks when consuming a sample, native answer
diagnostics, cache readiness/immutability and retirement. Default resume is fast.
Final fixed-control reporting uses committed scores/token counts and writes a
completion receipt declaring that independent replay was not performed. Explicit
offline training/evidence checks remain separate from routine reporting.
