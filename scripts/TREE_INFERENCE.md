# Testing and inference from an exported tree

These interfaces accept individual `ruler-shared-tree-v1` trees from the reusable
corpus trainer. The older frozen three-policy 1/20/50% bundle continues to use
`a800_router.sh`; it is deliberately not reinterpreted as a new action inventory.
A learned tree and its checksum sidecar are sufficient policy transfers. Model,
environment, inputs and clean Git code must already be deployed on the server.
There is no automatic SSH, GPU launch, training or hardware cost recalibration.

## Offline evaluation

```bash
scripts/ruler_corpus.sh test --tree /trees/router1.json --mode offline \
  --output /results/offline-router1
scripts/ruler_corpus.sh replay --decisions /data/decisions.jsonl \
  --output /results/replayed-decisions
```

Set corpus paths through `.env`; see [training](RULER_CORPUS_TRAINING.md).
Offline reports estimate router TTFT from independent measurements. They cannot
establish fresh GPU answer equivalence or measured live routing speed.

## Fresh RULER inference

Use a new result path, the corpus's prepared root, its result root and its exported
tree. Inputs are the frozen held-out set pinned in that tree:

```bash
export EXPERIMENT_DIR=/results/ruler-tree1-live
export PREPARED_DIR=/inputs/ruler-corpus-v1
export CORPUS_ROOT=/results/ruler-corpus-v1
export ROUTER_POLICY_FILE=/trees/router1.json
scripts/router_infer.sh --dataset ruler --tree /trees/router1.json
scripts/router_infer.sh verify
# Following separate user authorization for the configured UUIDs:
scripts/router_infer.sh detach
scripts/router_infer.sh status
```

`router_infer.sh` without a subcommand means `configure`, never launch. Subsequent
commands read the frozen settings. Configure requires the corpus's held-out
receipts and input hashes; transfer them with the prepared inputs for remote
RULER tests. There is no fresh RULER regeneration that could silently change the
held-out split. Execution remains exact 64,000 input, non-thinking greedy256,
BF16 TP4, independent KV65920 and native YaRN4 table131072.

## LongBench v2 cross-dataset evaluation

Supply the same unchanged tree, including a tree trained only on RULER:

```bash
export EXPERIMENT_DIR=/results/longbench-tree1-live
export PREPARED_DIR=/inputs/longbench-v2-native
export LONGBENCH_DATA=/data/LongBench-v2/data.json
export ROUTER_POLICY_FILE=/trees/router1.json
scripts/router_infer.sh --dataset longbench-v2 --tree /trees/router1.json
scripts/router_infer.sh prepare
scripts/router_infer.sh verify
# Following separate user authorization for the configured UUIDs:
scripts/router_infer.sh detach
scripts/router_infer.sh status
```

Preparation reuses the clean LongBench adapter, all 503 distinct source rows and
answer-only official MCQ scoring. Formatted input is at most 114688; documented
middle truncation protects complete question, four choices and answer instructions.
Native thinking retains temperature .6, top_p .95, top_k20, min_p0, seed0 and a
16384-token combined output cap. BF16 TP4, YaRN4 table131072, chunk/activation
4096 and scheduled prefill <=16384 remain fixed. LongBench uses full 131072 KV
allocation (2048 blocks), independently of the RULER profile. Activation tiling
is explicit in engine configuration, not inferred from the KV block count.
Verification fails clearly if memory cannot fit; it never shortens input limits,
changes precision or disables thinking to force a fit.

References are separate scoring metadata. LongBench is never accepted by the
corpus training commands, never tunes thresholds, never ranks policies, and never
refits action costs. Results are labeled cross-dataset evaluation; RULER accuracy
and speed targets are not assumed to transfer.

## A800/L20, controls, resume and timing

Both adapters use `.env`, `PYTHON_BIN`, `MODEL_PATH`, `EXPERIMENT_DIR`,
`PREPARED_DIR`, `CACHE_ROOT`, `GPU_A`, optional `GPU_B`, and `ROUTER_POLICY_FILE`.
See the [collection guide](RULER_CORPUS_COLLECTION.md) for A800/L20 path examples,
explicit UUID configuration, capacity checks and disk planning. One or two TP4
groups are supported. Group identity remains fixed for each prompt/method.

The scheduled tranche first gets fresh connector-free baselines. A persistent
cached engine then builds one prompt's immutable cache. Each routed answer gets
its own independent all64/1% one-token probe, discarded internal token, export,
retirement, features, unchanged-tree decision, TP synchronization and final answer.
The chosen sparse action is also measured as a fresh fixed-action control on the
same group. A nocache choice reuses the fresh connector-free baseline as control.
Routed and control output token IDs and sparse scores/selections must match.
Dense fallback must show all 64 native attention layers, zero reused KV tokens
and zero lookup/load/store operations. Validation errors halt acceptance.

Live total TTFT starts before probe synchronization and ends at the first routed
answer token. It includes required export/archive replay, feature extraction,
retirement, decision and TP synchronization. Archive duration is also retained
separately. Model loading, warmup, cache construction/readiness and ordinary
priming are separate setup costs. Fixed-action control execution occurs after the
routed answer and is excluded from its live TTFT. Training and inference GPU
provenance are recorded without changing the tree or its cost estimates.

```bash
scripts/router_infer.sh resume   # missing accepted records only, after owned exit
scripts/router_infer.sh status
scripts/router_infer.sh report  # completion requires every input and owned exit
```

Duplicate locks, CPU-only detached supervision, identity-checked owned process
cleanup, 900-second progress watchdog and one operational retry apply. Existing
accepted records and trees remain immutable. Do not reuse a result path for a new
tree/dataset/group. CPU tests cover fake controls, profile separation and resume
logic; real A800/L20 numerical equivalence, runtime fit and latency acceptance are
pending an explicitly authorized launch.

## CPU verification (2026-09-29)

The clean suite passed 144 CPU tests, including 19 new corpus/router tests:

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s tests -v
```

Coverage includes exact 64,000-token formatting with deterministic fixtures for
all 13 tasks; lossless archives, FP32 arithmetic, rank disagreement and corruption;
2/4/6/7-action inventories and independent policy counts; all 216 pooled settings,
fold-local statistics, nonconstant trees and full-precision export boundaries;
concurrent snapshots/publication, partial extension and missing-only resume;
A800/L20 configuration, profile separation, fake live controls and existing clean
cache/retirement/LongBench protected-tail and scoring tests. Synthetic reporting
checked 10,400 answers/2,600 probes, and synthetic 50/task data exercised the full
390-train/260-held-out workflow. Shell syntax, Python compilation and whitespace
checks also passed.

These are CPU fixtures and synthetic records, not collected model answers. The
new real 2,600-input corpus has not been generated here, and neither A800 nor L20
GPU execution was launched. Full pinned-tokenizer/source reconstruction runs at
server preparation; real memory fit, native TP diagnostics, routed/control token
equivalence and measured latency still require the authorized GPU acceptance run.

For metrics over the same accepted prompts across methods, use
`status_same_count`; see [matched status](STATUS_SAME_COUNT.md).
