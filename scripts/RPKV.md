# rpkv: original RULER 64K and metadata-only chunking

This branch inherits `prophetkv-clean` at `c07faff` and its uncommitted source
changes. Its worktree is `.worktrees/rpkv`. Existing worktrees, frozen runtimes,
processes, policies, caches and results are separate and must remain untouched.
Initial implementation was CPU-only. The user subsequently authorized the GPU
control run described below. Historical experiments remain untouched; no policy
refit or branch push is part of either scope.

## Input and cache contract

All methods submit the same prepared `token_ids`. Chunking adds no marker,
EOT, newline, filler or replacement separator. Native chat special tokens and
source tokens, including actual EOTs anywhere in a prompt, remain intact.

For input length `L` and first question token `q`, suffix start is
`64 * floor(min(q, L - 256) / 64)`, clamped to zero for short baseline inputs.
The context consists of consecutive block-aligned chunks of at most 4096 tokens,
including a final partial chunk when needed. The remaining tokens, complete
question, chat tail and original answer prefix are in the suffix.
`question_positions` contains only question tokens, not the answer prefix.
Short inputs can use baseline; sparse reads require two context chunks.

`runner/layout.py` defines prompt protocol `rpkv-original-tokens-v1` and cache
protocol `rpkv-local-chunk-kv-v1`. Each request carries phase `populate` or `read`,
request ID, token hash, boundaries and question positions in
`SamplingParams.extra_args['rpkv_layout']`. Scheduler dispatch also carries that
layout to workers. The scheduler rejects missing or mismatched metadata before
lookup; workers compare it with their armed request. No delimiter discovery or
last-token phase inference exists in the new connector path.

Independent chunks are built at local position zero. Chained block hashes use
local content/history and the versioned model/cache namespace. Identical chunks
share stored shards; loads target separate original-position slots. Only loaded
keys receive delta RoPE, once per layer. Stored shards remain immutable during
reads. Exact-prefix reuse, global selection, all-layer alignment, original-position
causal attention, bounded prefill, readiness, missing-shard failures and retirement
remain enforced. Baseline has no connector/prefix reuse; dense fallback audits
zero reused KV, all 64 native layers and zero UCM operations.

Unversioned prepared inputs and historical policies are rejected for inference.
Rebuild from original sources; do not remove all tokens with an EOT value or
relabel an old padded prompt. No automatic conversion, policy refit or timing
recalibration is provided. Historical analysis readers can still read their own
separate immutable receipts. Use fresh prepared/cache/result directories.

## Original RULER adapter

Pinned upstream: [NVIDIA/RULER c3f5e3b](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a).
The defaults are all 13 tasks, 500 samples per task and seed 42. Each task is
produced by one upstream generator invocation; sharding happens after the batch.
Task arguments come from `scripts/synthetic.yaml`. `max_seq_length=65536`
includes the fully formatted input and the task's output reserve:

| Family | Reserve and answer cap |
|---|---:|
| NIAH | 128 |
| VT | 30 |
| CWE | 120 |
| FWE | 50 |
| QA | 32 |

The tokenizer renders the original task template as a native Qwen3 user message
with `enable_thinking=False` and an open assistant response. The original answer
prefix is appended to that assistant opening before the generator computes its
budget. Upstream's `input` and `answer_prefix` fields are joined exactly once;
there is no second chat wrapping. An over-budget row leaves the raw batch intact,
writes a validation error and stops, without retrying seeds or editing text.

Answers use greedy temperature 0, top-p 1, top-k 32, no added stop strings and
natural EOS. vLLM normally normalizes greedy top-k to zero; the submission adapter
retains the requested 32 on the wire. Internal probes generate one token without
changing the sample's answer cap. Scoring uses only newly generated text, with
upstream strip/control-character preprocessing and case-insensitive substring
metrics. Reports expose task score rounded to two decimals and null prediction
counts; no prefix reconstruction or custom answer extraction is used for RULER.

Qwen3 non-thinking is an adaptation: the pinned upstream revision has no Qwen3
profile. ProphetKV/router reuse and probes are the evaluated algorithms. BF16,
YaRN4, backend and hardware remain separately recorded; this does not promise
reproduction of published RULER numbers. LongBench retains its own template,
thinking, truncation and scoring protocol while sharing metadata-only chunking.

## CPU preparation and verification

From this worktree, using the installed environment and a local Qwen3-32B model:

```bash
export PYTHON_BIN=/path/to/environment/bin/python
export MODEL_PATH=/path/to/Qwen3-32B
export RULER_SOURCE="$PWD/.cache/RULER-rpkv"
export CUDA_VISIBLE_DEVICES=''

"$PYTHON_BIN" scripts/ruler.py download --ruler "$RULER_SOURCE"
"$PYTHON_BIN" scripts/ruler.py prepare --ruler "$RULER_SOURCE" \
  --model "$MODEL_PATH" --output inputs/rpkv-ruler64k

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  -m unittest discover -s tests -v

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON_BIN" \
  scripts/validate_rpkv.py --ruler "$RULER_SOURCE" --model "$MODEL_PATH" \
  --output outputs/rpkv-upstream-check --samples 2
```

The last command runs two independent seeded task batches for every task at
64K, compares raw bytes, independently tokenizes upstream text, verifies budgets,
prefix placement and question offsets, replays all method metadata, and compares
scoring/rounding with upstream evaluator functions. It performs no model inference.
It validates a small cohort; the full 6500-row preparation checks every actual row
and halts on any overflow. Source, template, tokenizer, generator settings and
raw/token hashes are retained in preparation receipts and result provenance.

`ruler_64000.py` remains an alias-compatible filename, but implements this new
protocol. `l20_ruler.sh`, corpus preparation and native router preparation now use
original batches and new default paths. The native router requires independently
provided policies declaring matching prompt/evaluation protocols; old trees are
not automatically accepted. No compatible learned policy is created here.

## GPU checks prepared, not executed

After a separate GPU validation request with explicit UUIDs, run these opt-in
scripts in fresh output directories; they are never part of CPU discovery:

- `tests/gpu_chunked_prefill.py`: unchanged prepared input across baseline,
  ProphetKV/selective 0%, 20%, 100%, multi-step prefill and empty repair ranges.
- `tests/gpu_persistent_setup.py`: persistent reuse, missing shards, immutable KV,
  and baseline/control comparisons for at least two prepared inputs.
- `tests/gpu_temporary_sweep.py`: prompt-major build/reuse/retirement/deletion and
  paired single-input versus sweep outputs.
- `tests/gpu_rpkv.py`: TP4 independent probe replay, 1% and 100% controls, explicit
  dense bypass versus connector-free baseline, all-rank readiness/retirement,
  YaRN delta audit, immutable cache and task caps. The dense check uses a direct
  control action and creates no learned or substitute tree.

Use a prepared original RULER sample longer than 32K for `gpu_rpkv.py`; arguments
are `--model`, `--input`, `--output`, and `--gpu-uuids UUID UUID UUID UUID`.
A complete GPU acceptance additionally includes source-authored duplicate chunks,
real EOT inside/end-of-block tokens, partial chunks and long suffixes. Those edge
cases already have CPU tests, but require hardware checks before claiming GPU
correctness. Investigate any 100% repair or dense-output discrepancy; do not
change thresholds, prompts or accepted artifacts to force a pass.


## User-requested GPU controls (2026-10-01)

Package: `outputs/rpkv-gpu-validation-20261001/`. Source is snapshotted in
`frozen-code/`; do not mutate an active frozen runtime or duplicate supervisors.
The native control driver is `scripts/rpkv_gpu_validation.py`. There are 210
scheduled answers: original RULER13 ×5 and five LongBench v2 inputs, each with
connector-free no-cache, original all64 ProphetKV1%, and ProphetKV5%. Each sparse
pair has an independent diagnostic one-token probe; 70 probes total, separate
from the answer count and answer TTFT. Sparse controls run before baselines.

Physical GPUs1–4 are UUID-pinned RTX4500Ada24GB cards. The local allocation is
KV65920/1031 blocks with native YaRN4, BF16 TP4, chunk4096 and prefill16384.
RULER uses five-row upstream task batches, seed42, non-thinking and original task
caps. LongBench retains native thinking and16384 output tokens. The user explicitly
approved selecting from full-text inputs whose complete prompt plus output reserve
fits this allocation, after three initial random selections exceeded memory.
All503 rows were tokenized;158 fit. Random seed42 selects source rows87,25,238,220,
202, with10243–48397 input tokens and no truncation. Selection receipts preserve
the eligible pool, all lengths, source identity and the initial unused selection.
This is a constrained five-input LongBench comparison, not a full503-row result.

`results/{ruler,longbench-v2}/report.json`, `report.md` and `report.csv` update after
validated prompts. Separate initialization/construction/readiness/priming receipts,
raw predictions/token IDs, per-rank diagnostics, immutable-cache checks, retirement
and process-exit receipts are retained. The CPU sequence starts LongBench only
after the RULER-owned GPU group exits, then independently validates all210 answers
and70 probes and writes `final/report.md`, `report.json`, `measurements.csv` and
`validation.json`. RULER is rescored with the pinned original evaluator functions;
all generated thinking/answer/control counts are independently replayed.

Do not infer completion from a launch or startup receipt. Only `final/validation.json`
and the accepted records establish completed evidence. The separate100% repair,
dense fallback and synthetic GPU edge-case matrix above remains outside these
requested three measured methods.


The initial GPU attempt halted while priming original `niah_single_1-000`:
PcStore flattened repeated file hashes into an oversized read. Its frozen code,
logs,14 accepted answers and7 probes are preserved unchanged under
`startup-history/duplicate-cache-load/` and excluded from the final comparison.
`TrackedStore` now submits batches with unique file hashes, retaining all separate
destination slots and tracking each backend task through retirement. The real
Python/native pointer-flattening regression and failed-transfer retirement tests
pass; the full CPU suite passes232 tests. The corrected GPU run starts with the
same failing input:16 context chunks, including repeated content. Both scheduled
1% and5% answers and the independent probe passed all-rank replay, immutable-cache,
YaRN alignment and retirement checks. See `results/ruler/startup-verification.json`.
The same frozen70 inputs are used in the corrected full comparison.

### Completed validation

All210 answers and70 independent probes passed the final audit, including
identical input hashes, all-rank replay, cache immutability, retirement and
independent scoring. All owned GPU engines exited. The final evidence is in
`outputs/rpkv-gpu-validation-20261001/final/validation.json`; per-task results
and raw measurement columns are in `final/report.md` and `final/measurements.csv`.

| Dataset | Method | N | Accuracy % | Mean TTFT s | Mean thinking tokens | Mean answer tokens |
|---|---|---:|---:|---:|---:|---:|
| RULER | No cache | 65 | 86.13 | 29.175 | 0 | 27.43 |
| RULER | ProphetKV 1% | 65 | 74.18 | 2.119 | 0 | 28.26 |
| RULER | ProphetKV 5% | 65 | 76.26 | 3.553 | 0 | 27.69 |
| LongBench v2 | No cache | 5 | 40.00 | 9.041 | 2623.4 | 168.2 |
| LongBench v2 | ProphetKV 1% | 5 | 40.00 | 0.909 | 1336.2 | 170.2 |
| LongBench v2 | ProphetKV 5% | 5 | 40.00 | 1.232 | 2163.6 | 99.6 |

TTFT includes request submission and TP action synchronization, but excludes
separate cache construction, readiness, ordinary priming and independent probes.
Four RULER outputs per method reached their task cap; no LongBench output reached
the16K cap. These small-cohort results do not establish general accuracy or
validate the separate full-repair/dense-fallback GPU matrix. Data, caches and
result artifacts remain local and are not included in the Git commit.
