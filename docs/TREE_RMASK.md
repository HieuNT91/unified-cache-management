# tree-prophetkv

F3 tree selection and masked recomputation, ported from
[`prophetkv/tree-rmask-qwen3-4b` at `f96cb874`](https://github.com/HieuNT91/unified-cache-management/tree/f96cb874c7becddd868f8f47872701f6111489ac).
This integration uses the current branch's Qwen3-32B, original prepared token
IDs, metadata-only boundaries, TP support and cache lifecycle.

## Algorithm and agreed differences

1. During independent chunk construction, record F3 on every query head in
   layers 0–15 (the first quarter of 64 layers), excluding the four sink tokens.
   Retain the source's 16-token **local-window denominator**, rather than the
   all-earlier-token denominator described in the upstream document. F3 sums
   and head counts are reduced across TP ranks before publication.
2. Probe the suffix from the first question token through the end of the prompt,
   across all layers and heads. Recompute four initial tokens of each reused
   chunk in temporary gathered KV copies before each probe layer; stored and
   paged context KV remain unchanged. Original `question_positions` stay intact.
3. Use the source's min-Cartesian-tree density calculation and stable top-k
   selection. **The source's equal-boundary inheritance behavior is deliberately
   preserved**, including its discrepancy from recursive ancestor-density
   inheritance. A regression test freezes that behavior.
4. Force the first four tokens of every reused chunk (S4). They count **within**
   `floor(ratio * eligible_context_tokens)`, excluding the exact first chunk.
   Reject a budget smaller than S4; do not enlarge it silently. This differs
   from the upstream code, which adds S4 on top of the percentage budget.
5. A repaired context query reads the exact first chunk, repaired earlier
   chunks, and its own chunk's causal prefix. Suffix queries read the exact first
   chunk, repaired context, and causal suffix. Decode uses native unmasked
   attention over the full cache. The masked paths also handle contiguous
   prefill and multi-step suffixes; numerical audits use the matching mask.

The method is selected by `--method tree-prophetkv`; it does not use the upstream
`UCM_TREE`, `UCM_S4`, `UCM_PROBE_S4`, `UCM_MASK_QUESTION` or `UCM_RMASK` environment
flags. Existing ProphetKV, baseline, router and collection action inventories
retain their behavior. The older marker-inserting preparation and Qwen3-4B
profile/benchmark launchers are not imported.

For the prepared L20 RULER thinking / A800 LongBench v2 cohorts, use the
[tree-only launchers](deployment/TREE_PROPHETKV_L20_A800.md). They reuse manually
copied inputs and bound fresh KV/F3 to one prompt per TP2 pair.

## Single prepared input

Use a fresh output directory. These are invocation examples, not executed GPU
jobs. Set the existing environment to explicit device UUIDs matching `--tp`.

```bash
export PYTHON_BIN=/path/to/python
export CUDA_VISIBLE_DEVICES=GPU-uuid-a,GPU-uuid-b
"$PYTHON_BIN" run.py run --model /path/to/Qwen3-32B \
  --input /path/to/original-prepared.json --output outputs/tree-prophetkv-new \
  --method tree-prophetkv --ratio .15 --tp 2
```

Add `--dry-run` to print the validated configuration without starting an engine.
The single-input path constructs both KV and F3 before measured answer TTFT.

## Persistent collection

Build a fresh setup with `--tree-prophetkv`. This flag becomes part of the setup
identity; existing setups cannot be upgraded in place. A manifest uses the
current `id`/`prepared` format and the existing RULER or LongBench protocol.

```bash
"$PYTHON_BIN" run.py setup --model /path/to/Qwen3-32B \
  --manifest /path/to/manifest.jsonl --output outputs/tree-setup-new \
  --tp 2 --tree-prophetkv

"$PYTHON_BIN" run.py run --setup outputs/tree-setup-new \
  --output outputs/tree-answers-new --method tree-prophetkv --ratio .20
```

F3 records live in `cache/tree-f3/`, keyed by original chunk-token hash and bound
to the setup fingerprint and versioned algorithm. Rank zero publishes them
atomically after all-head TP reduction. Completion requires both committed KV
and valid F3; completion receipts bind the F3 file hashes. Resume computes missing
F3 while preserving complete KV shards and committed F3. Readers reject missing,
corrupt or mismatched metadata and never generate or rewrite it. Ordinary cached
methods may also read the KV from a tree-enabled setup.
Tree-enabled population disables prefix reuse so F3 always sees every local
query row, including when KV is already complete; the persistent store skips
writes to complete shards.

Construction and tree JIT compilation occur outside measured answer TTFT.
Answer TTFT includes S4 probing, tree selection, F3 loading, masked repair and
normal request processing. Native diagnostics retain F3 and global scores for
selection replay, TP agreement, exact budgets and all-layer coverage checks.

## Validation scope

CPU tests cover the source F3 formula and tree behavior, masked attention against
a dense reference, GQA, TP reductions/broadcast, S4 budget boundaries, unchanged
input metadata and probe KV, prefill/decode dispatch, native evidence rejection,
and persistent F3 identity/read-only/resume behavior. GPU correctness, model
accuracy and latency have not been measured for this port. Upstream Qwen3-4B
benchmark numbers are not results for this Qwen3-32B integration.

Validation on 2026-10-24: 111 focused CPU tests passed, including 18 tree-method
tests and retained setup, sweep, original-token, prefill and attention regressions.
Two additional FP32 masked-ragged/GQA cases passed with the actual Triton kernel
under its CPU interpreter. Python syntax and `git diff --check` passed. The
source tree kernel's AST is unchanged.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest discover -s tests -p test_tree_prophetkv.py -v
```
