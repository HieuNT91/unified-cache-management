# Required invariants

These details implement the [root rules](../../AGENTS.md). Protocol-specific
requirements are in [RULER](../protocols/ruler.md) and
[LongBench v2](../protocols/longbench-v2.md).

## Tokens and metadata

- Baseline, sparse answers, applicable probes and dense fallback use identical
  prepared `token_ids`. Do not insert `[Context chunk N]`, EOT padding, newlines,
  filler or replacement separators. Preserve genuine source/chat special tokens;
  never delete tokens globally by value.
- For input length `L` and first question position `q`, suffix start is
  `max(0, 64 * floor(min(q, L - 256) / 64))`. Consecutive context chunks are
  block-aligned, at most 4096 tokens; the final context chunk may be smaller.
  No padding is permitted. The suffix includes the complete question, chat tail
  and answer prefix; `question_positions` marks only the question.
- Boundaries start at zero, increase strictly, cover the prompt without overlap
  or gaps, and end at `L`. Verify token hash, layout and request identity in setup,
  single-input, sweep and router paths. Short inputs remain valid for baseline;
  sparse reads require at least two context chunks and must otherwise fail clearly.
- Send `populate`/`read`, request ID, hashes, boundaries and question positions in
  `SamplingParams.extra_args['rpkv_layout']` and propagate to scheduler/workers.
  Missing, corrupt or mismatched metadata is an error. No `chunk_end_token_id`
  dependency, last-token phase inference or delimiter fallback is allowed.
- Version identities: prompt `rpkv-original-tokens-v1`, cache
  `rpkv-local-chunk-kv-v1`. Record tokenizer/template/source/generator identities.
  Old inputs require original-source rebuilding or validated source mapping/hash;
  never relabel old padded inputs, policies or results as compatible.

## KV identity and attention

- Chunk hashes depend on content/local history and the versioned model/cache
  namespace. Build at local position zero; reuse identical stored shards at
  separate global destination slots. Apply delta RoPE only to loaded keys, once
  per layer; never mutate stored KV or rotate already aligned keys again.
- Keep setup identity separate from execution UUIDs. Selection ratio/layers and
  answer output cap do not change construction identity; token/layout/model and
  relevant runtime changes do.
- Preserve exact prefix, all-layer cache alignment, original-position causal
  attention and one global token selection across multi-step prefill. Reserve
  full prompt KV slots; never compact into sparse-position slots.
- ProphetKV uses ascending FP32 accumulation over all 64 layers, divided by 64,
  then native TP averaging. Selective modes use explicit zero-based layers or
  last-N layers. Preserve all-context-key normalization, query weighting, exact
  floor budgets and deterministic ascending-position ties.
- Bound prefill steps and suffix query/projection tiles to at most 16384 tokens.
  Do not overwrite repaired KV while scoring. Empty repair steps advance without
  forward/sampling; generate only after the full prompt. Reject unsupported
  preemption and clear state on finish, cancel and retirement.
- Baseline disables connector and prefix caching. Dense fallback must audit
  zero reused KV, zero UCM lookup/load/store and all native layers, and match the
  baseline for the same prompt/decoding configuration.

## Readiness, ownership and results

- Build under exclusive ownership; persistent readers hold shared locks. Wait
  for committed shards on every rank, publish setup completion atomically, and
  fail on missing shards. Resume only incomplete construction; keep complete KV.
- Readers must not populate, write warmup KV, delete shared KV or silently rebuild.
  Keep all-rank/all-layer replay, immutable-cache checks and operation retirement.
- Freeze references/scoring as evaluation metadata; references never enter model
  input or policy features. Score RULER newly generated text with its upstream
  scorer; use LongBench's answer channel under its separate protocol.
- Publish live aggregation only for validated, retired records. Final aggregation
  requires the complete declared scope and owned engine exit. Keep task/prompt
  denominators, null predictions, output caps and content/control counts explicit.
- Never weaken checks, edit accepted records or substitute results to force a pass.
