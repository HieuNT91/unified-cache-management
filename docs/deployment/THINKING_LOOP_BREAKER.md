# Qwen3-32B thinking loop breaker

New answer requests through `runner.generation.generate` with a validated
`ruler_thinking_budget` policy automatically enable the loop breaker. This covers
RULER's separate thinking/answer budgets and collection v3's RULER/LongBench v2
15,000-thinking / 16,384-total policy on vLLM 0.9.2. It uses the existing worker
logits and scheduler hooks; no vLLM upgrade or site-packages patch is needed.
Non-thinking requests, one-token probes, populate requests and legacy requests
without a managed thinking policy retain their existing behavior.

The detector follows vLLM's
[`check_sequence_repetition` and `_has_repeating_pattern`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/utils.py)
and [`RepetitionDetectionParams`](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py),
inspected on 2026-11-05. After every thinking content token, it checks whether
the tail contains three consecutive, identical blocks of 8 through 128 token
IDs, trying block lengths in ascending order. Matching is exact; no text
normalization, approximate similarity or separated occurrences qualify. A shorter
primitive period can still qualify: 24 identical tokens form three 8-token
blocks, just as in vLLM. History is bounded to 384 thinking content tokens.

On a match, the next sampling step forces Qwen3's native `</think>` token using
the delimiter ID validated by the existing policy. The existing newline and
deferred answer prefix then run once, followed by ordinary answer sampling in
the same request and KV state. The scheduler does not terminate on detection.
The detector observes neither prompt tokens nor answer tokens. Prepared token
IDs, sampling temperature/top-p/top-k/seed and output budgets are unchanged.

Completed results record:

- `thinking_loop_breaker`: the versioned runtime configuration (`8`, `128`, `3`).
- `thinking_broken: true`, `thinking_closure: "repetition"` on intervention.
- `thinking_break_pattern_size` and `thinking_break_token_index` (number of
  thinking content tokens consumed at detection).
- `stop_reason: "thinking_broken"` once the answer finishes.
- Native `finish_reason: "stop"` on answer EOS, or `"length"` if the answer
  reaches its cap. A broken thinking phase does not imply a failed generation
  or a correct/nonempty answer; normal answer scoring still applies.

The forced close token is a control token and is included in `forced_tokens`.
It does not count as generated answer content or as reaching the thinking cap.
If repetition first appears exactly at that cap, both the intervention and the
actual cap count are retained. Answer latency is still measured at the first
answer content token. Collection result JSON, portable JSON outcomes, and v3
reports retain the intervention fields/counts. Compact NPZ keeps its existing
feature/outcome schema; per-request details remain in result JSON.

This is a versioned runtime intervention recorded separately from the prepared
budget policy. Use fresh result paths for comparisons with earlier decoding.
Existing prepared policies and accepted result files are not rewritten; older
records without intervention metadata remain readable with their original
semantics. Direct `BudgetState(policy)` reconstruction keeps historical behavior;
pass the recorded `thinking_loop_breaker` configuration to replay a new request.

CPU coverage is in `tests/test_thinking_loop.py`, with retained budget and
collection coverage in `tests/test_ruler_thinking.py` and
`tests/test_collection_v3.py`. Tests use synthetic tokens and mocked sampling;
they do not establish GPU throughput or real Qwen3 answer quality.
