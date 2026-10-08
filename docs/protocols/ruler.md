# RULER 64K protocol

This is the default non-thinking `rpkv` protocol. Historical exact-64000/padded/thinking guides
are retained only for their original experiments. Follow [invariants](../agents/invariants.md)
for the common token/layout/cache contract.

Pinned upstream: [NVIDIA/RULER c3f5e3b](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a).
The defaults are all 13 tasks, 500 samples per task and seed 42. Each task is
produced by one upstream generator invocation; sharding happens after the batch.
Task arguments come from upstream `scripts/synthetic.yaml`. `max_seq_length=65536`
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

## Preparation, provenance and scoring

Use `scripts/ruler.py` for the implementation and CLI, or import `scripts.ruler`
from Python. See [README preparation commands](../../README.md#prepare-and-check-on-cpu)
and [CPU regression tests](../agents/testing.md).

The 13 tasks are `niah_single_1/2/3`, `niah_multikey_1/2/3`,
`niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, `qa_2`.
Use upstream task parameters and templates, including FWE alpha 2.0. Generate a
complete seeded batch for each task before sharding; do not reseed individual
samples. A smaller authorized cohort changes batch size explicitly, not its
budgeting or formatting rules.

Retain the full source text and complete formatted prompt. Do not enforce a
63,000/63,360 source cap or exactly 64,000 tokens. No filler, whitespace deletion,
newline padding, or truncation to hit a target length is allowed. Preserve raw
batches on post-tokenization overflow and report an error; do not silently select
a replacement seed or edit content.

The original answer prefix appears exactly once in input/prefill, in the open
assistant turn, and stays in the fresh suffix. References are scoring-only.
Score newly generated text directly with the pinned preprocessing and substring
metrics, without adding the prefix back or applying custom extraction. Record
per-task scores, null predictions and upstream rounding along with raw outputs.
Apply each task's output cap to every answer method. One-token probes apply only
where explicitly required by a router or diagnostic protocol; new fixed controls
have zero independent probes.

The evaluation identity is `nvidia-ruler-c3f5e3b-qwen3-nonthinking-v1`.
Record prompt/cache/evaluation versions, full token and template hashes, generator
arguments/seed/batch size, source commit/assets, tokenizer/model identity, YaRN,
backend and hardware. Incompatible old cache/results/policies require a separate
protocol and may not be silently accepted or refitted.

## Separate thinking TP2 protocol

The [thinking RULER collection](../deployment/L20_RULER_THINKING_DATA.md) reuses
the original source generation with30 rows/task and seed42, but renders the native
thinking prompt, defers the original prefix, and scores only new answer content
after a separately budgeted thinking phase. Its distinct evaluation identity and
82,304-token execution profile must not be applied to existing non-thinking runs.

The [L40 extension](../deployment/L40_RULER_THINKING_DATA.md) supports a fresh
200-row source batch at seed42, with a verified30-row prefix from a thinking run with completed controls.
Parent probes may finish separately; a complete portable union requires both
probe exports. Only source ordinals30..199 receive new inference. A separate
extension receipt pins the parent and prefix comparison; the portable200-row
union preserves both runs' measurement provenance and original sample hashes.
