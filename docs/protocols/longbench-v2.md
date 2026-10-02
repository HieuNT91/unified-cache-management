# LongBench v2 protocol

LongBench v2 retains its own protocol; RULER's non-thinking template, task output
caps and substring scorer do not apply. Common [invariants](../agents/invariants.md)
remove chunk markers/EOT padding without deleting genuine source/chat tokens.

## Prompt and generation

`scripts/longbench_v2.py` formats the dataset with `scripts/longbench_0shot.txt`
and the Qwen3 native thinking chat template, once. The full question, choices,
instructions and chat tail remain in the suffix; `question_positions` marks the
question. Existing formatter normalization at field boundaries is distinct from
chunking: do not claim byte-for-byte raw-file preservation where the adapter
strips field boundaries. All methods receive identical prepared token IDs.

No-cache baseline has no connector or prefix reuse. ProphetKV receives the same
prompt without text inserted between chunks. Native chat special tokens such as
`im_start`/`im_end` remain intact; their values never determine chunk boundaries
or request phase.

Native generation uses thinking, output cap 16384, temperature 0.6, top-p 0.95,
top-k 20, min-p 0 and request seed 0. The output cap covers thinking, answer and
control tokens together. Keep natural completion behavior and record actual
runtime/model configuration for each experiment.

## Full-window preparation versus the approved local subset

The general adapter supports the dataset's middle truncation path, with complete
question/choices/tail preserved and full formatting/recounting afterward. With
YaRN window 131072 and output reserve 16384, its formatted input limit is 114688.
That protocol remains separate from a full-text fitting subset; do not silently
switch between them or imply that every source row fits a smaller GPU allocation.

For the user-authorized local controls, `scripts/rpkv_longbench_inputs.py` records
all 503 formatted lengths, selects randomly with seed 42 from inputs fitting the
local KV allocation, and preserves the selected full text without middle
truncation. Allocation 65920 leaves at most 49536 input tokens after the 16384
reserve; selection also requires enough context for sparse execution. The recorded
eligible pool has 158 rows. Five-input and later 20-input cohorts use this explicit
selection rule, not a claim about the complete 503-row benchmark.

Retain the source identity, eligible pool, seed, selected source rows, full lengths,
token/template hashes and output reserve in receipts. Never truncate an oversized
row or silently change its seed to make it fit this local full-text protocol.

## Evaluation and reporting

Use the dataset answer format/extraction (the `The correct answer is (A)` form)
and multiple-choice scoring on the answer channel. References never enter
inference. Do not score hidden/thinking text or add an unrelated extraction
heuristic; absent answers remain explicit in the recorded scoring outcome.

Report accuracy, answer TTFT, thinking-content tokens and answer-content tokens;
account for control tokens separately and expose output-cap counts and denominator.
Unfinished thinking has no answer content. Keep construction/readiness/priming
separate from answer TTFT; include online routing overhead when reporting router
total TTFT. Native thinking is expected here, whereas RULER thinking should be zero.

See [control study evidence](../studies/rpkv-controls.md) and
[deployment](../deployment/README.md) for scope-specific records and commands.
