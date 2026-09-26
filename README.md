# ProphetKV · Qwen3-32B

A focused runtime with three modes: no-cache baseline, original all-layer
ProphetKV, and selective ProphetKV with configurable attention-scoring layers.
No experiment schedulers, datasets, results or model weights are included.

All modes use Qwen3-32B BF16, YaRN **4×** with an original 32768-position window
and a **131072-token** total window. Chunked prefill is **disabled**. Prefix
caching is disabled; baseline also omits the UCM connector entirely. Normal
autoregressive KV caching within a request remains enabled.

The default is TP4, eager execution, native thinking and a 16384-token output
cap. Input plus output must fit the window; preparation rejects oversized inputs
without truncation. `prepare --no-thinking --max-output-tokens 256` selects
non-thinking generation. Every method reads the same prepared token IDs.

## Environment

Use Linux/CUDA and Python 3.10 with an installed CUDA build of **uc-manager
0.3.0**, **vLLM 0.9.2**, **torch 2.7.0**, and **transformers 4.53.2**.
This tree supplies the Python runtime and process-local vLLM patches; the
installed UCM distribution supplies its ABI-compatible `ucmpcstore` and
`ucmmetrics` native libraries. It does not edit the installed environment.
This is not a standalone replacement for the native UCM build system.

Use an unquantized local Qwen3-32B checkpoint and GPUs with enough aggregate
memory for BF16 weights, full-context KV, and unchunked prefill activations.
There is no quantization or CPU offload. A full 128K run needs substantially
more memory than weights alone; GPU execution of this extracted branch has
not yet been validated.

```bash
export PYTHON_BIN=/path/to/environment/bin/python
export MODEL_PATH=/path/to/Qwen3-32B
# Supply your own long context.txt and question.txt (at least two context chunks).
bash run.sh prepare --model "$MODEL_PATH" --context context.txt \
  --question question.txt --output inputs/prompt.json

# Inspect configuration without starting an engine:
bash run.sh run --model "$MODEL_PATH" --input inputs/prompt.json \
  --method selective_prophetkv --num-layers 5 --output outputs/selective --dry-run

# Set actual UUIDs for your intended devices before running.
export CUDA_VISIBLE_DEVICES=GPU-uuid1,GPU-uuid2,GPU-uuid3,GPU-uuid4
bash run.sh run --model "$MODEL_PATH" --input inputs/prompt.json \
  --method baseline --output outputs/baseline
bash run.sh run --model "$MODEL_PATH" --input inputs/prompt.json \
  --method prophetkv --ratio 0.2 --output outputs/prophetkv
bash run.sh run --model "$MODEL_PATH" --input inputs/prompt.json \
  --method selective_prophetkv --ratio 0.2 --layers 45 48 50 56 58 \
  --output outputs/selective
```

Each output directory must be new. Runs are foreground, one prompt and one
engine per invocation. Use `--tp` to match the number of visible UUIDs.
`--num-layers N` uses the **last N** decoder layers, e.g. 5 means 59–63.
`--layers` accepts an explicit set instead; the runtime sorts IDs ascending.
Both forms are mutually exclusive. With all 64 layers, selective scoring
uses the same layer mean as original ProphetKV.

## Algorithm and outputs

The runtime constructs independent context chunks of at most 4096 tokens,
reuses the exact first chunk, and leaves the complete question in a fresh
suffix of at least 256 tokens. Scoring averages attention across question
queries, heads, chosen layers and TP ranks, with softmax over all context
keys. The budget selects `floor(eligible_tokens × ratio)` positions, breaking
ties by ascending position. Selective probing propagates the suffix through
preceding layers; choosing fewer scoring layers does not remove model layers
from inference. All 64 cached layers are aligned before recomputation.

Each run verifies committed cache shards before reuse, replays selection on
every rank, checks all 64 recomputed layer sets, and retires transfers before
deleting its own cache files. `result.json` contains predictions, token IDs,
finish reason, input hash, scoring layers, configuration and separate timings.
TTFT includes measured probing/recomputation, but excludes model loading,
warmup, offline cache construction, priming, validation and export.
`diagnostics.json` retains scores and selected positions. Task-specific accuracy
evaluation is intentionally left to the caller's dataset/evaluator.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s tests -v
```

Tests cover configuration isolation, layer validation, actual probe averaging
and stopping with a mocked model, cache alignment, dense attention equivalence,
floor budgets/ties, prompt validation, and YaRN delta rotation scaling.
They do not replace a GPU integration run.

`ucm/sparse/blend/` and the Blend connector are internal reuse infrastructure,
not an exposed CacheBlend experiment. `provenance.json` identifies extracted
source files before focused edits. License notices are preserved in `licenses/`.
