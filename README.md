> **rpkv branch:** use [the original RULER 64K and metadata-boundary guide](scripts/RPKV.md).
> Historical exact-64000/padded-input and old-policy commands below do not apply to this branch.

# ProphetKV · Qwen3-32B

A focused runtime with four modes: no-cache baseline, original all-layer
ProphetKV, selective ProphetKV with configurable scoring layers, and a frozen
attention router. Results and model weights are not included.

Native attention router deployment starts at [scripts/A800_ROUTER.md](scripts/A800_ROUTER.md).
It adds `--method router --router-policy PATH --router-id router1` for prepared
inputs and persistent setups. The three learned policies remain frozen; the A800
workflow regenerates the 1300-row test cohort and validates all 9100 answers.

All modes use Qwen3-32B BF16, YaRN **4×** with an original 32768-position window
and a **131072-token** total window. All modes use **chunked prefill**,
with at most **16384 scheduled tokens per step**. This budget does not reduce
the context window or the full-prompt KV allocation. Prefix caching is disabled;
baseline also omits the UCM connector entirely. Normal autoregressive KV caching
within a request remains enabled.

The default is TP4, eager execution, native thinking and a 16384-token output
cap for the fixed methods. The router uses non-thinking256 and independent
KV65920/1031-block capacity beneath the same 131072-position YaRN table.
Input plus output must fit the window; preparation rejects oversized inputs
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
memory for BF16 weights, full-context KV, bounded prefill activations, and
suffix-probe state.
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

Each experiment output directory must be new. Legacy `--input` runs are
foreground, one prompt and one engine per invocation. Use `--tp` to match the number of visible UUIDs.
`--num-layers N` uses the **last N** decoder layers, e.g. 5 means 59–63.
`--layers` accepts an explicit set instead; the runtime sorts IDs ascending.
Both forms are mutually exclusive. With all 64 layers, selective scoring
uses the same layer mean as original ProphetKV.

## Prepare a persistent collection once

A setup holds frozen tokens and independent context KV for the entire collection.
Create a JSONL manifest with unique string IDs. Paths are relative to the manifest:

```jsonl
{"id":"raw-example","context":"context.txt","question":"question.txt","subtask":"retrieval","references":["target"],"scoring":"exact_match"}
{"id":"imported-example","prepared":"inputs/prompt.json","subtask":"multiple-choice","references":["B"],"scoring":"choice"}
```

Raw entries use the tokenizer/chat template once. Imported JSON must have the
existing `prepare` schema, including model-config hash, token IDs, boundaries,
question positions, thinking flag and output cap. Imported tokens and boundaries
are preserved exactly. All samples must have at least two block-aligned context
chunks, a common delimiter, and the complete question in a fresh suffix of at
least 256 tokens. Duplicate independent chunks share storage.

```bash
# CPU configuration and disk estimate only; writes nothing and starts no engine.
bash run.sh setup --model "$MODEL_PATH" --manifest prompts.jsonl \
  --output .cache/setups/collection --tp 4 --dry-run

# After explicitly assigning CUDA_VISIBLE_DEVICES to suitable GPU UUIDs:
bash run.sh setup --model "$MODEL_PATH" --manifest prompts.jsonl \
  --output .cache/setups/collection --tp 4

bash run.sh run --setup .cache/setups/collection \
  --method prophetkv --ratio .1 --output outputs/prophet10
bash run.sh run --setup .cache/setups/collection \
  --method selective_prophetkv --num-layers 5 --ratio .2 --output outputs/selective20
bash run.sh run --setup .cache/setups/collection \
  --method baseline --output outputs/baseline

# Optional subset, in the requested order, with a new generation cap:
bash run.sh run --setup .cache/setups/collection --prompt-ids imported-example \
  --method selective_prophetkv --layers 45 48 50 56 58 --ratio .2 \
  --max-output-tokens 256 --output outputs/subset --dry-run
```

`setup --kv-chunk-size` defaults to **4096** and accepts multiples of 64 from 64
through **16384**. It controls raw formatting and the maximum imported context
chunk length. `--context-length` defaults to **131072** and limits the *formatted
input*. Independently, input plus output must fit the model's 131072-token window.
Neither limit truncates inputs or changes the 16384-token scheduling budget.
Raw entries default to thinking with a 16384-token output cap; use `--no-thinking`
or `--max-output-tokens` when preparing them. Imports retain their generation
settings unless the output cap is explicitly overridden.

Runs inherit model and TP from the setup. `--model` may point to a relocated,
content-identical checkpoint; `--tp`, if supplied, must match. Changing the method,
ratio, either selective-layer interface, or a fitting output cap reuses the same
KV keys. Model weights/configuration, tokenizer/template, frozen tokens/boundaries,
chunk size, context limit, suffix policy, logical TP layout, dtype, RoPE, runtime
sources and cache format are fingerprinted. Initial preparation hashes local
checkpoint files; unchanged file size/mtime/ctime/inode/device stamps allow later
calls to reuse those hashes. Relocated or changed files are hashed again.

Setup prints expected KV payload disk usage before GPU construction; allow extra
space for metadata and temporary writes. One engine constructs incomplete chunks,
checking every TP shard's committed filename and byte size, waiting for transfers,
and retiring each request. The exclusive build lock excludes readers and other
builders. Repeating the matching setup command validates the inventory and returns
without tokenization or an engine when complete. After interruption, repeat that
command to fill only incomplete chunks; complete shards remain intact. Incompatible
preparation settings require a new setup directory. Live orphan build workers block
reuse until they exit. Completion is atomically published after shard fsync and
verified retirement/worker exit.

Each experiment holds a shared lock, runs selected prompts sequentially with one
engine, and uses only read-only KV operations. Warmup uses the first frozen prompt;
priming and measured requests get independent execution IDs. Alignment, scores,
masks and repaired KV stay in request-local GPU memory. No population, persistent
warmup writes, cache deletion, or reconstruction fallback occurs. Missing/incomplete
KV fails with instructions to rerun setup. Baseline reads frozen tokens, creates no
connector, and never accesses KV shards. Legacy single-input runs still construct
and retire their own temporary cache.

Setup artifacts are `setup.json`, `samples/`, `inventory.json`, `progress.json`
(one-time construction timings and resumable attempts), `cache/`, and
`complete.json`. Collection outputs contain numbered per-prompt directories with
`result.json` and `diagnostics.json`, plus `summary.json`. Results retain prompt IDs,
setup fingerprint, generation/validation/retirement timings and a reference to
one-time construction timings; loading and warmup appear once in the summary.
TTFT excludes preparation, loading, warmup, priming, readiness and validation.

### Live and final aggregation

Every `run --setup` writes two aggregate reports in its experiment output directory:

- `live_aggregation.json`, `.csv`, `.md`: an initial empty report, refreshed after
  each validated and retired measured prompt. Caught failures/interruption preserve
  completed results and mark the live report `failed`/`interrupted`.
- `final_aggregation.json`, `.csv`, `.md`: published only after all selected prompts
  succeed and the engine has shut down. Failed or partial runs have no final report.

Each report includes a row per **subtask** and an **overall** row: completed/expected
counts, mean/median TTFT in seconds, accuracy percent and scored/unscored counts,
mean thinking length and mean answer length in tokens. JSON/CSV also include token
totals, separate control/output token counts, output-cap counts, and unfinished
thinking counts. Overall metrics are **prompt-weighted**; they are not an unweighted
average of subtask means. Empty subtask rows remain visible while work is pending.
The JSON file is the authoritative atomic snapshot. To inspect progress:

```bash
cat outputs/prophet10/live_aggregation.md
# Once the run finishes:
cat outputs/prophet10/final_aggregation.md
```

Add optional `subtask`, `references` (a string or list of strings), and `scoring`
to each setup manifest row, as in the examples above. Imported prepared JSON can
also carry these fields; `task`/`answers` are accepted as imported aliases. Explicit
manifest fields override imported metadata. These labels and reference answers are
frozen separately from KV identity and never enter inference or token selection.
Repeated setup rejects changed evaluation metadata; use a different setup directory
for a revised evaluation. Each run saves its selected evaluation in `evaluation.json`.

| Per-prompt `scoring` | Score in [0, 1] |
|---|---|
| `exact_match` (default) | Final answer equals any reference after case-folding and whitespace normalization; punctuation is retained. |
| `choice` | An extracted A/B/C/D choice equals a reference; explicit answer statements take precedence, ambiguous choices fail. |
| `reference_coverage` | Fraction of reference strings present in the final answer, ignoring case and normalizing whitespace. |
| `contains_any` | Any reference string occurs in the normalized final answer. |

Accuracy is **100 × mean per-prompt score** among completed prompts with references.
Unannotated prompts retain TTFT and token statistics, use subtask `unlabeled` if no
label is supplied, and show accuracy **N/A** rather than an invented score. These
are explicit scoring rules, not automatic benchmark-specific evaluators.

Accuracy uses only the final answer, excluding reasoning. Thinking and answer
lengths count the actual generated token IDs, including whitespace tokens and
excluding `<think>`, `</think>` and other special/control tokens. Their sum plus
control tokens equals total generated output tokens. Native thinking that starts
in the prompt is handled; a response stopped before `</think>` has no answer tokens
and is marked unfinished. TTFT remains time to the first generated token (including
thinking), with reporting/scoring outside the measured generation interval.
Per-prompt `result.json` retains the extracted answer, score, scoring rule, references,
subtask and all token counts. Warmup and priming never enter aggregates.

Persistent-setup GPU validation remains pending assigned devices. The following
opt-in matrix prepares at least two supplied prompts once, repeats setup to verify
reuse, runs two ProphetKV ratios, both selective interfaces, baseline and a subset,
and compares outputs and selection diagnostics with fresh single-input construction.
It hashes setup artifacts after each run to verify survival through retirement.
This is an expensive validation sweep; input tokens are preserved and generation
is made deterministic with a one-token cap.

```bash
"$PYTHON_BIN" tests/gpu_persistent_setup.py --model "$MODEL_PATH" \
  --inputs inputs/prompt-a.json inputs/prompt-b.json \
  --output outputs/setup-gpu-validation \
  --gpu-uuids GPU-uuid1 GPU-uuid2 GPU-uuid3 GPU-uuid4
```

## Algorithm and outputs

The runtime constructs independent context chunks (4096-token default),
reuses the exact first chunk, and leaves the complete question in a fresh
suffix of at least 256 tokens. Scoring averages attention across question
queries, heads, chosen layers and TP ranks, with softmax over all context
keys. The budget selects `floor(eligible_tokens × ratio)` positions, breaking
ties by ascending position. Selective probing propagates the suffix through
preceding layers; choosing fewer scoring layers does not remove model layers
from inference. All 64 cached layers are aligned before recomputation.

Cached requests reserve the full prompt's KV slots before loading. The complete
context loads and aligns once; one global probe reads the full saved question
suffix before any selected context token is repaired. Its scores and selection
persist across every prefill step. Long suffix projection and attention-query
work is tiled into batches of at most 16384, using query-weighted score means
and full-context normalization. This still requires retaining suffix state.

Each step repairs the selected positions and fresh suffix positions within its
original range. The attention key range ends at that step's endpoint, with
original-position causality. Empty repair ranges advance scheduler progress
without a model forward; partial prefills never sample a token. A final one-token
prefill is handled as prefill. Continuations retain full block mappings and never
reload or rotate cached KV. Unsupported preemption fails explicitly; completion,
cancellation and retirement release request state.

Each run verifies committed cache shards before reuse, replays selection on
every rank, checks exact coverage across all steps for all 64 recomputed layer
sets (including empty sets, with no missing or duplicate positions), and retires
transfers before retiring requests. Legacy runs delete their temporary cache;
collection runs preserve setup KV. `result.json` contains predictions, token IDs,
finish reason, input hash, scoring layers, configuration and separate timings.
TTFT includes all measured prefill steps and probing/recomputation, but excludes
model loading, warmup, offline cache construction and priming. Router total TTFT
also includes its routing-required export, features, retirement, decision and TP
synchronization; post-answer validation/export remain outside TTFT.
`diagnostics.json` retains global scores, per-layer positions, and scheduled/
recomputed counts for every prefill step. Collection runs add
the per-prompt scoring and aggregation described above; legacy single-input runs
retain their existing output schema.

## CPU checks

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m unittest discover -s tests -v
```

Persistent setup tests cover raw/imported collections, deduplication, subsets,
compatibility, output-cap reuse, interrupted builds, incomplete TP shards, locks,
read-only stores, baseline isolation, sequential engine reuse and CLI dry runs.
Aggregation tests cover per-prompt scorers, native/truncated thinking, control-token
accounting, subtask/overall weighting, live updates and final-publication gates.
The chunked-prefill regressions cover configuration isolation, prompts below/at/above 16K, final one-token
prefill, zero-selection ranges, 0/100% masks, suffix crossings, both selective
interfaces, TP replay, real worker/connector state transitions, cancellation,
complete cache hits, one-time alignment, bounded probe tiling and query weighting.
A deterministic multilayer causal reference compares chunked repair with a single
forward. CPU tests also exercise the patched runner's empty/no-sampling path.

A focused GPU check on 2026-09-26 completed **six validated measurements** on
two NIAH single-3 prompts of exactly 64000 tokens. All three modes retrieved both
UUIDs correctly. Mean TTFT was 28.581s (baseline), 8.655s (all-layer ProphetKV20)
and 8.138s (selective ProphetKV20, layers 45/48/50/56/58). It used BF16 TP4,
16384-token prefill steps and an explicitly approved **test-only 64256-token KV
allocation**; the complete YaRN4 RoPE table remained 131072 positions. All ranks
passed selection/coverage/retirement checks and all experiment processes exited.
Local artifacts: `outputs/niah-single3-64000-2samples-20260926/comparison.md`.
TTFT excludes construction, priming and cleanup; two samples are a limited check.

Full-window GPU validation and 0/100% GPU controls remain **pending**. After
explicitly assigning suitable devices, this opt-in matrix runs eight independent
engines on one prepared multi-step prompt (all
three modes, 0/20/100% repair, and both selective interfaces). It preserves all
input tokens and uses a one-token deterministic output cap. Each run checks all
ranks, exact coverage, cache retirement and the 16K scheduling limit; the matrix
also compares the 100% repair first token against baseline.

```bash
"$PYTHON_BIN" tests/gpu_chunked_prefill.py --model "$MODEL_PATH" \
  --input inputs/prompt.json --output outputs/gpu-validation \
  --gpu-uuids GPU-uuid1 GPU-uuid2 GPU-uuid3 GPU-uuid4
```

Use a prompt with over 32768 tokens after the first context chunk and enough
cached context for an empty 0% repair range. For a GPU final-one-token check,
use a prepared prompt whose total length minus first-chunk length is congruent
to 1 modulo 16384. The same ordinary `run` commands and prepared input format
remain compatible.

`ucm/sparse/blend/` and the Blend connector are internal reuse infrastructure,
not an exposed CacheBlend experiment. `provenance.json` identifies extracted
source files before focused edits. License notices are preserved in `licenses/`.

For the all-503 LongBench v2 sweep on the supplied eight-A800 server, see
[scripts/A800_LONGBENCH.md](scripts/A800_LONGBENCH.md). It includes remote paths,
Git worktree setup, the existing server dataset path, offline CPU preparation,
UUID-pinned launch commands, the repository middle-truncation adapter, disk
estimates and runtime scenarios. No archive transfer or dataset download is needed. This launcher uses `run.sh sweep`: construct one
prompt's KV, reuse it across all 12 cached configurations, then retire and delete
it before the next prompt. Two TP4 groups retain at most about56GiB of KV payload
in total. No collection-wide KV setup is needed. `CACHE_ROOT` controls temporary
storage; live/final per-method reports combine both prompt shards. Its `longbench_v2`
scoring rule uses the official answer extractor. See `tests/gpu_temporary_sweep.py`
for the opt-in fresh-construction equivalence check (GPU validation pending).

For the L20 server's eight-task RULER experiment (100 samples/task, exactly64000
formatted input tokens, thinking/output16384, baseline plus nine ratios for each
ProphetKV mode), use [scripts/L20_RULER_64000.md](scripts/L20_RULER_64000.md).
It provides Git-based deployment, official asset downloads, CPU preparation,
UUID-pinned launch commands, and combined live/final per-task reports for15200
measurements. Temporary KV is built once per prompt and deleted after all18 cached
configurations. No environment installation or archive transfer is required.

For an interrupted A800/L20 sweep with completed baselines, see
[ProphetKV-only continuation](scripts/RESUME_PROPHETKV.md). Existing launchers
can stop the owned sweep and resume only missing vanilla results while retaining
completed baseline, ProphetKV and discontinued selective artifacts.

A800/L20 launchers automatically load server paths and settings from a local
Git-ignored `.env`. Copy `.env.example` and keep the original run paths when
resuming. See [server configuration and update commands](scripts/SERVER_ENV.md).

Use `counts` on either launcher for a quick saved-result count. Resume prints these
counts before checking inputs. Set `RESUME_VALIDATION=fast` in `.env` to trust prior
saved diagnostic validation and skip its repeated replay/hashing; new answers
still receive full validation. Use `full` for the complete saved-diagnostic audit.

Reusable RULER collection and task-independent tree routers are documented in
[scripts/RULER_CORPUS_COLLECTION.md](scripts/RULER_CORPUS_COLLECTION.md),
[scripts/RULER_CORPUS_TRAINING.md](scripts/RULER_CORPUS_TRAINING.md), and
[scripts/TREE_INFERENCE.md](scripts/TREE_INFERENCE.md). The new corpus defaults to
nocache plus all64 1/20/40%, supports partial collection and versioned pooled
training, and exports individual trees for RULER or LongBench v2 inference on
A800/L20. This is separate from the frozen 1/20/50% router workflow. CPU verification
is separate from GPU acceptance; no GPU launch is implied by these interfaces.
