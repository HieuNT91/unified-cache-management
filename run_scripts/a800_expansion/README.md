# Qwen3-32B expansion study on four A800 80GB GPUs

This independent launcher runs two concurrent BF16 TP=2 engines: job 0 uses
physical GPUs **4,5**, job 1 uses **6,7**. Devices are resolved and pinned by UUID
before every engine launch; all other GPUs stay hidden. Preparation and reporting
are CPU-only. No dummy work or separate qualification/smoke is launched.

## Scope

* RULER: `niah_multikey_1`, `niah_multikey_2`, `niah_multikey_3`,
  `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, `qa_2`.
  Exactly 100 new seeded prompts/task, non-thinking, 256 maximum output tokens.
* LongBench v2: census of the **full 503-row data.json**, with every eligible row
  in **non-thinking mode only**. Eligibility is <=65,536 tokens after chat
  formatting and numbered, padded chunks.
  No inference prompt is truncated. An eligible prompt unsupported by the chunk
  layout fails preparation instead of being silently excluded.
* Six methods: `baseline`, `expansion-10`, `expansion-20`, `expansion-30`,
  `expansion-40`, `expansion-50`. These are **ProphetKV with expansion**, with
  anchor ratios .075/.15/.225/.30/.375. Other expansion settings retain their
  established defaults: gap 2, score exponent .5, window scale 8, window exponent
  .5, window bounds 8–64.
* All methods: 4,096-token chunks; fresh suffix max(256, full question-tail size);
  identical prompts and physical GPU pair for each sample. RULER source-row
  parity and LongBench eligible-source parity assign samples to the two pairs.
  Method order rotates between pairs.
* All requests use **non-thinking, greedy decoding, seed 0, 256 maximum new tokens**.
* YaRN factor **2**, original window 32,768. Every engine reserves **65,792**
  positions (65,536 input + 256 output), with a common **1,028-block** KV allocation
  per rank. A private worker appends 256 table positions using unchanged factor-2
  frequencies and magnitude; all original 65,536 entries are verified bitwise
  unchanged. This slightly exceeds the nominal factor-2 window.

RULER generation starts at a 64,512-token source target, reserving chat/chunk
overhead and 256 output tokens. If formatting exceeds the input limit, the
generator reruns with a lower target. It never truncates generated documents or
needles. `ruler-generation.json` records the final targets and actual lengths.

Total measurements: **6,000 + 6 × eligible LongBench non-thinking count**.
The preparation output gives the exact counts from the supplied dataset/tokenizer.

The prior CPU census of all 503 rows with the same Qwen3-32B tokenizer and
unchanged non-thinking formatter found **184 eligible non-thinking samples**,
with formatted lengths 10,240–64,832 and fresh suffixes up to 875 tokens. The
revised scope therefore yields **7,104 measurements**, **3,552 per GPU pair**.
Remote preparation recomputes eligibility from your source/tokenizer.
Dataset SHA-256: `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`.

Planning estimate: **24–48 hours** after inference starts, with both TP2 pairs
running and fast local NVMe. Dataset preparation is additional. This is a rough
extrapolation from local 32B timings; A800 throughput is unmeasured. It includes
cache construction, readiness, priming, decoding, validation and export rather
than multiplying measured TTFT alone. Removing thinking removes the long decode
component; adding QA2 adds 600 measurements.

## Why this still uses TP2

At 65,792 positions, the original BF16 checkpoint uses 61.024 GiB for weights
and the BF16 KV allocation uses 16.0625 GiB at TP1: **77.087 GiB** before
activations, workspaces, CUDA and ProphetKV buffers. In this runner, dense prefill
is not chunked or MLP-tiled: at 65,536 input tokens, Qwen3's merged gate/up MLP
projection alone produces a **6.25 GiB** intermediate tensor (65,536 × 51,200 ×
2 bytes). These allocations already sum to **83.34 GiB**, before further costs.

Thus the current BF16/full-prefill TP1 path does not fit an 80GB A800. TP2 gives
roughly 38.54 GiB per GPU for weights plus KV. TP1 would require another memory
adaptation, quantization, offloading or a shorter input limit; none is applied by
this launcher. ProphetKV recomputation ratios do not reduce the full KV allocation.

## Remote setup and launch

Sync the current checkout, including this new directory and the vendored RULER
assets, to the remote server. The older `run_scripts/job_*.sh` files launch a
different experiment; use the entry point below for this study.

The remote Python environment must already contain the coupled runtime:
UCM 0.3.0, **UCM/Qwen3-patched vLLM 0.9.2**, PyTorch 2.7.0, Transformers 4.53.2,
plus RULER's PyYAML, wonderwords, NLTK and `punkt`/`punkt_tab` assets. Stock pip
vLLM is insufficient. No installed packages are modified by this launcher.
Use an original complete BF16 Qwen/Qwen3-32B checkpoint, including tokenizer.
CUDA toolkit/library environment variables should match that prepared runtime.
QA2 uses HotpotQA; preparation extracts the packaged `hotpotqa.json.zip`
automatically when the JSON is absent. No dataset download is needed for QA2.

```bash
cd /path/to/unified-cache-management
export PYTHON_BIN=/path/to/prepared-env/bin/python
export MODEL_PATH=/path/to/Qwen3-32B
export LONGBENCH_DATA=/path/to/LongBench-v2/data.json
export RESULT_ROOT=/path/to/results/qwen3-32b-expansion-a800-nonthinking
export CACHE_ROOT=/path/to/local-nvme/qwen3-32b-expansion-a800-nonthinking
# Optional if the vendored RULER tree is not used:
# export RULER_ROOT=/path/to/RULER

nvidia-smi -L
bash run_scripts/a800_expansion/run.sh plan
bash run_scripts/a800_expansion/run.sh prepare
bash run_scripts/a800_expansion/run.sh detach
bash run_scripts/a800_expansion/run.sh status
```

`prepare` generates/finalizes data and validates CPU/runtime/device settings for
both jobs sequentially; it does not run inference. `detach` starts both jobs
and returns. Supervisors use nohup, separate sessions and `/dev/null` stdin.
Check `supervisor_live: true` for both after the command returns; SSH can then
disconnect. No trailing `&` or assistant polling is needed.

If the LongBench file is missing, download it on a networked machine before
preparation, or copy your existing full dataset:

```bash
mkdir -p "$(dirname -- "$LONGBENCH_DATA")"
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    curl --noproxy '*' -fL --retry 3 \
    https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json \
    -o "$LONGBENCH_DATA"
```

Use a **new empty result root** for this revised scope; an earlier both-mode
preparation cannot be resumed with these changed sources/settings. Sources, private UCM, inputs,
tokenizer/configuration and datasets are hash-pinned during preparation. Completed
records and their diagnostics/logs are hash-pinned on acceptance. Changing these
after preparation fails verification. Resuming requires the same code and paths.

Allow **at least 50 GiB total local cache space**, plus approximately **200 GiB
free for results and exported all-layer diagnostics**. Exact results size depends
on prompt lengths and selected-token counts. Sample caches are bounded and deleted
only after all TP workers retire requests and full-size block commits are ready;
engine cleanup verifies process-group exit. Long-lived engines run one method
at a time; normally there are 12 measured engine starts total.

## Results, monitoring and resume

```bash
# Explicit status check whenever desired:
bash run_scripts/a800_expansion/run.sh status

# View one job's current engine log (Ctrl-C stops the viewer only):
bash run_scripts/a800_expansion/run.sh logs 0

# After both jobs finish, validate and aggregate:
bash run_scripts/a800_expansion/run.sh aggregate
cat "$RESULT_ROOT/combined/REPORT.md"

# Stop both owned jobs; accepted measurements are retained:
bash run_scripts/a800_expansion/run.sh stop
bash run_scripts/a800_expansion/run.sh status

# Once stopped/exited, resume missing work:
bash run_scripts/a800_expansion/run.sh detach
```

Append `0` or `1` to prepare/detach/status/stop/report to operate on one job.
`run.sh report` refreshes per-job reports when both are stopped. Aggregation
requires both supervisors/engine groups to exit and every planned measurement
to validate. For an explicitly incomplete report after stopping, use
`run.sh aggregate --partial`; it writes `combined-partial/`.

Each job also automatically writes `job-N/final/` when its full matrix completes.
The combined output contains:

* `summary.csv`: per-task/mode/method accuracy, mean/median TTFT, paired mean TTFT
  speedup, generation latency, token counts and length-limited counts; also an aggregate RULER table.
* `REPORT.md`: readable table and protocol caveats.
* `raw_records.jsonl`: predictions, token IDs, references, timings and provenance.
* `engine_sessions.json`: model-loading/warmup times and RoPE/warmup receipts.
* `validation.json`: validated/expected counts and completeness.

TTFT is time to the first emitted token. Cache building, priming, readiness,
model loading, export and retirement are excluded from TTFT and recorded
separately. Wall-clock estimates include these costs plus decode time.

## Validation status

CPU checks cover restricted GPU pairs, extended RoPE entries, globally averaged
TP selection, variable fresh suffixes, expansion reference masks, full per-rank
record validation, tamper rejection, warmup readiness and non-thinking scope enforcement.
Shell syntax and CPU plan commands are checked. This revised TP2 expansion
configuration has **not been executed on A800 hardware in this workspace**.

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" -m unittest discover -s run_scripts/a800_expansion -p test_cpu.py -v
```
