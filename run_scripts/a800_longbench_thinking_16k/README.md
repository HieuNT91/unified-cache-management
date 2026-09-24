# Remote A800 LongBench v2 thinking comparison (16K output)

Qwen3-32B BF16, TP=2 per job: physical GPUs 0–1 and 2–3, pinned by UUID after checking the A800 inventory. No cache (UCM and prefix reuse disabled) versus ProphetKV with expansion at 10%, 20%, and 40% recompute. Anchor ratios are 7.5%, 15%, and 30%; max_gap=2, score/window exponents=.5, window_scale=8, min/max window=8/64. These ratios apply to the eligible cached region, excluding the exact first chunk and fresh suffix.

Thinking output: max_new_tokens=16384, temperature=.6, top_p=.95, top_k=20, seed=0. The output budget includes reasoning and the final answer. Only text after the closing `</think>` is scored; unfinished reasoning scores zero. TTFT measures the first generated token, which may be reasoning. Generation time, output-cap counts and unfinished-thinking counts are retained.

Every row of the full 503-row LongBench v2 dataset is considered. Inputs must be at most 65,536 tokens **after chat and chunk formatting**, without truncation. The earlier census found 184 eligible thinking inputs: 736 measurements, 368 per GPU pair. Preparation recomputes eligibility with your checkpoint tokenizer. No RULER tasks run here. All methods share inputs and GPU assignment; method order rotates between pairs.

YaRN factor stays 2. The position table is extended from 65,536 to 81,920 positions using unchanged frequencies and magnitude, preserving the original entries bitwise. This extrapolates past nominal YaRN 2×. All methods use the same 1280-block KV allocation. Chunks are 4096 tokens; the fresh suffix is at least 256 tokens and includes the complete question and choices.

## Run on the remote server

Use the same installed, UCM-patched environment as the non-thinking runner (uc-manager 0.3.0, vLLM 0.9.2, PyTorch 2.7.0, transformers 4.53.2). An ordinary unpatched vLLM installation is insufficient. Use the original unquantized Qwen3-32B checkpoint and full LongBench data.json.

```bash
git pull --ff-only origin prophetkv/import-20260918
export PYTHON_BIN=/absolute/path/to/cacheblend/bin/python
export MODEL_PATH=/absolute/path/to/Qwen3-32B
export LONGBENCH_DATA=/absolute/path/to/LongBench-v2/data.json
export THINKING_RESULT_ROOT="$PWD/.results/a800-qwen3-32b-longbench-thinking-16k"
export THINKING_CACHE_ROOT=/local/scratch/a800-qwen3-32b-longbench-thinking-16k
nvidia-smi -L
bash run_scripts/a800_longbench_thinking_16k/run.sh prepare
bash run_scripts/a800_longbench_thinking_16k/run.sh detach
bash run_scripts/a800_longbench_thinking_16k/run.sh status
```

`prepare` performs CPU preparation, tokenization and provenance checks; `detach` starts inference. Both pairs must be free. The supervisor refuses conflicting compute jobs and duplicate launches. This runner ignores inherited `RESULT_ROOT`/`CACHE_ROOT` and uses the separate `THINKING_*` variables above. Retain those exports in subsequent shells. Existing non-thinking and 8K thinking sources and results are untouched. Use a fresh results directory; prepared 8K records are not compatible with this output budget.

```bash
# Ongoing state and current engine logs
bash run_scripts/a800_longbench_thinking_16k/run.sh status
bash run_scripts/a800_longbench_thinking_16k/run.sh logs 0
bash run_scripts/a800_longbench_thinking_16k/run.sh logs 1
python3 run_scripts/a800_live_results.py --root "$THINKING_RESULT_ROOT"

# After both jobs finish: fully validated aggregate
bash run_scripts/a800_longbench_thinking_16k/run.sh aggregate
cat "$THINKING_RESULT_ROOT/combined/REPORT.md"
```

Each job writes `final/` automatically. Combined output includes summary.csv, raw_records.jsonl, engine_sessions.json and validation.json. Aggregation refuses running jobs; use the live reader for ongoing results. `detach` resumes missing measurements after failure; accepted records are retained. Check `status` and logs first. A silent engine has a 1800-second watchdog and one retry. Normal warmup, committed cache readiness, hit checks, all-layer TP selection audits and retirement barriers remain mandatory; no separate smoke run is scheduled.

The runner sources are hash-pinned at preparation: do not edit this directory while a prepared experiment is running. CPU regression suite (no model inference):

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
  -m unittest discover -s run_scripts/a800_longbench_thinking_16k -p test_cpu.py -v
```

Remote GPU execution must still be validated on the A800 server; local checks do not establish GPU runtime success.
