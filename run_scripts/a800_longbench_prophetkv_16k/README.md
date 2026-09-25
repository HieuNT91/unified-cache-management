# Original ProphetKV, LongBench v2 thinking, 16K output

Run original all-layer ProphetKV at 10%, 20%, 40%, 50%, and 60%. No expansion and no fresh baseline inference. The runtime sums FP32 question attention over all 64 layers, averages over the TP group, and selects floor(eligible_tokens * ratio) highest scores with ascending-position ties. The exact first chunk remains reused and the complete question suffix (at least 256 tokens) remains fresh.

Qwen3-32B BF16 eager, TP2 on remote physical A800 pairs 0–1 and 2–3, pinned by UUID. YaRN factor2; 81920-position allocation and 1280 KV blocks; the original 65536 entries are preserved bitwise and additional positions use unchanged frequencies. Thinking uses 16384 output tokens including reasoning, temperature .6, top_p .95, top_k20, seed0. Only final text after closing </think> is scored; unfinished reasoning scores zero.

Preparation scans all 503 LongBench v2 rows and accepts all fully chat/chunk-formatted inputs <=65536 tokens without truncation. The earlier census found184: five methods give920 measurements,460 per pair. The remote tokenizer census determines the actual count. Assignment is identical across methods, with rotated method order between pairs. No RULER or separate smoke runs.

Use the existing UCM-patched environment (uc-manager0.3.0, vLLM0.9.2, torch2.7.0, transformers4.53.2), original checkpoint, and full dataset. From the repository root:

```bash
git pull --ff-only origin prophetkv/import-20260918
export PYTHON_BIN=/absolute/path/to/cacheblend/bin/python
export MODEL_PATH=/absolute/path/to/Qwen3-32B
export LONGBENCH_DATA=/absolute/path/to/LongBench-v2/data.json
export PROPHETKV_RESULT_ROOT="$PWD/.results/a800-qwen3-32b-longbench-prophetkv-16k"
export PROPHETKV_CACHE_ROOT=/local/scratch/a800-qwen3-32b-longbench-prophetkv-16k
nvidia-smi -L
bash run_scripts/a800_longbench_prophetkv_16k/run.sh prepare
bash run_scripts/a800_longbench_prophetkv_16k/run.sh detach
bash run_scripts/a800_longbench_prophetkv_16k/run.sh status
```

This isolated runner ignores old THINKING_RESULT_ROOT/THINKING_CACHE_ROOT variables. Prior expansion sources and results stay untouched. Do not point the new run at an old result directory. Current preparation includes a new CPU tokenization census and freezes the private runtime/source hashes. Do not edit prepared sources while running.

```bash
bash run_scripts/a800_longbench_prophetkv_16k/run.sh logs 0
bash run_scripts/a800_longbench_prophetkv_16k/run.sh logs 1
python3 run_scripts/a800_live_results.py --root "$PROPHETKV_RESULT_ROOT"
# After both workers finish:
bash run_scripts/a800_longbench_prophetkv_16k/run.sh aggregate
cat "$PROPHETKV_RESULT_ROOT/combined/REPORT.md"
```

Each job writes final outputs automatically; aggregate produces combined summary.csv, raw_records.jsonl, engine_sessions.json and validation.json. Baseline speedup columns are empty because this experiment does not import earlier baseline records. Earlier baselines remain available in their original directory; timing cohorts differ.

Committed warmup/cache readiness, measured hits, independent Python ranking replay, all64-layer selected-set checks, TP rank agreement, retirement and bounded caches remain mandatory. TTFT excludes cache construction, priming and export; wall time includes them and thinking decode. The no-log watchdog is1800 seconds with one retry. CPU tests:

```bash
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" \
  -m unittest discover -s run_scripts/a800_longbench_prophetkv_16k -p test_cpu.py -v
```

No GPU execution was performed locally for this remote launcher.
