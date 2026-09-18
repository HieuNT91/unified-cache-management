# ProphetKV UCM/vLLM port: RULER100 and LongBench v2

New independent study, superseding the cancelled older ProphetKV sweeps.
Physical GPUs 1–4 only, always checked with nvidia-smi -L and restricted by UUID.
The supervisor and preparation expose no CUDA devices. Idle GPUs retain owned
placeholders as previously requested; unrelated processes are never terminated.

Qwen3-4B-Instruct-2507 pinned local revision cdbee75f17c01a7cc42f958dc650907174af0554,
BF16, TP1, eager, native non-thinking chat, greedy, maximum 128 generated tokens.
RULER: seven tasks, 100 newly generated seed-42 rows per task at 65536 target.
LongBench v2: all 184 of 503 documents whose pinned-tokenizer context count is
strictly below 65536, without truncation or a dataset length-category restriction.
Baseline plus ProphetKV 10% and 20%: 884 prompts, 2652 measured requests.
One measurement per prompt/method. Fresh engines, fixed per-prompt GPU assignment,
rotated method order, bounded one-sample cache per GPU, 600-second watchdog, one retry.

4096-token chunks include numbered markers and block padding, final chunk may be
shorter. Every method sees identical frozen token IDs. The first chunk is exact
prefix reuse. Fresh suffix=max(256, tokens from start of question through assistant
prefix), preserving every MCQ choice. Score question/choices only, excluding output
instructions, assistant prefix and reference answers. Stage I uses all 36 loaded
model layers; context-key softmax averaged over query tokens/heads and summed in
FP32 across layers. Deterministic stable token ties. Stage II restarts from embeddings
and writes both selected K and V before layer0, preserving unselected cached entries.
Budget denominator excludes exact prefix and fresh suffix. Original-position causal
attention and once-per-request cached-key alignment apply to both stages.

This study privately generalizes the earlier fixed 256-token probe to variable
query suffixes. Existing implementations, sources, measured artifacts and shared
installations are preserved. Numerical CPU/GPU tests and five-case all-layer smoke
checks on longest RULER prompt, longest LongBench prompt, and longest LongBench
query precede measurements. 0% and 100% are smoke controls; 100% fresh-suffix hidden
states must match full computation within 0.03 relative RMS at all 36 layers.
Failures halt the scheduler; failed artifacts are preserved.

TTFT is submission to first token-bearing engine output; online probing, transfers,
selection and recomputation are inside TTFT. Load/cache-build/warmup are separate.
Storage is buffered local warm cache. Require complete cache blocks/shards and file
sizes, actual measured hits, clean logs, immutable sources/prompts and selection
and per-layer compute diagnostics. RULER official substring cleanup/scoring;
LongBench official zero-shot answer extraction. Report capped and unparseable outputs.
TTFT speedup = paired mean full-prefill TTFT / paired mean ProphetKV TTFT. Geometric
speedup is a separately labeled supplemental metric. Report per-task/domain accuracy,
paired differences, relative improvement and 1000-draw bootstrap confidence intervals.

Operations (CPU-only environment):
CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/prophetkv_full/run_prophetkv.py prepare
CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python benchmarks/prophetkv_full/run_prophetkv.py detach
Use status/report for progress; resume only after verifying no live supervisor.
SIGTERM the verified supervisor to stop. Cleanup waits for owned engine groups to exit.
