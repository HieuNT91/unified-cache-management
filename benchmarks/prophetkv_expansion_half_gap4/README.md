# Independent expansion half-budget, gap4, output256 run

Qwen3-4B-Instruct-2507 BF16 TP1 eager, native RoPE, non-thinking, greedy256.
Seven fresh methods: no cache and expansion10/20/30/40/50/60%, anchor ratios
5/10/15/20/25/30%, max_gap4; other expansion geometry stays at defaults.
700 RULER prompts: all100 rows each of CWE, multikey1/2/3, QA1, multivalue,
multiquery from the preserved64K-target source datasets. LongBench v2 includes
all fully formatted prompts strictly below65536 tokens, no truncation.
Its question and choices remain entirely fresh: suffix=max(256, query tail).
This is implemented only in this experiment's private runtime. The common
engine allocation accommodates the longest unchanged RULER prompt +256 output.

`run.sh prepare|verify|detach|status` uses the CPU-only external runtime.
`data.py prepare` freezes inputs first. `checks.py` checks CPU configurations;
`checks.py --cuda` requires physical GPU1 pinned by UUID and compares selectors.
There are no model qualification/smoke requests. Engines use GPUs1–4 by UUID;
GPU0 and dummy workloads are forbidden. Each sample stays on the same GPU for
all seven methods, with rotated method order across GPUs. The supervisor and
CPU reporter use separate locked nohup sessions. Confirm current identities
and states before any launch; never duplicate them or start old schedulers.

Persistent workers retain cache readiness, full cache-hit checks, exact saved-score
reference masks, all36-layer selected sets and explicit request retirement.
Warmup waits for both full-size committed cache blocks before cleanup. Cache
storage is bounded to one request/GPU and cleaned only after retirement/exit.
A900-second progress watchdog and one retry protect each engine session.
Only missing or invalid records rerun. Previous results are hash-pinned and
read-only. Final reporting requires all validated records and engine exit,
and writes `prophetkv_expansion_half_gap4_results.txt`, raw records and CSVs.
