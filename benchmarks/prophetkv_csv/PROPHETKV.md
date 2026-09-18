# KV repair CSV supplemental requests

200 measured requests on GPUs3/4, queued until the running1200-request64K sweep
completes and releases its lock. Do not preempt it or unrelated workloads.
8K and32K,512-token chunks,10 samples each of niah_single_1,niah_multivalue,vt,cwe.
NIAH-single has a fresh baseline plus ProphetKV20/30/40 (80 requests).
Other tasks run only missing ProphetKV30/40 (120 requests), against byte-identical
frozen prompts from the earlier pilot. Historical baseline/P20 are not rerun.
No accuracy gate, but CPU/GPU numerical tests and6 audited smoke methods including
0/100% controls at each context gate measurement. No inference implementation change.

Qwen3-4B-Instruct-2507, non-thinking native chat, greedy128 tokens, BF16/TP1/eager.
Runtime/private implementation/model/input/source hashes pinned. TTFT includes
online probing/transfers/selection/recomputation. Buffered local warm cache.
Historical and supplemental timing phases differ, and some original baseline GPU
assignments differ from the new GPU3/4 assignments. Report this comparison limit.

Commands: CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python
benchmarks/prophetkv_csv/run_prophetkv.py prepare|detach|resume|status|report.
Only resume after checking no supervisor is live. UUID visibility,600-second
watchdog,one retry, bounded caches and owned idle placeholders preserved.
Results: .results/prophetkv-csv-backfill-20260918/.
CSV updater: benchmarks/fill_kv_repair_report.py; full group requires all10 rows.
