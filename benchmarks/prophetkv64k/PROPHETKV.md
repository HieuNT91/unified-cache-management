# ProphetKV 64K chunk comparison on GPUs 3 and 4

Independent user-authorized sweep: Qwen3-4B-Instruct-2507; native non-thinking chat;
BF16, TP1, eager, greedy max 128 output tokens. Tasks vt, cwe, niah_single_1,
niah_multikey_1, qa_1; original validated 64K source rows 0–9. 50 source prompts,
four chunk layouts (512,4096,8192,12000), baseline and ProphetKV20/30/40/50:
1000 fresh-engine measurements. The 12000 nominal layout uses 11999 content and
marker tokens plus padding to 12032 (64-token block alignment). Other full chunks
match nominal lengths. Record exact prompt lengths, boundaries and hashes.

Correctness gates: CPU/GPU tests, then all five methods plus 0% and 100% controls
on one CWE sample per layout, 36-layer writes/preservation/causality audits,
and per-layer suffix-hidden-state relative RMS <0.03 for 100% vs baseline.
The earlier 32K performance gate does not apply to this new explicitly requested
sweep. Failed correctness checks stop advancement. No automatic accuracy gate.

Same frozen UCM/vLLM method semantics as prior ProphetKV port. Online probe,
transfers, alignment, selection and recomputation remain inside TTFT; cache build,
model loading and warmup are reported separately. Buffered local warm cache.
Recomputation rates exclude exact first chunk and fresh 256-token suffix, so
larger chunks also increase exact-prefix reuse; report this budget difference.

Only physical GPU3 UUID GPU-f7a26c8e-4455-7a74-3e75-f02c21d7c9e5 and GPU4 UUID
GPU-d52293e0-0963-52cc-f251-0827ef68b02f. Check nvidia-smi -L before every launch;
CPU-only preparation/supervisor, GPU0 always forbidden. Other experiments remain
untouched. Use one worker per GPU with fixed task/row affinity across all layouts.
Owned small idle placeholder allocations are released before inference and on exit.
600-second log-progress watchdog; one retry; lock; fresh engines; bounded caches.

Run CUDA_VISIBLE_DEVICES='' .envs/cacheblend/bin/python
benchmarks/prophetkv64k/run_prophetkv.py prepare, then detach. Resume uses the same
driver's resume command, only after verifying no live supervisor. Status/report
commands are CPU-only. Detached nohup, separate session, stdin /dev/null,
persistent supervisor.log. Sources/checkpoint/runtime/private UCM/prompts pinned.
Final certificate requires all1000 valid measurements, reporting and owned cleanup.
