# AdaTreeV1.1 agent rules

Work only in this branch/worktree. Preserve the primary checkout, other worktrees,
prepared datasets, accepted results and caches from existing experiments.
Do not read or modify `.trash/`. Keep data, weights, local `.env`, cache and results
out of Git. `config/.env.example` is the only environment template.

This branch runs Qwen3-32B RULER64K thinking and LongBench v2 with no-cache
and fixed AdaTree at chunk4096/2048. Do not add fitting or configuration sweeps.
Preserve original prepared token IDs, input/model/protocol hashes, independent
chunk KV construction, all-layer alignment, cache immutability and retirement.
Chunk boundaries are metadata, never inserted tokens. Prompt formatting, tokenization and generation behavior match AdaTreeV1.
Git revision is not an execution or preparation gate; retain content/identity checks.

Generation uses8192 thinking /10000 total tokens and loop breaker32..128 x4.
Both methods use identical inputs, seeds and generation policy. Report committed
results through `status.py`; default full, optional exact same-input intersection.

GPU work needs user authorization and explicit device UUIDs. Local GPU0 is
excluded. Before authorized process intervention, verify PID/start/command and
GPU ownership. Never duplicate supervisors. Do not push or restart old jobs
without a request. Run focused CPU checks and authorized real-input smoke tests.
