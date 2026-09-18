# ProphetKV UCM/vLLM port

Authorized 2026-09-18. Paper: https://arxiv.org/html/2602.02579v3, Algorithm 1.

Scope: Qwen/Qwen3-4B-Instruct-2507 at cdbee75f17c01a7cc42f958dc650907174af0554,
BF16, TP=1, eager, native non-thinking chat, greedy 128-token outputs.
Each engine limit is its prompt length plus 128, rounded to 64, identically
across methods; GPU memory utilization is 0.93. All patched vLLM Python sources
are hashed and verified, in addition to package versions. Six tasks,
30 source rows/task at 8192/32768/65536 targets; 512/4096-token chunks; baseline,
ProphetKV 10/20/50/90%, corrected CacheBlend K/L1 10/90%. 7,560 requests.

The registered `ProphetKV` sparse method initially supports one Qwen3 prefill
request per worker. UCM 0.3's connector also requires the `Blend` configuration
alias. `runtime.setup` binds the loaded model and makes connector layer alignment
idempotent per request. `runtime.set_request` supplies request ID, boundaries and
question-token positions; no reference answers enter the method. Metadata is
consumed once and is never carried from warmup to measured inference.

Stage I runs the entire fresh suffix through all 36 layers against loaded,
independently cached context, using actual model weights. Only question queries
score context keys. Algorithm 1 softmax covers all context keys, including the
exact first chunk, and excludes suffix keys. Question/head averages and sums
across layers use FP32. Stable descending sort breaks ties by original position.
The tiled scoring and original-position attention kernels are adapted from the
locally pinned QCFuse reference (38795d91); provenance retains its license and
source. This is not QCFuse selection or a SGLang runtime reproduction.

Stage II starts again from original embeddings; selected tokens update both K
and V before layer-0 QKV, with the exact prefix reused and 256 suffix tokens fresh.
The connector loads cache once; Stage I position-aligns each layer once, and Stage
II reuses it without double rotation. Budgets floor(ratio * eligible tokens),
excluding prefix and suffix. On missing cached blocks, dense scheduled execution
is used and the cached measurement is rejected. Decode retains vLLM attention.
Both methods use original-position causal attention. The native FlashAttention
backend is retained when original query positions are exactly a contiguous suffix;
arbitrary compacted queries always use the positional Triton kernel. This removes
avoidable BF16 backend drift for dense and 0/100% paths. Model equivalence compares
all fresh-suffix hidden states at every layer (relative RMS < 0.03), separately
from per-attention FP32 reference checks (absolute/relative tolerance 0.03). CacheBlend retains original
K/L1 arithmetic, topk tie behavior, and layer-1 selection timing (layer-1 QKV is
dense, its attention and FFN are selected). Per-layer counts document that cost.

CPU generation uses local RULER, seed 42 and the pinned tokenizer; 64K sources are
rows 0–29 of the validated 50-row data. IDs use task/target/ordinal/chunk size.
Native chat with the RULER answer prefix is chunked with numbered markers and
64-token block padding. The requested chunk size includes markers and padding.
Question spans exclude assistant markers and answer preambles. Layouts differ
between chunk sizes; each has a separate baseline. This is a prompt-format and
recomputation-budget adaptation to the local UCM connector.

TTFT begins immediately before engine submission and ends on the first
token-bearing engine output. The in-engine probe, cache transfer, alignment,
selection, recomputation and output all occur inside TTFT. Model loading, cache
construction/readiness and warmup are separate. Storage is buffered local warm
cache, not cold-disk storage. Scores use local official RULER substring scoring
and control-character cleanup, and include length-limit counts.

`run_prophetkv.py prepare`, `pilot`, `detach`, `status`, `resume`, and `report`
manage preparation, gated execution and reporting. The full scheduler always
runs numerical checks, model smoke/audits with 0/100% controls, then 8K/512 pilot
(10/task), performance gate, 32K/512 pilot/gate, and 64K/512 pilot/gate. 4096
smoke cases at each length precede the remainder. A rate in 10/20/50 must have
macro score at least baseline minus 0.05 and paired geometric TTFT speedup >1.
Failure stops advancement and produces a diagnostic report. Unchanged pilot
records count toward final results. Source changes invalidate prior measurements;
the scheduler rejects changed snapshots on resume.

GPUs 1–4 only, with UUID checks before each launch. A source prompt retains its
GPU across methods/chunk sizes. Unrelated processes are awaited. Idle dummy jobs
hold a small CUDA allocation and periodic operation; each is released before
benchmark work. These are owned jobs, terminated on stop/completion. Supervisor
is CPU-only, locked and detached with nohup/session separation. Each fresh case
has a 600-second no-progress watchdog and one retry. Caches are bounded to one
sample/GPU and deleted only after the engine process group exits. Shared
installations and historical results are unchanged. Cancellation or failed gates
preserve measured/failed artifacts and produce cleanup receipts.
