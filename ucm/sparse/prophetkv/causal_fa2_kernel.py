# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supporst page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
# FA2 softmax candidate from ragged-fa2-softmax-tuning-20261014-v4.
# Default ragged inference uses the measured tile heuristic in attention.py.
# The explicit trajectory opt-in retains this wrapper's original tile default.
# Kernel arithmetic is preserved from the supplied candidate.
import torch
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _ragged_positions_fa2_kernel(
    Q,
    K,
    V,
    Q_Positions,  # Q token positions in original sequence
    sm_scale,
    Q_Start_Loc,
    Q_Seqlen,
    KV_Start_Loc,
    KV_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    Lk: tl.constexpr,
    logit_cap: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """
    Optimized ragged attention kernel with explicit Q positions for position-based causal mask.

    Key optimizations:
    1. Static loop bounds - uses cur_kv_seq_len directly instead of dynamic early-exit
    2. Early block-level exit for invalid blocks
    3. Simplified control flow for better GPU utilization
    4. Precomputed masks for reduced register pressure

    Unlike right-aligned causal mask, this kernel uses the actual positions of Q tokens
    to determine which K tokens each Q can attend to. This is essential for Blend Attention
    where Q tokens are sparsely selected from the original sequence.

    Args:
        Q_Positions: [total_q_len] tensor containing the actual position of each Q token
                     in the original sequence.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    # Load sequence metadata with explicit type conversion
    cur_q_seq_len = tl.load(Q_Seqlen + cur_batch).to(tl.int32)
    cur_kv_seq_len = tl.load(KV_Seqlen + cur_batch).to(tl.int32)
    cur_q_start_index = tl.load(Q_Start_Loc + cur_batch)
    cur_kv_start_index = tl.load(KV_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # Early exit for blocks beyond sequence length
    # This is a simple compile-time-friendly check
    if block_start_loc >= cur_q_seq_len:
        return

    # Initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Precompute masks (reduces redundant computation in loop)
    mask_d = offs_d < Lk
    mask_m = offs_m < cur_q_seq_len

    # Compute Q offset and load Q
    off_q = (
        (cur_q_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q + off_q,
        mask=(mask_m[:, None]) & (mask_d[None, :]),
        other=0.0,
    )

    # Load actual Q positions for this block
    cur_q_pos = tl.load(
        Q_Positions + cur_q_start_index + offs_m,
        mask=mask_m,
        other=0,
    )

    # Precompute K/V pointer offsets
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
    k_ptrs = K + off_k + cur_kv_start_index * stride_kbs
    v_ptrs = V + off_v + cur_kv_start_index * stride_vbs

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Optimization: Calculate the actual maximum K index this block needs to process
    # based on the causal mask and the maximum Q position in this block.
    # For causal attention, we don't need to compute when k_pos > max(cur_q_pos).
    # cur_q_pos has shape [BLOCK_M]
    if IS_CAUSAL:
        # Find the maximum Q position in this block (handling out-of-bounds Q tokens)
        max_q_pos = tl.max(tl.where(mask_m, cur_q_pos, -1))
        # Ensure we process up to max_q_pos, plus any sliding window requirements
        # We need to process up to max_q_pos + 1 tokens (since positions are 0-indexed)
        # Cap it at cur_kv_seq_len
        effective_kv_seq_len = tl.minimum(max_q_pos + 1, cur_kv_seq_len)
    else:
        # If not causal, or if sliding window has backwards component we can't easily prune,
        # we process the full KV sequence.
        effective_kv_seq_len = cur_kv_seq_len

    # Main attention loop with static bounds based on causal pruning
    for start_n in range(0, effective_kv_seq_len, BLOCK_N):
        # Compute K positions and validity mask
        k_pos = start_n + offs_n
        mask_kv = k_pos < cur_kv_seq_len

        # Load K
        k = tl.load(
            k_ptrs,
            mask=(mask_kv[None, :]) & (mask_d[:, None]),
            other=0.0,
        )

        # Compute attention scores
        qk = tl.dot(q, k)
        qk *= sm_scale

        # Apply logit capping
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        # Apply causal/sliding window mask
        if IS_CAUSAL:
            if SLIDING_WINDOW_SIZE > 0:
                valid_mask = (
                    (cur_q_pos[:, None] >= k_pos[None, :])
                    & ((cur_q_pos[:, None] - k_pos[None, :]) < SLIDING_WINDOW_SIZE)
                )
                qk += tl.where(valid_mask, 0.0, float("-inf"))
            else:
                valid_mask = (cur_q_pos[:, None] >= k_pos[None, :])
                qk += tl.where(valid_mask, 0.0, float("-inf"))
        else:
            if SLIDING_WINDOW_SIZE > 0:
                valid_mask = (tl.abs(cur_q_pos[:, None] - k_pos[None, :]) < SLIDING_WINDOW_SIZE)
                qk += tl.where(valid_mask, 0.0, float("-inf"))

        # FA2 online softmax: retain unnormalized FP32 accumulators.
        # Original-position causal mask above is unchanged.
        if USE_EXP2:
            qk = qk * 1.4426950408889634
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        if USE_EXP2:
            p = tl.exp2(qk - m_safe[:, None])
            alpha = tl.exp2(m_i - m_safe)
        else:
            p = tl.exp(qk - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
        l_i_new = alpha * l_i + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # Load V and update accumulator
        v = tl.load(
            v_ptrs,
            mask=(mask_kv[:, None]) & (mask_d[None, :]),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        # Update state
        l_i = l_i_new
        m_i = m_i_new

        # Advance pointers
        k_ptrs += BLOCK_N * stride_kbs
        v_ptrs += BLOCK_N * stride_vbs

    # Normalize once, after all K tiles; all-masked rows return zero.
    acc = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]

    # Store output
    off_o = (
        (cur_q_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :]
    )
    tl.store(
        Out + off_o,
        acc,
        mask=(mask_m[:, None]) & (mask_d[None, :])
    )


def ragged_positions_attention_fwd(
    q,
    k,
    v,
    o,
    q_start_loc,
    q_seq_len,
    kv_start_loc,
    kv_seq_len,
    q_positions,
    max_q_len,
    is_causal=True,
    sm_scale=None,
    logit_cap=0.0,
    sliding_window_size=-1,
    config=(64, 128, 8, 2),
    use_exp2=True,
):
    """
    Ragged attention with explicit Q positions for position-based causal mask.

    This function supports Blend Attention where Q tokens are sparsely selected
    from the original sequence. Each Q token has its actual position in the
    original sequence, and the causal mask is computed based on these positions.

    Args:
        q: Query tensor [total_q, num_heads, head_dim]
        k: Key tensor [total_kv, num_kv_heads, head_dim]
        v: Value tensor [total_kv, num_kv_heads, head_dim]
        o: Output tensor [total_q, num_heads, head_dim]
        q_start_loc: Start location of each batch's Q in the flattened tensor [batch]
        q_seq_len: Number of Q tokens for each batch [batch]
        kv_start_loc: Start location of each batch's KV in the flattened tensor [batch]
        kv_seq_len: Number of KV tokens for each batch [batch]
        q_positions: Actual position of each Q token in original sequence [total_q]
                     Used for position-based causal masking.
        max_q_len: Maximum Q sequence length across batches
        is_causal: Whether to apply causal masking
        sm_scale: Softmax scale factor (default: 1/sqrt(head_dim))
        logit_cap: Logit capping value (0 means no capping)
        sliding_window_size: Sliding window size (-1 means no sliding window)
    """
    Lq, Lk = q.shape[-1], k.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = q_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    max_q_len = max(1, max_q_len)

    BLOCK_M, BLOCK_N, num_warps, num_stages = config

    grid = (batch, head, triton.cdiv(max_q_len, BLOCK_M))

    _ragged_positions_fa2_kernel[grid](
        q,
        k,
        v,
        q_positions,
        sm_scale,
        q_start_loc,
        q_seq_len,
        kv_start_loc,
        kv_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        num_warps=num_warps,
        num_stages=num_stages,
        Lk=Lk,
        logit_cap=logit_cap,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        USE_EXP2=use_exp2,
    )
