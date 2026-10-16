"""Read-only greedy draft readers, ported from traj4f@99fe9a8."""
import torch
from .runtime import project
from .selection import context_importance


def greedy_token(lm, hidden):
    """vLLM gathers logits only on TP rank zero; broadcast its greedy choice."""
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    logits = lm.compute_logits(hidden, None)
    token = torch.empty(1, dtype=torch.long, device=hidden.device)
    if group.rank_in_group == 0:
        if logits is None:
            raise RuntimeError('Missing draft logits on TP rank zero')
        token[0] = logits[-1].argmax()
    if group.world_size > 1:
        token = group.broadcast(token, src=0)
    return int(token.item())


@torch.inference_mode()
def draft_readers(sparse, context, slots, hidden, residual, suffix_kv, b, drafts):
    """D greedy draft tokens decoded over the stale cache (nothing is written to the paged cache). Returns their
    attention rows over the context [D, b[-2]] (mean over layers and heads, softmax over context keys) and ids.

    Context attention runs on the paged cache (FlashAttention with the request's block table, no gather) and is
    merged by log-sum-exp with exact attention over the few fresh suffix/draft keys; the rows are computed after
    decoding with one key gather per layer."""
    from vllm.vllm_flash_attn import flash_attn_with_kvcache
    model = sparse.model
    h, _ = model.norm(hidden[-1:], residual[-1:])
    token = greedy_token(sparse.lm, h)
    device = hidden.device; L = len(model.layers)
    attn0 = model.layers[0].self_attn
    groups = attn0.num_heads // attn0.num_kv_heads; scale = attn0.head_dim ** -0.5
    table = sparse.attn_metadata.block_table[0][:b[-2] // 64][None].to(torch.int32)
    seqlens = torch.tensor([b[-2]], device=device, dtype=torch.int32)
    own = [[] for _ in range(L)]; queries = [[] for _ in range(L)]; tokens = []
    for j in range(drafts):
        tokens.append(token)
        x = model.get_input_embeddings(torch.tensor([token], device=device)); r = None
        pos = torch.tensor([b[-1] + j], device=device)
        for i, layer in enumerate(model.layers):
            cache = context.no_compile_layers[f'model.layers.{i}.self_attn.attn'].kv_cache[context.virtual_engine]
            q, k, v, r = project(layer, x, r, pos)
            queries[i].append(q); own[i].append((k, v))
            out_c, lse_c = flash_attn_with_kvcache(q[None], cache[0], cache[1], cache_seqlens=seqlens,
                block_table=table, softmax_scale=scale, causal=False, return_softmax_lse=True)
            sk, sv = suffix_kv[i]
            keys = torch.cat([sk] + [a for a, _ in own[i]]).repeat_interleave(groups, 1).float()
            vals = torch.cat([sv] + [c for _, c in own[i]]).repeat_interleave(groups, 1).float()
            logits = torch.einsum('hd,khd->hk', q[0].float(), keys) * scale
            lse_l = torch.logsumexp(logits, -1)
            out_l = torch.einsum('hk,khd->hd', (logits - lse_l[:, None]).exp(), vals)
            lse_c = lse_c.reshape(-1).float()
            lse = torch.logaddexp(lse_c, lse_l)
            out = out_c.reshape(-1, attn0.head_dim).float() * (lse_c - lse).exp()[:, None] + out_l * (lse_l - lse).exp()[:, None]
            o, _ = layer.self_attn.o_proj(out.to(q.dtype).reshape(1, -1))
            o, r = layer.post_attention_layernorm(o, r)
            x = layer.mlp(o)
        h, _ = model.norm(x, r)
        token = greedy_token(sparse.lm, h)
    rows = torch.zeros(drafts, b[-2], device=device, dtype=torch.float32)
    for i in range(L):
        cache = context.no_compile_layers[f'model.layers.{i}.self_attn.attn'].kv_cache[context.virtual_engine]
        ck = cache[0, slots // 64, slots % 64]
        for j in range(drafts):
            rows[j].add_(context_importance(queries[i][j], ck), alpha=1.0 / L)
        del ck
    return rows, tokens


