"""All-layer probing using the loaded Qwen3 model and once-aligned paged KV."""
import os
import time
import torch
from .selection import context_importance, select, select_units, LayerAlignment


def delta_rotation_table(table):
    """Remove only YaRN's common magnitude multiplier from delta rotations."""
    if (not isinstance(table, torch.Tensor) or table.ndim != 2 or
            not table.shape[0] or not table.shape[1] or table.shape[1] % 2):
        raise RuntimeError('Expected an initialized two-dimensional RoPE cos/sin cache')
    amplitude = table[0, :table.shape[-1] // 2].float().mean()
    if not torch.isfinite(amplitude) or amplitude <= 0:
        raise RuntimeError('Invalid RoPE magnitude scale')
    return (table.float() / amplitude).to(table.dtype)


def global_selection(scores, prefix, ratio, units=()):
    from vllm.distributed import tensor_model_parallel_all_reduce, get_tp_group
    group = get_tp_group()
    scores = tensor_model_parallel_all_reduce(scores) / group.world_size
    eligible = scores[prefix:]
    selected = select_units(eligible, prefix, ratio, units) if units else select(eligible, prefix, ratio)
    if group.world_size > 1 and selected.numel():
        selected = group.broadcast(selected, src=0)
    return eligible, selected


def project(layer,hidden,residual,positions):
    if residual is None:
        residual=hidden
        hidden=layer.input_layernorm(hidden)
    else:
        hidden,residual=layer.input_layernorm(hidden,residual)
    a=layer.self_attn
    qkv,_=a.qkv_proj(hidden)
    q,k,v=qkv.split([a.q_size,a.kv_size,a.kv_size],dim=-1)
    q=a.q_norm(q.reshape(-1,a.num_heads,a.head_dim)).reshape(-1,a.q_size)
    k=a.k_norm(k.reshape(-1,a.num_kv_heads,a.head_dim)).reshape(-1,a.kv_size)
    q,k=a.rotary_emb(positions,q,k)
    return q.reshape(-1,a.num_heads,a.head_dim),k.reshape(-1,a.num_kv_heads,a.head_dim),v.reshape(-1,a.num_kv_heads,a.head_dim),residual


def setup(worker):
    from ucm.sparse.state import get_ucm_sparse,has_ucm_sparse
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    if not has_ucm_sparse():return {'baseline':True}
    sparse=get_ucm_sparse()
    model=worker.model_runner.model.model
    if (model.__class__.__name__!='Qwen3Model' or
            len(model.layers)!=worker.vllm_config.model_config.hf_config.num_hidden_layers):
        raise RuntimeError('Model must be an unpartitioned Qwen3 decoder (TP supported; PP unsupported)')
    sparse.model=model
    group=get_kv_transfer_group()
    connector=getattr(group,"connector",group)
    sparse.connector=connector
    # YaRN embeds a magnitude scale in Q/K. Delta rotation must not apply it
    # twice to keys that were already rotated during chunk construction.
    if not hasattr(connector, 'prophet_delta_normalized'):
        # vLLM 0.9.2 does not always call the connector's model setup hook.
        # Register the loaded model before deriving the separate delta table.
        if getattr(connector, 'cos_sin_cache', None) is None:
            connector.setup_model(worker.model_runner.model)
        connector.cos_sin_cache = delta_rotation_table(connector.cos_sin_cache)
        connector.prophet_delta_normalized = True
    if hasattr(connector,'prophet_aligned'):return {'installed':True}
    guard=LayerAlignment()
    connector.prophet_aligned=guard.layers
    wait=connector.wait_for_layer_load
    def wait_once(name):
        guard.apply(name,wait)
    connector.wait_for_layer_load=wait_once
    return {'installed':True,'layers':len(model.layers)}


def set_request(worker,request_id,boundaries,question_positions,units=(),drafts=0,fix=False):
    from ucm.sparse.state import get_ucm_sparse
    from .selection import RequestMetadata
    sparse=get_ucm_sparse()
    if drafts and not hasattr(sparse,'lm'):
        sparse.lm=worker.model_runner.model   # Qwen3ForCausalLM: compute_logits for the greedy drafts
    sparse.request_state.arm(RequestMetadata(request_id,tuple(boundaries),tuple(question_positions),tuple(units),int(drafts),
                                            fix=bool(fix)))


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
    token = int(sparse.lm.compute_logits(h, None)[-1].argmax())
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
        token = int(sparse.lm.compute_logits(h, None)[-1].argmax())
    rows = torch.zeros(drafts, b[-2], device=device, dtype=torch.float32)
    for i in range(L):
        cache = context.no_compile_layers[f'model.layers.{i}.self_attn.attn'].kv_cache[context.virtual_engine]
        ck = cache[0, slots // 64, slots % 64]
        for j in range(drafts):
            rows[j].add_(context_importance(queries[i][j], ck), alpha=1.0 / L)
        del ck
    return rows, tokens


@torch.inference_mode()
def probe(sparse,positions,embeddings,query_budget=16384):
    if not 0 < query_budget <= 16384:
        raise ValueError('Probe query batches must be at most 16384 tokens')
    from vllm.forward_context import get_forward_context
    from torch.nn.attention.bias import causal_lower_right
    context=get_forward_context()
    meta=sparse.request
    b=meta.boundaries
    suffix_tokens=b[-1]-b[-2]
    device=positions.device
    qp=torch.tensor(meta.question_positions,device=device)-b[-2]
    suffix_positions=positions[-suffix_tokens:]
    if not torch.equal(suffix_positions,torch.arange(b[-2],b[-1],device=device)):
        raise RuntimeError('Fresh suffix positions changed')
    hidden=embeddings[-suffix_tokens:].clone();residual=None
    scores=torch.zeros(b[-2],device=device,dtype=torch.float32)
    # The drafts are decoded from the LAST prompt token, so that row -- not whichever tokens the probe reads -- is
    # the one that stands one step before draft_1 in the trajectory. One extra query row per scoring layer.
    last_scores=torch.zeros_like(scores) if (meta.fix and meta.drafts) else None
    blocks=sparse.attn_metadata.block_table[0].long()
    cp=torch.arange(b[-2],device=device)
    slots=blocks[cp//64]*64+cp%64
    started=time.perf_counter()
    scoring_layers = sparse.scoring_layers
    capture = getattr(sparse, 'router_capture', False)
    captured = []
    drafts = meta.drafts
    if drafts and (sparse.method != 'prophetkv' or list(scoring_layers) != list(range(len(sparse.model.layers)))):
        raise RuntimeError('Draft readers need the all-layer ProphetKV probe')
    suffix_kv = []
    for i,layer in enumerate(sparse.model.layers):
        name=f'model.layers.{i}.self_attn.attn'
        sparse.connector.wait_for_layer_load(name)
        cache=context.no_compile_layers[name].kv_cache[context.virtual_engine]
        ck=cache[0,slots//64,slots%64];cv=cache[1,slots//64,slots%64]
        # Project the complete saved suffix in bounded batches. No context KV
        # is changed by the probe; attention uses all context keys per query.
        projected = [project(layer, hidden[a:a+query_budget],
            None if residual is None else residual[a:a+query_budget],
            suffix_positions[a:a+query_budget])
            for a in range(0, suffix_tokens, query_budget)]
        q,k,v,residual = (torch.cat([part[j] for part in projected]) for j in range(4))
        del projected
        if i in scoring_layers:
            # Tile means must be weighted by their question-query counts.
            layer_scores=torch.zeros_like(scores)
            for a in range(0, len(qp), query_budget):
                indices = qp[a:a+query_budget]
                layer_scores.add_(context_importance(q[indices],ck), alpha=len(indices)/len(qp))
            scores.add_(layer_scores)
            if last_scores is not None:
                last_scores.add_(context_importance(q[suffix_tokens-1:suffix_tokens],ck))
            if capture:
                captured.append(layer_scores.detach().clone())
            del layer_scores
        if sparse.method == "selective_prophetkv" and i == scoring_layers[-1]:
            del ck,cv,q,k,v
            break
        if drafts:
            suffix_kv.append((k,v))
        full_k=torch.cat((ck,k));full_v=torch.cat((cv,v))
        next_hidden, next_residual = [], []
        for a in range(0, suffix_tokens, query_budget):
            z = min(a + query_budget, suffix_tokens)
            key_end = len(ck) + z
            output=torch.nn.functional.scaled_dot_product_attention(q[a:z].transpose(0,1)[None],
                full_k[:key_end].transpose(0,1)[None],full_v[:key_end].transpose(0,1)[None],
                attn_mask=causal_lower_right(z-a,key_end),enable_gqa=True,dropout_p=0.)
            h,_=layer.self_attn.o_proj(output[0].transpose(0,1).reshape(z-a,-1))
            h,r=layer.post_attention_layernorm(h,residual[a:z])
            next_hidden.append(layer.mlp(h));next_residual.append(r)
        hidden=torch.cat(next_hidden);residual=torch.cat(next_residual)
        del ck,cv,full_k,full_v,output,next_hidden,next_residual
        if not drafts:
            del q,k,v
    # All remaining cached layers must be loaded and rotated before recomputation.
    for i in range(len(sparse.model.layers)):
        sparse.connector.wait_for_layer_load(f"model.layers.{i}.self_attn.attn")
    torch.cuda.synchronize()
    # Each rank owns an equal number of Q heads. Average local head means
    # across the TP group before ranking, so every rank repairs the same tokens.
    scores.div_(len(scoring_layers))
    if last_scores is not None:
        last_scores.div_(len(scoring_layers))
    local_mean = scores.detach().clone() if capture else None
    extra = {}
    if drafts:
        rows, tokens = draft_readers(sparse, context, slots, hidden, residual, suffix_kv, b, drafts)
        del suffix_kv
        from vllm.distributed import tensor_model_parallel_all_reduce, get_tp_group
        rows = tensor_model_parallel_all_reduce(rows) / get_tp_group().world_size
        scores = tensor_model_parallel_all_reduce(scores) / get_tp_group().world_size
        if last_scores is not None:
            last_scores = tensor_model_parallel_all_reduce(last_scores) / get_tp_group().world_size
        from .trajectory import gated_selection
        # fix: the true last-token row heads the trajectory and the probe ranks the non-copy branch; the null is
        # length-unbiased and the chain price is relative to the unary mass (UCM_TRAJ_LAMREL, default 1.0).
        head = scores if last_scores is None else last_scores
        pr = None if last_scores is None else scores
        lamrel = float(os.environ.get('UCM_TRAJ_LAMREL', '1.0')) if meta.fix else 0.0
        selected, z, copying = gated_selection(torch.cat((head[None], rows)), b[1], sparse.ratio,
                                               probe=pr, unbiased=bool(meta.fix), lam_rel=lamrel)
        eligible = scores[b[1]:]
        extra = dict(selection_method='copy_gated_trajectory', drafts=drafts, draft_tokens=tokens,
                     copy_z=float(z), copying=bool(copying), copy_z_threshold=2.0,
                     trajectory_lambda=(16384.0 if not lamrel else None), trajectory_lam_rel=lamrel,
                     traj_true_last=last_scores is not None, traj_unbiased=bool(meta.fix))
    else:
        eligible, selected = global_selection(scores, b[1], sparse.ratio, *((meta.units,) if meta.units else ()))
    sparse.selection_diagnostics.append(dict(kind='prophetkv_selection',request_id=meta.request_id,
        scores=eligible,selected_positions=selected,eligible_count=len(eligible),selected_count=len(selected),
        question_positions=list(meta.question_positions),units=list(meta.units),prefix_tokens=b[1],suffix_tokens=suffix_tokens,
        probe_layers=len(scoring_layers),scoring_layers=list(scoring_layers),sequential_project_layers=scoring_layers[-1]+1,
        suffix_forward_layers=scoring_layers[-1] if sparse.method == "selective_prophetkv" else len(sparse.model.layers),probe_tokens_per_layer=suffix_tokens,probe_query_budget=query_budget,probe_seconds=time.perf_counter()-started,
        normalization='all_context_keys',fusion='mean_layers_fp32',ratio=sparse.ratio,
        selection_stage='before_layer_0_qkv',alignment_count=len(sparse.connector.prophet_aligned),**extra))
    if capture:
        sparse.router_arrays = (torch.stack(captured), eligible.detach().clone(), local_mean)
    return selected
