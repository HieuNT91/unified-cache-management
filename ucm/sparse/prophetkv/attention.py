"""Positional causal fusion with FA2-style softmax for ragged queries.

No installed vLLM code is changed. Cache writes use vLLM's existing operator;
both ProphetKV and CacheBlend prefill attention are replaced. Decode retains
the installed backend. Paged K/V gathers and kernel work are inside TTFT.
"""
import torch

DEFAULT_POSITION_ATTENTION_KERNEL = 'fa2-ragged-v1'


def _tuned_ragged_attention(query, *args, **kwargs):
    """Use the fixed tile heuristic evaluated in the October 14 inference run."""
    from .causal_fa2_kernel import ragged_positions_attention_fwd
    config = (32, 64, 4, 2) if len(query) <= 256 else (128, 64, 8, 3)
    return ragged_positions_attention_fwd(query, *args, config=config, **kwargs)


def dense_reference(q, k, v, positions):
    """Independent float32 GQA reference, for small validation query subsets."""
    groups=q.shape[1]//k.shape[1]
    scores=torch.einsum('qhgd,khd->qhgk',q.float().reshape(len(q),k.shape[1],groups,-1),k.float())
    scores/=q.shape[-1]**.5
    mask=torch.arange(len(k),device=q.device)[None,:]>positions[:,None]
    scores.masked_fill_(mask[:,None,None,:],float('-inf'))
    return torch.einsum('qhgk,khd->qhgd',scores.softmax(-1),v.float()).reshape_as(q)


def is_contiguous_suffix(positions,length):
    return torch.equal(positions,torch.arange(length-len(positions),length,device=positions.device))


def positional_kernel(sparse):
    name = getattr(sparse, 'position_attention_kernel', DEFAULT_POSITION_ATTENTION_KERNEL)
    if name == DEFAULT_POSITION_ATTENTION_KERNEL:
        return _tuned_ragged_attention
    elif name == '_ragged_positions_fa2_kernel':
        from .causal_fa2_kernel import ragged_positions_attention_fwd
    elif name == 'original':
        from .causal_kernel import ragged_positions_attention_fwd
    else:
        raise ValueError('Unknown original-position attention kernel: ' + name)
    return ragged_positions_attention_fwd


def suffix_attention(sparse, query, key, value, positions):
    """Keep native suffix scoring; retain the trajectory full-FA2 opt-in."""
    if getattr(sparse, 'position_attention_kernel', DEFAULT_POSITION_ATTENTION_KERNEL) in (
            'original', DEFAULT_POSITION_ATTENTION_KERNEL):
        from torch.nn.attention.bias import causal_lower_right
        output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1)[None], key.transpose(0, 1)[None], value.transpose(0, 1)[None],
            attn_mask=causal_lower_right(len(query), len(key)), enable_gqa=True, dropout_p=0.)
        return output[0].transpose(0, 1)
    output = torch.empty_like(query)
    zero = torch.zeros(1, device=query.device, dtype=torch.int32)
    q_len = torch.tensor([len(query)], device=query.device, dtype=torch.int32)
    k_len = torch.tensor([len(key)], device=query.device, dtype=torch.int32)
    positional_kernel(sparse)(query, key, value, output, zero, q_len, zero, k_len,
                              positions.contiguous(), len(query), is_causal=True)
    return output

def install(layer, sparse):
    """Install once per attention instance; active request state is never captured."""
    impl=layer.impl
    if hasattr(impl,'_prophetkv_original_forward'):
        return
    if impl.__class__.__name__!='FlashAttentionImpl':
        raise RuntimeError('Unvalidated attention backend')
    original=impl.forward
    impl._prophetkv_original_forward=original
    def forward(layer,query,key,value,kv_cache,attn_metadata,output=None,output_scale=None):
        if not sparse.active:
            if getattr(sparse.connector, 'router_dense_id', None):
                sparse.router_native_layers.add(layer.layer_name)
            return original(layer,query,key,value,kv_cache,attn_metadata,output,output_scale)
        from vllm.v1.attention.backends.flash_attn import reshape_and_cache_flash
        ragged_positions_attention_fwd = positional_kernel(sparse)
        if (output is None or output_scale is not None or impl.kv_cache_dtype!='auto'
            or impl.kv_sharing_target_layer_name is not None or attn_metadata.use_cascade
            or impl.sliding_window!=(-1,-1) or impl.logits_soft_cap!=0):
            raise RuntimeError('Unsupported positional fusion configuration')
        contiguous=is_contiguous_suffix(sparse.fusion_positions,len(sparse.fusion_slots))
        k_cache,v_cache=kv_cache.unbind(0)
        kernel_name = getattr(sparse, 'position_attention_kernel', DEFAULT_POSITION_ATTENTION_KERNEL)
        use_fa2 = kernel_name == '_ragged_positions_fa2_kernel'
        tree = getattr(sparse, 'method', None) == 'tree-prophetkv'
        if contiguous and not use_fa2 and not tree:
            # Native bottom-right causality is correct only for an actual,
            # contiguous suffix in original positions. This includes 0/100%
            # controls and CacheBlend's dense first layer, never arbitrary
            # compacted queries. It also avoids avoidable BF16 backend drift.
            original(layer,query,key,value,kv_cache,attn_metadata,output,output_scale)
        else:
            reshape_and_cache_flash(key,value,k_cache,v_cache,attn_metadata.slot_mapping,
                impl.kv_cache_dtype,layer._k_scale,layer._v_scale)
        if not contiguous or use_fa2 or sparse.audit or tree:
            slots=sparse.fusion_slots
            k=k_cache[slots//sparse.block_size,slots%sparse.block_size]
            v=v_cache[slots//sparse.block_size,slots%sparse.block_size]
        if tree:
            from .tree_attention import rmask_attention, masked_suffix_attention
            bounds = sparse.request.boundaries
            rmask_attention(query, k, v, sparse.fusion_positions, bounds,
                            sparse.coverage_selected, impl.scale, output, sparse=sparse)
            allowed = torch.cat((torch.arange(bounds[1], device=query.device), sparse.coverage_selected))
            masked_suffix_attention(query, k, v, sparse.fusion_positions, bounds, allowed, impl.scale, output)
        elif not contiguous or use_fa2:
            ragged_positions_attention_fwd(query,k,v,output,sparse.fusion_zero,
                sparse.fusion_q_len,sparse.fusion_zero,sparse.fusion_k_len,
                sparse.fusion_positions,len(query),is_causal=True,sm_scale=impl.scale)
        if sparse.audit:
            indices=torch.tensor(sorted(set([0,len(query)//2,max(0,len(query)-257),len(query)-1])),device=query.device)
            if tree:
                from .tree_attention import dense_reference as tree_reference
                reference=tree_reference(query[indices],k,v,sparse.fusion_positions[indices],
                                         sparse.request.boundaries,sparse.coverage_selected)
            else:
                reference=dense_reference(query[indices],k,v,sparse.fusion_positions[indices])
            actual=output[indices].float()
            error=(actual-reference).abs()
            passed=torch.all(error<=.03+.03*reference.abs())
            event=sparse.pending_audit[-1]
            event.update(causal_attention_verified=passed,causal_audit_queries=len(indices),
                causal_audit_heads=query.shape[1],causal_attention_max_abs_error=error.max(),
                attention_backend='tree-prophetkv-masked' if tree else 'ragged_positions_fa2' if (use_fa2 or
                    (not contiguous and kernel_name == DEFAULT_POSITION_ATTENTION_KERNEL))
                    else 'native_verified_contiguous_suffix' if contiguous else 'authors_position_causal_triton')
            if not passed:
                raise RuntimeError('Original-position causal attention numerical audit failed')
        return output
    impl.forward=forward
