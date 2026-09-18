"""Private positional causal fusion using the pinned authors' Triton kernel.

No installed vLLM code is changed. Cache writes use vLLM's existing operator;
both ProphetKV and CacheBlend prefill attention are replaced. Decode retains
the installed backend. Paged K/V gathers and kernel work are inside TTFT.
"""
import torch


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
            return original(layer,query,key,value,kv_cache,attn_metadata,output,output_scale)
        from vllm.v1.attention.backends.flash_attn import reshape_and_cache_flash
        from .causal_kernel import ragged_positions_attention_fwd
        if (output is None or output_scale is not None or impl.kv_cache_dtype!='auto'
            or impl.kv_sharing_target_layer_name is not None or attn_metadata.use_cascade
            or impl.sliding_window!=(-1,-1) or impl.logits_soft_cap!=0):
            raise RuntimeError('Unsupported positional fusion configuration')
        contiguous=is_contiguous_suffix(sparse.fusion_positions,len(sparse.fusion_slots))
        k_cache,v_cache=kv_cache.unbind(0)
        if contiguous:
            # Native bottom-right causality is correct only for an actual,
            # contiguous suffix in original positions. This includes 0/100%
            # controls and CacheBlend's dense first layer, never arbitrary
            # compacted queries. It also avoids avoidable BF16 backend drift.
            original(layer,query,key,value,kv_cache,attn_metadata,output,output_scale)
        else:
            reshape_and_cache_flash(key,value,k_cache,v_cache,attn_metadata.slot_mapping,
                impl.kv_cache_dtype,layer._k_scale,layer._v_scale)
        if not contiguous or sparse.audit:
            slots=sparse.fusion_slots
            k=k_cache[slots//sparse.block_size,slots%sparse.block_size]
            v=v_cache[slots//sparse.block_size,slots%sparse.block_size]
        if not contiguous:
            ragged_positions_attention_fwd(query,k,v,output,sparse.fusion_zero,
                sparse.fusion_q_len,sparse.fusion_zero,sparse.fusion_k_len,
                sparse.fusion_positions,len(query),is_causal=True,sm_scale=impl.scale)
        if sparse.audit:
            indices=torch.tensor(sorted(set([0,len(query)//2,max(0,len(query)-257),len(query)-1])),device=query.device)
            reference=dense_reference(query[indices],k,v,sparse.fusion_positions[indices])
            actual=output[indices].float()
            error=(actual-reference).abs()
            passed=torch.all(error<=.03+.03*reference.abs())
            event=sparse.pending_audit[-1]
            event.update(causal_attention_verified=passed,causal_audit_queries=len(indices),
                causal_audit_heads=query.shape[1],causal_attention_max_abs_error=error.max(),
                attention_backend='native_verified_contiguous_suffix' if contiguous else 'authors_position_causal_triton')
            if not passed:
                raise RuntimeError('Original-position causal attention numerical audit failed')
        return output
    impl.forward=forward
