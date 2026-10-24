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


def masked_suffix_attention(query,k,v,positions,bounds,allowed,scale,out):
    """UCM_MASK_QUESTION: prompt-suffix query rows (positions >= ctx) attend only the context keys in `allowed`
    (chunk 0 + recomputed positions) and the causal suffix; reused (stale) context keys are hidden, so the question
    is encoded on a consistent context. Rows < ctx keep the output already in `out`. Two varlen FlashAttention
    calls merged by log-sum-exp. k/v: [step_end, Hkv, D] in original positions."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    ctx=int(bounds[-2])
    rows=torch.nonzero(positions>=ctx)[:,0]
    if len(rows)==0:
        return
    a,z=int(positions[rows[0]]),int(positions[rows[-1]])+1
    if z-a!=len(rows):
        raise RuntimeError('Masked suffix rows must be contiguous')
    q=query[rows]  # [rows, H, D]
    def run(kk,vv,causal):
        cq=torch.tensor([0,len(q)],device=q.device,dtype=torch.int32); ck=torch.tensor([0,len(kk)],device=q.device,dtype=torch.int32)
        return flash_attn_varlen_func(q,kk,vv,max_seqlen_q=len(q),cu_seqlens_q=cq,max_seqlen_k=len(kk),cu_seqlens_k=ck,
                                      softmax_scale=scale,causal=causal,return_softmax_lse=True)
    oA,lA=run(k[allowed],v[allowed],False)
    oB,lB=run(k[ctx:z],v[ctx:z],True)
    lA=lA.transpose(0,1)[...,None].float(); lB=lB.transpose(0,1)[...,None].float()
    m=torch.maximum(lA,lB); wA=(lA-m).exp(); wB=(lB-m).exp()
    merged=((oA.float()*wA+oB.float()*wB)/(wA+wB)).to(out.dtype)
    o=out.view(len(query),*q.shape[1:])
    o[rows]=merged


def rmask_attention(query,k,v,positions,bounds,recomputed,scale,out):
    """UCM_RMASK: a recomputed context row at position p in chunk c attends only KV that is valid for it: chunk 0
    (the exact prefix), the recomputed positions of earlier chunks (fresh, this layer), and the causal prefix of its
    own chunk; stale tokens of other chunks are hidden. One ragged-kernel call, one batch per chunk: the allowed keys
    of chunk c are gathered in position order and each query's position is remapped to its index in that gather,
    so the kernel's positional causality is exactly 'allowed and earlier'. Suffix rows (positions >= ctx) are left
    to masked_suffix_attention. k/v: [step_end, Hkv, D] in original positions."""
    from .causal_kernel import ragged_positions_attention_fwd
    dev=query.device; b=torch.as_tensor(bounds,device=dev); ctx=int(b[-2]); b1=int(b[1])
    rows=torch.nonzero(positions<ctx)[:,0]
    if len(rows)==0:
        return
    p=positions[rows]
    if not bool((p[1:]>p[:-1]).all()):
        raise RuntimeError('RMASK: recomputed rows must be sorted by position')
    if bool((p<b1).any()):
        raise RuntimeError('RMASK: chunk 0 is the exact prefix and is never recomputed')
    rec=recomputed.to(dev).long()
    c=torch.searchsorted(b,p,right=True)-1
    cs,counts=torch.unique_consecutive(c,return_counts=True)
    kv_idx=[];kv_start=[];kv_len=[];qpos=torch.empty_like(p);q_start=[];q_len=[];off=0;r0=0
    for ci,n in zip(cs.tolist(),counts.tolist()):
        bc=int(b[ci]); pmax=int(p[r0+n-1])
        earlier=rec[(rec>=b1)&(rec<bc)]
        idx=torch.cat((torch.arange(b1,device=dev),earlier,torch.arange(bc,pmax+1,device=dev)))
        kv_idx.append(idx); kv_start.append(off); kv_len.append(len(idx)); q_start.append(r0); q_len.append(n)
        qpos[r0:r0+n]=p[r0:r0+n]-bc+b1+len(earlier)
        off+=len(idx); r0+=n
    kv_idx=torch.cat(kv_idx)
    i32=lambda x:torch.tensor(x,device=dev,dtype=torch.int32)
    q=query[rows].contiguous(); o=torch.empty_like(q)
    ragged_positions_attention_fwd(q,k[kv_idx].contiguous(),v[kv_idx].contiguous(),o,i32(q_start),i32(q_len),i32(kv_start),i32(kv_len),
        qpos.to(torch.int64),max(q_len),is_causal=True,sm_scale=scale)
    out.view(len(query),*q.shape[1:])[rows]=o


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
            if getattr(sparse.connector, 'router_dense_id', None):
                sparse.router_native_layers.add(layer.layer_name)
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
        if not contiguous and getattr(sparse,'rmask',False):
            if not getattr(sparse,'mask_question',False):
                raise RuntimeError('UCM_RMASK needs UCM_MASK_QUESTION (suffix rows are computed by the masked suffix pass)')
            rmask_attention(query,k,v,sparse.fusion_positions,sparse.request.boundaries,sparse.coverage_selected,impl.scale,output)
        elif not contiguous:
            ragged_positions_attention_fwd(query,k,v,output,sparse.fusion_zero,
                sparse.fusion_q_len,sparse.fusion_zero,sparse.fusion_k_len,
                sparse.fusion_positions,len(query),is_causal=True,sm_scale=impl.scale)
        if getattr(sparse,'mask_question',False) and bool((sparse.fusion_positions>=sparse.request.boundaries[-2]).any()):
            if contiguous and not sparse.audit:
                slots=sparse.fusion_slots
                k=k_cache[slots//sparse.block_size,slots%sparse.block_size]
                v=v_cache[slots//sparse.block_size,slots%sparse.block_size]
            masked_suffix_attention(query,k,v,sparse.fusion_positions,sparse.request.boundaries,
                                    sparse.mask_allowed(query.device),impl.scale,output)
        if sparse.audit and not getattr(sparse,'mask_question',False):  # che cau hoi lam lech co chu y khoi dense_reference
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
