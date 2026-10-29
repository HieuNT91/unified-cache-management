"""Masked recompute from tree-rmask f96cb874; decode remains native.

Keys are gathered in original-position order. No dense mask is used by the
production attention path; dense_reference is for small audit queries only.
"""
import torch

def masked_suffix_attention(query,k,v,positions,bounds,allowed,scale,out):
    """Prompt-suffix query rows (positions >= ctx) attend only the context keys in `allowed`
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

def rmask_attention(query,k,v,positions,bounds,recomputed,scale,out, *, sparse=None):
    """A recomputed context row at position p in chunk c attends only KV that is valid for it: chunk 0
    (the exact prefix), the recomputed positions of earlier chunks (fresh, this layer), and the causal prefix of its
    own chunk; stale tokens of other chunks are hidden. One ragged sequence per chunk: the allowed keys
    of chunk c are gathered in position order and each query's position is remapped to its index in that gather,
    so the kernel's positional causality is exactly 'allowed and earlier'. Suffix rows (positions >= ctx) are left
    to masked_suffix_attention. Kernel calls pack at most len(k) gathered rows, bounding temporary K/V copies
    to one prompt's K/V even when many chunks share earlier keys. k/v: [step_end, Hkv, D] in original positions."""
    from .attention import positional_kernel
    ragged_positions_attention_fwd = positional_kernel(sparse)
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
    i32=lambda x:torch.tensor(x,device=dev,dtype=torch.int32)
    q=query[rows].contiguous(); o=torch.empty_like(q)
    qs,qn,ks,kn=map(i32,(q_start,q_len,kv_start,kv_len))
    qpos=qpos.to(torch.int64); maxq=max(q_len)
    first=0
    while first<len(kv_len):
        last=first
        while last<len(kv_len) and kv_start[last]+kv_len[last]-kv_start[first]<=len(k):
            last+=1
        if last==first:
            raise RuntimeError('RMASK: a chunk gather exceeds the available KV rows')
        idx=torch.cat(kv_idx[first:last])
        # Keep full Q/O and global query offsets: the kernel tile heuristic and
        # each chunk's query tiles remain identical to the single-call path.
        # Only KV offsets are rebased. Argument temporaries are released before
        # the next gather, so shared earlier keys cannot multiply peak VRAM.
        ragged_positions_attention_fwd(q,k[idx].contiguous(),v[idx].contiguous(),o,
            qs[first:last],qn[first:last],ks[first:last]-kv_start[first],kn[first:last],
            qpos,maxq,is_causal=True,sm_scale=scale)
        first=last
    out.view(len(query),*q.shape[1:])[rows]=o


def dense_reference(query, key, value, positions, bounds, recomputed):
    """Independent masked GQA reference for a small sample of audit rows."""
    prefix, suffix = bounds[1], bounds[-2]
    keys = torch.arange(len(key), device=query.device)
    fresh = (keys < prefix) | torch.isin(keys, recomputed)
    masks = []
    for position in positions.tolist():
        if position >= suffix:
            allowed = fresh | (keys >= suffix)
        else:
            chunk = next(i for i in range(len(bounds)-1) if bounds[i] <= position < bounds[i+1])
            allowed = fresh | ((keys >= bounds[chunk]) & (keys < bounds[chunk+1]))
        masks.append(allowed & (keys <= position))
    groups = query.shape[1] // key.shape[1]
    logits = torch.einsum('qhgd,khd->qhgk',
        query.float().reshape(len(query), key.shape[1], groups, -1), key.float()) / query.shape[-1]**.5
    logits.masked_fill_(~torch.stack(masks)[:, None, None, :], -float('inf'))
    return torch.einsum('qhgk,khd->qhgd', logits.softmax(-1), value.float()).reshape_as(query)
