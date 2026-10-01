"""Opt-in worker instrumentation and native-score handoff; no global runtime edits."""
import math
import time
import numpy as np
from runner.gpu_study_features import LAYERS, query_indices, dependencies, GPU_FEATURES, CONFIDENCE
from runner.gpu_study_policy import Handoff
LIMIT=256*2**20


def distributions(q,k,prefix,need_head=True,need_query=True,retained_bytes=0):
    """Independent FP32 bounded GEMMs, full-context softmax, at most 8 queries/head.

    No tensor from the native scorer is mutated. Full query-key logits never
    coexist for multiple heads. Persistent distributions count against the cap.
    """
    import torch
    nq,nh,d=q.shape;nk=len(k);indices=query_indices(nq);n=nk-prefix
    if k.ndim!=3 or nh%k.shape[1] or n<=1:raise ValueError('Invalid attention shape')
    # K float32 + two logits buffers + reduction work, outputs and reference sample.
    bound=retained_bytes+4*(nk*d+4*8*nk+(nh if need_head else 0)*n+(len(indices) if need_query else 0)*n)+16*2**20
    if bound>LIMIT:raise MemoryError(f'Study scratch bound {bound} exceeds {LIMIT}')
    head=torch.zeros((nh,n),device=q.device,dtype=torch.float32) if need_head else None
    query=torch.zeros((len(indices),n),device=q.device,dtype=torch.float32) if need_query else None
    selected={v:i for i,v in enumerate(indices)};group=nh//k.shape[1]
    old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:
        for kh in range(k.shape[1]):
            keys=k[:,kh].float().T
            for h in range(kh*group,(kh+1)*group):
                for a in range(0,nq,8):
                    z=min(a+8,nq)
                    if not need_head and not any(t in selected for t in range(a,z)):continue
                    logits=(q[a:z,h].float()@keys)/math.sqrt(d)
                    weights=logits.softmax(-1)[:,prefix:]
                    if head is not None:head[h].add_(weights.sum(0)/nq)
                    if query is not None:
                        for t in range(a,z):
                            if t in selected:query[selected[t]].add_(weights[t-a]/nh)
            del keys
    finally:torch.backends.cuda.matmul.allow_tf32=old
    return head,query,bound


def statistic_arrays(distributions,scores,required,kind):
    import torch
    out={};n=len(scores)
    if not torch.isfinite(distributions).all() or (distributions<0).any():raise ValueError('Corrupt feature distribution')
    p=distributions/distributions.sum(-1,keepdim=True)
    if not torch.isfinite(p).all():
        if (distributions<0).any() or not torch.isfinite(distributions).all():raise ValueError('Corrupt feature distribution')
    for ratio in (1,5,10):
        if any(f.startswith(f'{kind}_coverage{ratio}_') for f in required):
            mask=torch.argsort(scores,descending=True,stable=True)[:math.floor(n*ratio/100)]
            out[str(ratio)]=p[:,mask].sum(-1).cpu().tolist()
    if any(f.startswith(kind+'_entropy') for f in required):
        entropy=(-(torch.where(p>0,p*p.clamp_min(torch.finfo(p.dtype).tiny).log(),torch.zeros_like(p))).sum(-1)/math.log(n))
        entropy[distributions.sum(-1)<=0]=float('nan')
        out['entropy']=entropy.cpu().tolist()
    # JSON nulls instead of nonfinite statistics; reduce_statistics treats as missing.
    return {k:[v if math.isfinite(v) else None for v in vals] for k,vals in out.items()}


def confidence_values(logits):
    import torch
    if logits.ndim!=2 or len(logits)!=1 or not torch.isfinite(logits).all():raise ValueError('Expected one full-vocabulary token distribution')
    p=logits[0].float().softmax(-1);v=len(p);top=p.topk(20).values
    vals=[-(torch.where(p>0,p*p.clamp_min(torch.finfo(p.dtype).tiny).log(),torch.zeros_like(p))).sum()/math.log(v),
          1/(v*(p*p).sum()),top[0],top[0]-top[1],top[:5].sum(),top.sum()]
    return dict(zip(CONFIDENCE,[float(x.item()) for x in vals]))


def install(worker):
    import torch
    from ucm.sparse.state import get_ucm_sparse
    from ucm.sparse.prophetkv import runtime,prophetkv
    sparse=get_ucm_sparse()
    if hasattr(sparse,'study'):return dict(installed=True)
    state=dict(active=False,handoff=None,calls=0);sparse.study=state
    native=prophetkv.probe;importance=runtime.context_importance
    def observed(q,k,*a,**kw):
        result=importance(q,k,*a,**kw)
        if state['active']:
            i=state['layer'];state['layer']+=1
            needs=state['needs']
            if i in LAYERS and (needs['head'] or needs['query']):
                torch.cuda.synchronize();start=time.perf_counter()
                h,z,bound=distributions(q,k,sparse.request.boundaries[1],needs['head'],needs['query'],state['retained'])
                state['distributions'].append((i,h,z));state['retained']+=sum(t.numel()*t.element_size() for t in (h,z) if t is not None)
                state['scratch_bound']=max(state['scratch_bound'],bound)
                torch.cuda.synchronize();state['kernel_seconds']+=time.perf_counter()-start
                if state['reference']:
                    # Reference inputs are copied AFTER kernel timing, separately charged.
                    start=time.perf_counter()
                    state['references'].append((i,q.cpu().float().numpy(),k.cpu().float().numpy(),h[0].cpu().numpy() if h is not None else None,z[0].cpu().numpy() if z is not None else None))
                    state['reference_transfer_seconds']+=time.perf_counter()-start
        return result
    def probe(s,positions,embeddings,query_budget=16384):
        if state['handoff'] is not None:
            payload=state['handoff'];start=time.perf_counter()
            raw,event=payload.consume(state['binding'],s.request.request_id)
            for i in range(64):s.connector.wait_for_layer_load(f'model.layers.{i}.self_attn.attn')
            scores=torch.from_numpy(raw).to(positions.device)
            from ucm.sparse.prophetkv.selection import select
            selected=select(scores,s.request.boundaries[1],s.ratio)
            event.update(request_id=s.request.request_id,scores=scores,selected_positions=selected,
                selected_count=len(selected),ratio=s.ratio,probe_seconds=0.,alignment_count=len(s.connector.prophet_aligned),
                scoring_executed=False,source_request=payload.binding['source_request'],handoff_seconds=time.perf_counter()-start)
            s.selection_diagnostics.append(event);state['handoff']=None;state['consumed']+=1
            return selected
        if state['active']:
            state['calls']+=1
            if state['calls']!=1:raise ValueError('More than one study scoring probe')
        return native(s,positions,embeddings,query_budget)
    runtime.context_importance=observed;prophetkv.probe=probe
    model=worker.model_runner.model;original=model.compute_logits
    def compute_logits(*a,**kw):
        logits=original(*a,**kw)
        if state['active'] and state['needs']['confidence'] and logits is not None:
            if state['confidence'] is not None:raise ValueError('Multiple confidence tokens')
            if logits.shape[-1]!=model.config.vocab_size:raise ValueError('Incomplete vocabulary')
            torch.cuda.synchronize();start=time.perf_counter();state['confidence']=confidence_values(logits)
            torch.cuda.synchronize();state['confidence_seconds']+=time.perf_counter()-start
        return logits
    model.compute_logits=compute_logits
    return dict(installed=True)


def arm_probe(worker,binding,required,reference=False):
    from ucm.sparse.state import get_ucm_sparse
    install(worker);s=get_ucm_sparse();state=s.study
    if state['active'] or state['handoff'] is not None:raise ValueError('Unretired study state')
    state.update(active=True,binding=binding,required=required,needs=dependencies(required),calls=0,consumed=0,layer=0,
        distributions=[],retained=0,scratch_bound=0,kernel_seconds=0.,confidence=None,confidence_seconds=0.,
        reference=reference,references=[],reference_transfer_seconds=0.)
    return True


def finish_probe(worker):
    import torch
    from ucm.sparse.state import get_ucm_sparse
    from vllm.distributed import get_tensor_model_parallel_rank,tensor_model_parallel_all_reduce
    s=get_ucm_sparse();state=s.study;rank=get_tensor_model_parallel_rank()
    if not state['active'] or state['calls']!=1 or state['layer']!=64 or s.router_arrays is None:raise ValueError('Incomplete native probe')
    state['active']=False
    torch.cuda.synchronize();start=time.perf_counter()
    layers,scores,means=s.router_arrays
    cpu_layers=layers.cpu().numpy();cpu_scores=scores.cpu().numpy();cpu_means=means.cpu().numpy()
    transfer=time.perf_counter()-start
    stats={};start=time.perf_counter()
    for kind,col in [('head',1),('query',2)]:
        if not state['needs'][kind]:continue
        pieces=[]
        for item in state['distributions']:
            tensor=item[col]
            if kind=='query':tensor=tensor_model_parallel_all_reduce(tensor)/4
            pieces.append(tensor)
        stats[kind]={}
        for tensor in pieces:
            for key,values in statistic_arrays(tensor,scores,state['required'],kind).items():
                stats[kind].setdefault(key,[]).extend(values)
    torch.cuda.synchronize();reduction=time.perf_counter()-start
    # Keep an immutable score payload across ordinary request retirement.
    event=next(e for e in s.selection_diagnostics if e['kind']=='prophetkv_selection')
    safe={k:v for k,v in event.items() if not torch.is_tensor(v)}
    state['payload']=Handoff(state['binding'],cpu_scores,safe)
    references=state['references'];state['distributions']=[];state['references']=[]
    s.router_arrays=None;s.router_capture=False
    from runner.gpu_study_transport import pack
    return pack(dict(rank=rank,layers=cpu_layers,scores=cpu_scores,local_mean=cpu_means,statistics=stats,
        confidence=state['confidence'],calls=state['calls'],scratch_bound_bytes=state['scratch_bound'],references=references,
        timings=dict(attention_statistics_seconds=state['kernel_seconds'],confidence_seconds=state['confidence_seconds'],
        native_transfer_seconds=transfer,statistics_reduce_transfer_sync_seconds=reduction,reference_transfer_seconds=state['reference_transfer_seconds'])))


def handoff(worker,binding,target,dense=False):
    from ucm.sparse.state import get_ucm_sparse
    s=get_ucm_sparse();state=s.study
    if state['active'] or state['handoff'] is not None or s.request is not None:raise ValueError('Probe has not retired')
    payload=state.pop('payload');payload.arm(binding,target)
    if dense:payload.consume(binding,target)
    else:state['handoff']=payload
    return dict(armed=not dense,dense=dense,source_request=binding['source_request'],target=target)


def completed(worker,dense=False):
    from ucm.sparse.state import get_ucm_sparse
    s=get_ucm_sparse();state=s.study
    if state['active'] or state['handoff'] is not None or 'payload' in state or state['calls']!=1 or state['consumed']!=(0 if dense else 1):raise ValueError('Handoff was not consumed exactly once')
    return dict(scoring_probes=1,handoffs_consumed=state['consumed'])


def discard(worker):
    from ucm.sparse.state import get_ucm_sparse
    state=get_ucm_sparse().study
    if state['active'] or state['handoff'] is not None:raise ValueError('Active handoff')
    state.pop('payload',None)
    return True
