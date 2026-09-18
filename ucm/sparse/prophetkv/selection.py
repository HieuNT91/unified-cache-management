"""Algorithm 1 primitives; FP32 context-only softmax and layer fusion."""
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class RequestMetadata:
    request_id: str
    boundaries: tuple
    question_positions: tuple

    def __post_init__(self):
        b=self.boundaries
        if not self.request_id or len(b)<4 or b[0]!=0 or any(a>=z for a,z in zip(b,b[1:])):
            raise ValueError('Invalid request boundaries')
        if any(x%64 for x in b[:-1]) or b[-1]-b[-2]!=256:
            raise ValueError('Expected block-aligned context and 256 fresh tokens')
        q=self.question_positions
        if not q or list(q)!=sorted(set(q)) or q[0]<b[-2] or q[-1]>=b[-1]:
            raise ValueError('Question must lie entirely in fresh suffix')


def context_importance(q,k,key_tile=2048,query_tile=16):
    """Mean across question queries/heads, softmax over ALL context keys.

    The exact prefix participates in normalization but is excluded from selection.
    Two tiled passes avoid allocating a query-by-context matrix.
    """
    if q.ndim!=3 or k.ndim!=3 or not len(q) or not len(k) or q.shape[-1]!=k.shape[-1] or q.shape[1]%k.shape[1]:
        raise ValueError('Invalid grouped-query shapes')
    if q.is_cuda:
        from .score_kernel import compute_att_full_softmax_importance
        return compute_att_full_softmax_importance(q[None],k[None],target_start=0,
            target_len=len(k),q_start=len(k),causal=False)
    groups=q.shape[1]//k.shape[1]
    result=torch.zeros(len(k),device=q.device,dtype=torch.float32)
    for h in range(k.shape[1]):
        for a in range(0,len(q),query_tile):
            query=q[a:a+query_tile,h*groups:(h+1)*groups].float().transpose(0,1)
            lse=torch.full(query.shape[:2],-float('inf'),device=q.device)
            for b in range(0,len(k),key_tile):
                logits=(query@k[b:b+key_tile,h].float().T)/math.sqrt(q.shape[-1])
                lse=torch.logaddexp(lse,torch.logsumexp(logits,-1))
            for b in range(0,len(k),key_tile):
                logits=(query@k[b:b+key_tile,h].float().T)/math.sqrt(q.shape[-1])
                result[b:b+key_tile]+=(logits-lse[...,None]).exp().sum((0,1))
    return result/(len(q)*q.shape[1])


def select(scores,prefix,ratio):
    if not 0<=ratio<=1 or scores.ndim!=1 or not torch.isfinite(scores).all():
        raise ValueError('Invalid selection')
    count=math.floor(len(scores)*ratio)
    return (torch.argsort(scores,descending=True,stable=True)[:count]+prefix).sort().values


def request_mask(positions,selected,prefix,end,hits,block_size=64):
    if torch.any(positions<prefix) or len(selected)!=len(torch.unique(selected)) or torch.any(selected<prefix) or torch.any(selected>=end):
        raise ValueError('Invalid request selection positions')
    cached=positions<end
    hit=torch.zeros_like(cached)
    hit[cached]=hits[(positions[cached]-prefix)//block_size].bool()
    return ~hit | torch.isin(positions,selected)

class RequestState:
    """Single-use request metadata; masks cannot leak across warmup/decode."""
    def __init__(self):self.pending=None
    def arm(self,meta):
        if self.pending is not None:raise RuntimeError('Unconsumed request metadata')
        self.pending=meta
    def take(self,request_id):
        if self.pending is None or self.pending.request_id!=request_id:
            raise RuntimeError('Request metadata identity mismatch')
        result=self.pending;self.pending=None
        return result

class LayerAlignment:
    """A new request resets alignment; repeated waits never rotate twice."""
    def __init__(self):self.layers=set()
    def reset(self):self.layers.clear()
    def apply(self,name,operation):
        if name not in self.layers:
            operation(name)
            self.layers.add(name)
