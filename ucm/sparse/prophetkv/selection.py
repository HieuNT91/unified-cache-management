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
        if any(x%64 for x in b[:-1]) or b[-1]-b[-2]<256:
            raise ValueError('Expected block-aligned context and at least 256 fresh tokens')
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


class F3Recorder:
    """Construction-time link strength of a chunk (outside the measured TTFT), for the hierarchical selection.

    For the gap before token t, F3[t] = attention mass that tokens [t, t+W) put on [t-W, t), divided by their mass on
    [t-W, i), averaged over EVERY head of layers 0..layers-1 (no head selection). High = both sides read each other;
    low = a boundary. The chunk's first four tokens (sinks) are excluded. Same arithmetic as the simulator's
    linkfeat.f3_capture, so the tree built from it is the one evaluated there."""
    def __init__(self, tokens, head_dim, layers=9, W=16):
        self.key = hash(tokens); self.T = len(tokens); self.D = head_dim; self.L = layers; self.W = W
        self.seen = 0; self.heads = 0; self.acc = None

    def add_layer(self, layer_index, query, key):
        if layer_index >= self.L:
            return False
        T, W = self.T, self.W
        if query.shape[0] != T:
            raise RuntimeError('Construction must be a single prefill step')
        q = query.reshape(T, -1, self.D); kv = key.reshape(T, -1, self.D)
        g = q.shape[1] // kv.shape[1]; dev = query.device
        if self.acc is None:
            self.acc = torch.zeros(T, device=dev, dtype=torch.float32)
        mask = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)
        ar = torch.arange(T, device=dev); jj = ar[:, None] - torch.arange(1, 2 * W + 1, device=dev)[None, :]
        for h in range(q.shape[1]):
            lg = (q[:, h].float() @ kv[:, h // g].float().T) / math.sqrt(self.D)
            lg.masked_fill_(mask, float('-inf')); lg[:, :4] = float('-inf'); lg[:4] = 0
            A = lg.softmax(-1); del lg
            B = torch.where(jj >= 4, A.gather(1, jj.clamp(min=0)), torch.zeros((), device=dev)); del A
            CB = B.cumsum(1); num = torch.zeros(T, device=dev); den = torch.zeros(T, device=dev)
            for k in range(min(W, T)):
                hi = CB[:, k + W - 1]; lo = CB[:, k - 1] if k > 0 else torch.zeros_like(hi)
                num[:T - k] += (hi - lo)[k:]; den[:T - k] += hi[k:]
            self.acc.add_(num / den.clamp(min=1e-9)); self.heads += 1
        self.seen += 1
        return self.seen == self.L

    def result(self):
        out = (self.acc / max(self.heads, 1)).cpu().numpy(); self.acc = None
        return out


def tree_need(s, u):
    """need(t) = max(u_t, max over the ancestors A of t of sum_A u / |A|) on the min-Cartesian tree of s (split
    top-down at the weakest link; s[t] is the gap before t, s[0] and chunk starts are -inf). A token is needed as much
    as the densest constituent containing it: the value of a needle inherits its key's density. O(N log N)."""
    import numpy as np
    N = len(u); S = np.empty(N + 1); S[:N] = s; S[0] = -np.inf; S[N] = -np.inf
    order = np.lexsort((np.arange(N + 1), S))
    return _tree_need_nb(S, np.asarray(u, dtype=np.float64), order)


import numba as _numba
import numpy as _np


@_numba.njit(cache=True)
def _tree_need_nb(S, u, order):
    N = len(u); cs = _np.zeros(N + 1)
    for i in range(N): cs[i + 1] = cs[i] + u[i]
    L = _np.zeros(N + 1, _np.int64); R = _np.full(N + 1, N, _np.int64)
    st = _np.empty(N + 1, _np.int64); top = 0
    for k in range(N + 1):                    # nearest gap to the left with S <= S_k
        while top > 0 and S[st[top - 1]] > S[k]: top -= 1
        L[k] = st[top - 1] if top > 0 else 0
        st[top] = k; top += 1
    top = 0
    for k in range(N, -1, -1):                # nearest gap to the right with S < S_k
        while top > 0 and S[st[top - 1]] >= S[k]: top -= 1
        R[k] = st[top - 1] if top > 0 else N
        st[top] = k; top += 1
    root = cs[N] / N; best = _np.full(N + 1, -_np.inf)
    for i in range(N + 1):                    # increasing S = top-down split order, so parents come first
        k = order[i]
        if k == 0 or k == N:
            best[k] = root; continue
        a = L[k]; z = R[k]
        par = a if S[a] > S[z] else z
        pb = best[par] if best[par] > -_np.inf else root
        d = (cs[z] - cs[a]) / (z - a)
        best[k] = pb if pb > d else d
    out = _np.empty(N)
    for t in range(N):
        par = t if S[t] > S[t + 1] else t + 1
        out[t] = u[t] if u[t] > best[par] else best[par]
    return out


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
