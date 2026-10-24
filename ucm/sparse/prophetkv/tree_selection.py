"""F3 tree primitives ported from prophetkv/tree-rmask-qwen3-4b.

Source: f96cb874c7becddd868f8f47872701f6111489ac. F3 retains the source
local-window denominator; all query heads in the first quarter of layers.
Imported only by the explicit tree-prophetkv method.
"""
import math
import torch
import numba as _numba
import numpy as _np

class F3Recorder:
    """Construction-time link strength of a chunk (outside the measured TTFT), for the hierarchical selection.

    For the gap before token t, F3[t] = attention mass that tokens [t, t+W) put on [t-W, t), divided by their mass on
    [t-W, i), averaged over EVERY head of the first quarter of the decoder layers (no head selection; 9 of 36 layers
    for Qwen3-4B, 16 of 64 for Qwen3-32B). High = both sides read each other; low = a boundary. The chunk's first four
    tokens (sinks) are excluded. Same arithmetic as the simulator's linkfeat.f3_capture, so the tree built from it is
    the one evaluated there. Under tensor parallelism each rank sums its own heads; result() all-reduces the sum and
    the head count, so every rank holds the same F3."""
    DEPTH_FRACTION = 0.25

    def __init__(self, tokens, head_dim, num_layers, W=16):
        from runner.layout import token_hash
        self.key = token_hash(list(tokens)); self.T = len(tokens); self.D = head_dim; self.W = W
        if not 64 <= self.T <= 4096 or self.T % 64 or head_dim <= 0 or num_layers <= 0 or W != 16:
            raise ValueError('F3 requires an original block-aligned chunk and W=16')
        self.L = max(1, round(num_layers * self.DEPTH_FRACTION))
        self.seen = 0; self.heads = 0; self.acc = None

    def add_layer(self, layer_index, query, key):
        if layer_index >= self.L:
            return False
        if layer_index != self.seen:
            raise RuntimeError('F3 layers must be recorded exactly once in ascending order')
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
        if self.seen != self.L or self.acc is None or not self.heads:
            raise RuntimeError('Incomplete F3 construction')
        from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
        acc, heads = self.acc, torch.tensor([float(self.heads)], device=self.acc.device)
        if get_tp_group().world_size > 1:
            acc = tensor_model_parallel_all_reduce(acc); heads = tensor_model_parallel_all_reduce(heads)
        out = (acc / heads.clamp(min=1)).cpu().numpy(); self.acc = None
        return out

def tree_need(s, u):
    """need(t) = max(u_t, max over the ancestors A of t of sum_A u / |A|) on the min-Cartesian tree of s (split
    top-down at the weakest link; s[t] is the gap before t, s[0] and chunk starts are -inf). A token is needed as much
    as the densest constituent containing it: the value of a needle inherits its key's density. O(N log N).

    Preserve the upstream implementation's equal-boundary inheritance behavior,
    even where it differs from full recursive ancestor-density inheritance.
    """
    import numpy as np
    s, u = np.asarray(s, dtype=np.float64), np.asarray(u, dtype=np.float64)
    if (s.ndim != 1 or u.ndim != 1 or len(s) != len(u) or not len(u)
            or np.isnan(s).any() or np.isposinf(s).any()
            or not np.isfinite(u).all() or (u < 0).any()):
        raise ValueError('Invalid F3 links or tree density scores')
    N = len(u); S = np.empty(N + 1); S[:N] = s; S[0] = -np.inf; S[N] = -np.inf
    order = np.lexsort((np.arange(N + 1), S))
    return _tree_need_nb(S, np.asarray(u, dtype=np.float64), order)


def select_tree(links, scores, boundaries, ratio):
    """Source tree arithmetic and stable ties; S4 consumes the declared budget."""
    from runner.tree_prophetkv import budget
    count, forced = budget(boundaries, ratio)
    prefix, end = boundaries[1], boundaries[-2]
    links = _np.asarray(links, dtype=_np.float64).copy()
    scores = _np.asarray(scores, dtype=_np.float64)
    if (links.shape != (end-prefix,) or scores.shape != links.shape
            or not _np.isfinite(links).all() or (links < 0).any() or (links > 1.0001).any()
            or not _np.isfinite(scores).all() or (scores < 0).any()):
        raise ValueError('Invalid F3/score shape or values')
    links[_np.asarray(boundaries[1:-2])-prefix] = -_np.inf
    need = tree_need(links, scores / max(scores.sum(), 1e-30))
    selected = set(forced)
    for position in _np.argsort(-need, kind='stable') + prefix:
        if len(selected) >= count:
            break
        selected.add(int(position))
    return sorted(selected), need

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
