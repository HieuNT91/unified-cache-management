"""Algorithm 1 primitives; FP32 context-only softmax and layer fusion."""
from dataclasses import dataclass
import math
from numbers import Real
from functools import lru_cache
import torch


@dataclass(frozen=True, init=False)
class ExpansionConfig:
    """Explicit budgets and geometry for ProphetKV right expansion.

    A changed total budget must declare its anchor budget; the latter is never
    silently scaled or clipped.
    """
    total_ratio: float = .20
    anchor_ratio: float = .15
    max_gap: int = 2
    score_exponent: float = .5
    window_scale: float = 8.
    window_exponent: float = .5
    min_window: int = 8
    max_window: int = 64

    def __init__(self, total_ratio=.20, anchor_ratio=None, max_gap=2,
                 score_exponent=.5, window_scale=8., window_exponent=.5,
                 min_window=8, max_window=64):
        if anchor_ratio is None:
            if total_ratio != .20:
                raise ValueError('A nondefault total_ratio requires explicit anchor_ratio')
            anchor_ratio = .15
        values = dict(total_ratio=total_ratio, anchor_ratio=anchor_ratio,
                      max_gap=max_gap, score_exponent=score_exponent,
                      window_scale=window_scale, window_exponent=window_exponent,
                      min_window=min_window, max_window=max_window)
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite and numeric')
        for name in ('max_gap', 'min_window', 'max_window'):
            if not isinstance(values[name], int) or values[name] < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not 0 <= anchor_ratio <= total_ratio <= 1:
            raise ValueError('Expected 0 <= anchor_ratio <= total_ratio <= 1')
        if min_window > max_window or window_scale < 0:
            raise ValueError('Invalid expansion window bounds or scale')
        for name, value in values.items():
            object.__setattr__(self, name, value)

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


def select(scores,prefix,ratio):
    if not 0<=ratio<=1 or scores.ndim!=1 or not torch.isfinite(scores).all():
        raise ValueError('Invalid selection')
    count=math.floor(len(scores)*ratio)
    return (torch.argsort(scores,descending=True,stable=True)[:count]+prefix).sort().values


@lru_cache(maxsize=32)
def expansion_lookup(config, anchors):
    """Identical host-computed normalization and ties-to-even window values."""
    normalization, windows = [1.], [0]
    for count in range(1, anchors + 1):
        try:
            denominator = count ** config.score_exponent
            proposed = config.window_scale * count ** config.window_exponent
        except OverflowError as exc:
            raise ValueError('Expansion parameters overflow at this anchor count') from exc
        if not math.isfinite(denominator) or denominator <= 0 or not math.isfinite(proposed):
            raise ValueError('Expansion parameters produce invalid lookup values')
        normalization.append(denominator)
        windows.append(max(config.min_window, min(config.max_window, round(proposed))))
    return tuple(normalization), tuple(windows)


def _expansion_input(scores, prefix, config):
    if not isinstance(config, ExpansionConfig):
        raise TypeError('Expected ExpansionConfig')
    if scores.ndim != 1 or not scores.is_floating_point() or not torch.isfinite(scores).all():
        raise ValueError('Expected finite one-dimensional floating-point scores')
    if isinstance(prefix, bool) or not isinstance(prefix, int) or prefix < 0:
        raise ValueError('Expected nonnegative integer prefix')
    return math.floor(len(scores) * config.total_ratio), math.floor(len(scores) * config.anchor_ratio)


def select_expansion_reference(scores, prefix, config=None, *, diagnostics=False, trace=False):
    """Deterministic specification: anchors, transitive segments, right windows.

    Only anchor scores contribute to a segment. Accumulate them sequentially in
    increasing position order as float64, then rank by normalized score and start.
    Internal gaps are not expanded; final fallback uses the original token ranks.
    This CPU oracle intentionally copies GPU inputs and is never the CUDA path.
    """
    config = ExpansionConfig() if config is None else config
    budget, anchor_count = _expansion_input(scores, prefix, config)
    values = scores.detach().cpu().double().tolist()
    ranking = sorted(range(len(values)), key=lambda i: (-values[i], i))
    anchors = sorted(ranking[:anchor_count])
    normalization, windows = expansion_lookup(config, anchor_count)
    groups = []
    for pos in anchors:
        if not groups or pos - groups[-1][-1] - 1 > config.max_gap:
            groups.append([])
        groups[-1].append(pos)
    segments = []
    for group in groups:
        total = 0.
        for pos in group:
            total += values[pos]
        segments.append(dict(start=group[0], end=group[-1], anchors=len(group),
                             score=total / normalization[len(group)], window=windows[len(group)]))
    segments.sort(key=lambda segment: (-segment['score'], segment['start']))
    chosen = set(anchors)
    expanded = 0
    for segment in segments:
        for pos in range(segment['end'] + 1, min(len(values), segment['end'] + 1 + segment['window'])):
            if len(chosen) == budget:
                break
            if pos not in chosen:
                chosen.add(pos)
                expanded += 1
    fallback = 0
    for pos in ranking:
        if len(chosen) == budget:
            break
        if pos not in chosen:
            chosen.add(pos)
            fallback += 1
    selected = torch.tensor(sorted(pos + prefix for pos in chosen), device=scores.device, dtype=torch.long)
    details = dict(eligible_count=len(scores), anchor_count=anchor_count, segment_count=len(groups),
                   unique_expansion_count=expanded, fallback_count=fallback, final_count=budget)
    if trace:
        details['segments'] = segments
    return (selected, details) if diagnostics else selected


def select_expansion(scores, prefix, config=None, *, diagnostics=False):
    """Production dispatch. CUDA scores and candidates stay on their device."""
    config = ExpansionConfig() if config is None else config
    budget, anchors = _expansion_input(scores, prefix, config)
    if scores.is_cuda:
        from .expansion_kernel import select_cuda
        selected, details = select_cuda(scores, prefix, config, budget, anchors)
        return (selected, details) if diagnostics else selected
    return select_expansion_reference(scores, prefix, config, diagnostics=diagnostics)


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
