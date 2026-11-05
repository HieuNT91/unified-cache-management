"""Fixed AdaTreeV1 descriptors from the existing native post-S4 TreeKV probe.

Both token masks are hypothetical feature masks. They never repair KV, invoke
attention, or change the native score vector. The feature order is G5, S60.
"""
import math
from numbers import Integral


FEATURE_NAMES = ('G5_mean', 'S60_p90')
PROTOCOL = 'adatree-v1-native-tailspread-v1'


def route_features(G5_mean, S60_p90):
    """CPU receipt validation, with the same inclusive thresholds as the GPU."""
    if (not math.isfinite(G5_mean) or not math.isfinite(S60_p90)
            or not 0 <= G5_mean <= 1 or not 0 <= S60_p90 <= 1 + 1e-12):
        raise ValueError('Invalid AdaTreeV1 features')
    if S60_p90 <= .73:
        return 'tree-prophetkv-5'
    if G5_mean <= .22:
        return 'tree-prophetkv-20'
    return 'tree-prophetkv-40'


def compute_features(raw, scores, boundaries):
    """Return device FP64 [G5_mean, S60_p90] from eligible attention only.

    ``raw`` is the global TP mean of native per-layer attention, shaped [L, N].
    ``scores`` is the unchanged native FP32 global score vector, shaped [N].
    Boundaries cover the complete prompt: prefix, eligible chunks, fresh suffix.
    Ranking is stable once globally; only reductions are performed per layer.
    No attention or score vector is copied to the host.
    """
    import torch

    b = tuple(boundaries)
    if (len(b) < 4 or any(not isinstance(x, Integral) or isinstance(x, bool) for x in b)
            or b[0] != 0 or any(z <= a for a, z in zip(b, b[1:]))):
        raise ValueError('Invalid AdaTreeV1 boundaries')
    n = b[-2] - b[1]
    if (not isinstance(raw, torch.Tensor) or not isinstance(scores, torch.Tensor)
            or raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != n
            or scores.shape != (n,) or raw.device != scores.device
            or not raw.is_floating_point() or not scores.is_floating_point()):
        raise ValueError('Expected native layer-by-eligible-token attention and scores')
    a = raw.double()
    total = a.sum(-1, keepdim=True)
    valid = (torch.isfinite(a).all() & (a >= 0).all()
             & torch.isfinite(scores).all() & torch.isfinite(total).all() & (total > 0).all())
    if not bool(valid):
        raise ValueError('Invalid native eligible attention or scores')
    p = a / total
    order = torch.argsort(scores, descending=True, stable=True)
    # Cumulative sums followed by boundary differences avoid atomics and a
    # separate sort for each layer. The masks have no effect on source tensors.
    ends = torch.tensor([x - b[1] - 1 for x in b[2:-1]], device=raw.device)
    tail = torch.ones(n, dtype=torch.bool, device=raw.device)
    chunk_masses = []
    for rate in (.05, .60):
        tail[order[:math.floor(rate * n)]] = False
        cumulative = (p * tail).cumsum(-1).index_select(-1, ends)
        q = torch.cat((cumulative[:, :1], cumulative[:, 1:] - cumulative[:, :-1]), dim=-1)
        chunk_masses.append(q)
    q5, q60 = chunk_masses
    squared60 = q60.square().sum(-1)
    if not bool((squared60 > 0).all()):
        raise ValueError('Zero residual attention: S60 spread is undefined')
    g5 = q5.square().sum(-1).sqrt().mean()
    spread60 = q60.sum(-1).square() / (q60.shape[-1] * squared60)
    s60 = torch.quantile(spread60, .90, interpolation='linear')
    return torch.stack((g5, s60))


def decide(features):
    """Return device int64 percent 5/20/40 without rounding the FP64 features."""
    import torch

    if (not isinstance(features, torch.Tensor) or features.shape != (2,)
            or not features.is_floating_point()):
        raise ValueError('Expected AdaTreeV1 feature vector [G5_mean, S60_p90]')
    if not bool(torch.isfinite(features).all() & (features >= 0).all()
                & (features[0] <= 1) & (features[1] <= 1 + 1e-12)):
        raise ValueError('Invalid AdaTreeV1 features')
    # FP64 comparison retains the exact supplied thresholds even if a caller
    # provides an FP32 feature vector; the normal extractor always returns FP64.
    g5, s60 = features.double().unbind()
    return torch.where(s60 <= .73, 5, torch.where(g5 <= .22, 20, 40))
