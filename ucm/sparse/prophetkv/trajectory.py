"""Copy-gated trajectory selection (training-free).

Readers are the last prompt token and D greedy draft tokens decoded over the stale cache; A[t] is reader t's
attention over the context (mean over layers and heads, softmax over context keys).

* Copy evidence: the kappa-weighted trajectory mass at offset +1 (reader t reads j, reader t+1 reads j+1),
  M(k) = sum_t sqrt(kappa_t kappa_{t+1}) sum_j h(a_t(j), a_{t+1}(j+k)),  h(p, q) = pq / (p + q),
  kappa_t = sum_j a_t(j)^2 / (sum_j a_t(j))^2, is compared with the same mass at other offsets k (a null of
  non-copy trajectories): z = (M(1) - mean_k M(k)) / sd_k M(k).
* Copying (z >= z0): maximise sum_j x_j u_j - lam * sum_j c_j [x_j != x_{j+1}] subject to sum_j x_j = B, with
  u = sum_t a_t and c_j the offset +1 trajectory weight: a Potts chain, solved exactly by DP per Lagrange
  multiplier (bisection), topped up with the best single tokens.
* Not copying: top-k of the last prompt token (the drafts add nothing).
"""
import math
import numpy as np
import torch
from numba import njit

NULL_OFFSETS = [k for k in range(-16, 17) if k not in (0, 1)]


def _h(p, q):
    return p * q / (p + q).clamp(min=1e-30)


@torch.inference_mode()
def trajectory_stats(E):
    E = E.float()
    kap = (E ** 2).sum(1) / E.sum(1) ** 2
    w = torch.sqrt(kap[:-1] * kap[1:])[:, None]
    a, b = E[:-1], E[1:]
    def mass(k):
        if k > 0:
            return float((w * _h(a[:, :-k], b[:, k:])).sum())
        return float((w * _h(a[:, -k:], b[:, :k])).sum())
    edge = (w * _h(a[:, :-1], b[:, 1:])).sum(0)
    null = np.array([mass(k) for k in NULL_OFFSETS])
    z = (float(edge.sum()) - null.mean()) / max(null.std(ddof=1), 1e-30)
    return z, edge


@njit(cache=True)
def _chain(g, c):
    n = g.shape[0]
    s0 = 0.0
    s1 = g[0]
    b0 = np.zeros(n, np.int8)
    b1 = np.zeros(n, np.int8)
    for j in range(1, n):
        cj = c[j - 1]
        if s0 >= s1 - cj:
            n0 = s0; b0[j] = 0
        else:
            n0 = s1 - cj; b0[j] = 1
        if s1 >= s0 - cj:
            n1 = g[j] + s1; b1[j] = 1
        else:
            n1 = g[j] + s0 - cj; b1[j] = 0
        s0 = n0; s1 = n1
    x = np.zeros(n, np.bool_)
    st = 1 if s1 > s0 else 0
    for j in range(n - 1, -1, -1):
        x[j] = st == 1
        st = b1[j] if st == 1 else b0[j]
    return x


def chain_cut(u, c, budget, iters=40):
    u = np.ascontiguousarray(u, np.float64); c = np.ascontiguousarray(c, np.float64)
    lo, hi = 0.0, float(u.max()) + 1e-12
    best = None
    for _ in range(iters):
        mu = 0.5 * (lo + hi)
        x = _chain(u - mu, c); n = int(x.sum())
        if n > budget:
            lo = mu
        else:
            hi = mu
            if best is None or n > best[0]:
                best = (n, x)
        if best is not None and best[0] >= budget * 0.995:
            break
    x = best[1].copy() if best is not None else np.zeros(len(u), np.bool_)
    rest = budget - int(x.sum())
    if rest > 0:
        order = np.argsort(-np.where(x, -1e30, u), kind='stable')[:rest]
        x[order] = True
    return x


def gated_selection(rows, prefix, ratio, lam=16384.0, z0=2.0):
    """rows: [1 + D, context] reader attention. Returns (selected positions, z, copying)."""
    from .selection import select
    E = rows[:, prefix:]
    budget = math.floor(E.shape[1] * ratio)
    z, edge = trajectory_stats(E)
    if z < z0:
        return select(E[0].float().contiguous(), prefix, ratio), z, False
    x = chain_cut(E.sum(0).double().cpu().numpy(), (lam * edge).double().cpu().numpy(), budget)
    selected = torch.from_numpy(np.nonzero(x)[0]).to(rows.device) + prefix
    return selected.sort().values, z, True
