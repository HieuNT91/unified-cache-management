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
def trajectory_stats(E, unbiased=False):
    """z of the offset +1 trajectory mass against the same mass at 31 other offsets, plus its per-token breakdown.

    `unbiased` divides every mass by the number of terms it sums over. Offset k only has N - |k| overlapping
    positions, so without it the null offsets are mechanically smaller than offset +1 and the null mean is pulled
    down: measured on a context of 60736 with NO copying at all, z sat at exactly 1.70 against a threshold of 2.0,
    leaving 0.30 of real margin and firing the gate on 142 of 145 prompts."""
    E = E.float()
    kap = (E ** 2).sum(1) / E.sum(1) ** 2
    w = torch.sqrt(kap[:-1] * kap[1:])[:, None]
    a, b = E[:-1], E[1:]
    n = a.shape[1]
    def mass(k):
        if k > 0:
            v = float((w * _h(a[:, :-k], b[:, k:])).sum())
        else:
            v = float((w * _h(a[:, -k:], b[:, :k])).sum())
        return v / max(n - abs(k), 1) if unbiased else v
    edge = (w * _h(a[:, :-1], b[:, 1:])).sum(0)
    m1 = float(edge.sum())
    if unbiased:
        m1 /= max(n - 1, 1)
    null = np.array([mass(k) for k in NULL_OFFSETS])
    # Once the mechanical k-dependence is divided out, a reader set with no structure at all makes every offset
    # numerically identical and the sample sd collapses to float32 rounding noise, so z becomes 0/0 and reports
    # whatever the last bit did. Floor the denominator relative to the mass scale: no spread means no evidence.
    z = (m1 - null.mean()) / max(null.std(ddof=1), 1e-6 * abs(float(null.mean())), 1e-30)
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


def gated_selection(rows, prefix, ratio, lam=16384.0, z0=2.0, probe=None, unbiased=False, lam_rel=0.0):
    """rows: [1 + D, context] reader attention, in trajectory order. Returns (selected positions, z, copying).

    `rows[0]` must be the LAST PROMPT TOKEN's row: the drafts are decoded from it, so that is the only row that
    stands one step before draft_1. Splicing an arbitrary probe there (the question probe sits 48 tokens earlier,
    behind the chat template) counts a pair that is not a trajectory step and puts a non-trajectory row into the
    chain's unary. `probe` is the row the NON-copying branch ranks, which is where the probe belongs.
    """
    from .selection import select
    E = rows[:, prefix:]
    budget = math.floor(E.shape[1] * ratio)
    z, edge = trajectory_stats(E, unbiased=unbiased)
    if z < z0:
        p = E[0] if probe is None else probe[prefix:]
        return select(p.float().contiguous(), prefix, ratio), z, False
    u = E.sum(0).double().cpu().numpy()
    c = edge.double().cpu().numpy()
    if lam_rel > 0:
        # lam=16384 is a bare constant with no relation to the scale of u, and at that price the DP essentially
        # never pays for a cut: measured run lengths reach 1320 tokens at 10% and 3815 at 20%, against the ~20
        # tokens a copied uuid actually spans. Scale the total cut capacity to the total unary mass instead, the
        # same convention couple uses for its A-edges, so run length follows the copy evidence and not a constant.
        ctot = float(c.sum())
        c = c * (lam_rel * float(u.sum()) / ctot) if ctot > 0 else c * 0.0
    else:
        c = lam * c
    x = chain_cut(u, c, budget)
    selected = torch.from_numpy(np.nonzero(x)[0]).to(rows.device) + prefix
    return selected.sort().values, z, True
