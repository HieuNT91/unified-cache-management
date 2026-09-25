"""Deterministic GPU segment sums and integer first-visit expansion priorities."""
from functools import lru_cache
import torch
import triton
import triton.language as tl
from .selection import expansion_lookup


@triton.jit
def _segments(Scores, Anchors, Starts, Normalization, Windows, SegmentScores,
              Ends, Lengths, A: tl.constexpr):
    index = tl.program_id(0)
    start = tl.load(Starts + index)
    if start:
        cursor = index
        total = tl.full((), 0., tl.float64)
        end = tl.full((), 0, tl.int64)
        active = tl.full((), True, tl.int1)
        while active:
            end = tl.load(Anchors + cursor)
            total = total + tl.load(Scores + end).to(tl.float64)
            cursor += 1
            next_start = tl.load(Starts + cursor, cursor < A, other=1)
            active = (cursor < A) & ~next_start
        count = cursor - index
        denominator = tl.load(Normalization + count)
        normalized = tl.inline_asm_elementwise(
            "div.rn.f64 $0, $1, $2;", constraints="=d,d,d", args=[total, denominator],
            dtype=tl.float64, is_pure=True, pack=1)
        tl.store(SegmentScores + index, normalized)
        tl.store(Ends + index, end)
        tl.store(Lengths + index, tl.load(Windows + count))


@triton.jit
def _visits(Order, Starts, Ends, Lengths, Priorities, N: tl.constexpr, BLOCK: tl.constexpr):
    rank = tl.program_id(0)
    segment = tl.load(Order + rank)
    if tl.load(Starts + segment):
        end = tl.load(Ends + segment)
        length = tl.load(Lengths + segment)
        length = tl.minimum(length, N - end - 1)
        for tile in range(tl.cdiv(length, BLOCK)):
            offset = tile * BLOCK + tl.arange(0, BLOCK)
            pos = end + 1 + offset
            priority = rank.to(tl.int64) * (N + 1) + offset.to(tl.int64)
            tl.atomic_min(Priorities + pos, priority, offset < length, sem='relaxed')


@lru_cache(maxsize=32)
def _lookup(config, anchors, device):
    normalization, windows = expansion_lookup(config, anchors)
    return (torch.tensor(normalization, device=device, dtype=torch.float64),
            torch.tensor(windows, device=device, dtype=torch.long))


def select_cuda(scores, prefix, config, budget, anchor_count):
    n = len(scores)
    rank = torch.argsort(scores, descending=True, stable=True)
    anchors = rank[:anchor_count].sort().values
    details = dict(eligible_count=n, anchor_count=anchor_count, segment_count=0,
                   unique_expansion_count=0, fallback_count=budget-anchor_count, final_count=budget)
    if not anchor_count:
        return (rank[:budget] + prefix).sort().values, details
    normalization, windows = _lookup(config, anchor_count, scores.device)
    starts = torch.ones(anchor_count, dtype=torch.bool, device=scores.device)
    starts[1:] = anchors[1:] - anchors[:-1] - 1 > config.max_gap
    segment_scores = torch.full((anchor_count,), -float('inf'), dtype=torch.float64, device=scores.device)
    ends = torch.empty_like(anchors)
    lengths = torch.empty_like(anchors)
    _segments[(anchor_count,)](scores.contiguous(), anchors, starts, normalization, windows,
                               segment_scores, ends, lengths, anchor_count, num_warps=1,
                               enable_fp_fusion=False)
    order = torch.argsort(segment_scores, descending=True, stable=True)
    base = (anchor_count + 1) * (n + 1)
    priorities = torch.empty(n, dtype=torch.long, device=scores.device)
    priorities[rank] = base + torch.arange(n, device=scores.device)
    priorities[anchors] = -1
    block = triton.next_power_of_2(max(1, min(config.max_window, n, 256)))
    _visits[(anchor_count,)](order, starts, ends, lengths, priorities, n, block, num_warps=4)
    chosen = torch.argsort(priorities, stable=True)[:budget]
    expanded = ((priorities[chosen] >= 0) & (priorities[chosen] < base)).sum()
    details.update(segment_count=starts.sum(), unique_expansion_count=expanded,
                   fallback_count=budget-anchor_count-expanded)
    return (chosen + prefix).sort().values, details
