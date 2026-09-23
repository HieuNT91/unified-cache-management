"""Explicit user-authorized total/anchor pairs; no automatic rescaling."""
from prophetkv_gpu import REPO

METHOD = 'prophetkv_with_expansion'
PARENT = REPO / '.results/prophetkv-with-expansion-ruler4x50-20260923'
PAIRS = ((.30, .225), (.40, .30), (.50, .375))
CASES = tuple(f'{METHOD}-{round(total * 100)}' for total, _ in PAIRS)
PARAMETERS = {
    case: dict(total_ratio=total, anchor_ratio=anchor, max_gap=2,
               score_exponent=.5, window_scale=8., window_exponent=.5,
               min_window=8, max_window=64)
    for case, (total, anchor) in zip(CASES, PAIRS)
}
