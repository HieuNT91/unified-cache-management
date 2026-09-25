"""Explicit user-authorized total/anchor pairs; no automatic rescaling."""
from prophetkv_gpu import REPO

METHOD = 'prophetkv_with_expansion'
PARENT = REPO / '.results/prophetkv-with-expansion-ruler4x50-20260923'
PRIOR_EXTENSION = REPO / '.results/prophetkv-with-expansion-ratios-ruler4x50-20260923'
PAIRS = ((.10, .05), (.20, .10), (.30, .15), (.40, .20), (.50, .25), (.60, .30))
EXPANSION_CASES = tuple(f'{METHOD}-{round(total * 100)}' for total, _ in PAIRS)
CASES = ('baseline', *EXPANSION_CASES)
PARAMETERS = {'baseline': {}, **{
    case: dict(total_ratio=total, anchor_ratio=anchor, max_gap=4,
               score_exponent=.5, window_scale=8., window_exponent=.5,
               min_window=8, max_window=64)
    for case, (total, anchor) in zip(EXPANSION_CASES, PAIRS)
}}
TASKS = ('cwe', 'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
         'qa_1', 'niah_multivalue', 'niah_multiquery')
MAX_OUTPUT_TOKENS = 256
