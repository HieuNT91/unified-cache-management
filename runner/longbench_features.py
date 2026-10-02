"""Predeclared coverage routing features for LongBench/RULER; attention only."""
import numpy as np
from runner.corpus import selection

FEATURES = ('coverage5_median', 'head_coverage1_p10', 'coverage5_min',
            'group_agreement', 'top20_mass')
HEAD_LAYERS = (7, 15, 23, 31, 39, 47, 55, 63)
SCHEMA = 'attention-coverage-tree-v1'
DEFINITIONS = dict(
    version='attention-coverage-five-v1', head_layers=list(HEAD_LAYERS),
    masks='native ascending FP32 all64 mean then TP mean; floor eligible_count*ratio; ascending position ties',
    eligible='context after the exact first chunk, before fresh suffix',
    attention='FP32 softmax over ALL context keys, mean over complete question queries',
    coverage='selected eligible mass / total eligible mass, per layer or per head; prefix excluded from both',
    layers='equal-head TP mean in float64, all 64 layers',
    heads='all Q heads on all four TP ranks at the eight fixed zero-based layers; no head averaging before quantile',
    quantile='numpy linear quantile over all selected-layer/global-head coverage values, q=0.10',
    agreement='cosine of eligible attention means for layers 0-31 and 32-63',
    missing='nocache; zero mass/undefined feature is null; malformed or nonfinite arrays halt')


def features(attention, prefix, end):
    layers = np.asarray(attention['layers'])
    heads = np.asarray(attention['heads'])
    scores = np.asarray(attention['scores'])
    if (layers.dtype != np.float32 or layers.shape != (4, 64, end)
            or heads.dtype != np.float32 or heads.ndim != 4
            or heads.shape[:2] != (4, len(HEAD_LAYERS)) or heads.shape[2] < 1 or heads.shape[3] != end
            or tuple(attention['head_layers']) != HEAD_LAYERS
            or scores.dtype != np.float32 or scores.shape != (end-prefix,) or not 0 <= prefix < end):
        raise ValueError('Invalid LongBench feature capture shape/schema')
    if any(not np.isfinite(a).all() or (a < 0).any() for a in (layers, heads, scores)):
        raise ValueError('Invalid LongBench attention values')
    layer = layers.astype(np.float64).mean(0)[:, prefix:end]
    head = heads[..., prefix:end].astype(np.float64)
    masks = {p: selection(scores, 0, p/100) for p in (1, 5, 20)}
    def coverage(values, mask):
        denominator = values.sum(-1)
        if (denominator <= 0).any():
            return None
        return values[..., mask].sum(-1)/denominator
    c5 = coverage(layer, masks[5])
    c1 = coverage(head, masks[1])
    early, late = layer[:32].mean(0), layer[32:].mean(0)
    norm = np.linalg.norm(early)*np.linalg.norm(late)
    total = scores.astype(np.float64).sum()
    return dict(coverage5_median=None if c5 is None else float(np.median(c5)),
                head_coverage1_p10=None if c1 is None else float(np.quantile(c1, .1, method='linear')),
                coverage5_min=None if c5 is None else float(c5.min()),
                group_agreement=float(np.dot(early, late)/norm) if norm > 0 else None,
                top20_mass=float(scores[masks[20]].astype(np.float64).sum()/total) if total > 0 else None)
