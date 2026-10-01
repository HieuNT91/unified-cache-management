"""Registered features for the single-probe study; no outcome-dependent inputs."""
import math
import numpy as np
from runner.attention_features import ADDITIONS as CPU_FEATURES, DEFINITIONS as CPU_DEFINITIONS, additional_features
from runner.router_policy import FEATURES

LAYERS=(7,15,23,31,39,47,55,63)
STATS=('min','p10','median','std')
HEAD=[f'head_coverage{p}_{s}' for p in (1,5,10) for s in STATS]
QUERY=[f'query_coverage{p}_{s}' for p in (1,5,10) for s in STATS]
ENTROPY=[f'{kind}_entropy_{s}' for kind in ('head','query') for s in ('min','median','std')]
CONFIDENCE=[f'confidence_{s}' for s in ('entropy','effective_support','top1','margin','top5','top20')]
GPU_FEATURES=HEAD+QUERY+ENTROPY+CONFIDENCE
ADDITIONS=CPU_FEATURES+GPU_FEATURES
REGISTERED=list(FEATURES)+ADDITIONS
DEFINITIONS=dict(version='single-gpu-probe-features-v1',cpu=CPU_DEFINITIONS,layers=LAYERS,
    queries='min(16,Q) uniformly spaced question indices, floor(i*(Q-1)/(min(16,Q)-1)); endpoints included',
    heads='each head on every rank, averaged over ALL question queries; pool eight layers and all heads',
    query='each sampled query averaged over ALL heads across ranks BEFORE eligible normalization; pool eight layers and queries',
    attention='FP32 QK/sqrt(d), softmax over every context position including prefix, then normalize averaged distributions over eligible positions',
    summary='min, linear p10, median, population standard deviation; entropy=-sum(p log p)/log(N)',
    confidence='first-token full vocabulary pre-sampling FP32 softmax; normalized entropy, inverse Simpson support/V, top1, top1-top2, top5 and top20 mass; no token identities',
    scratch_limit_bytes=256*2**20,zero_mass='null; required missing/nonfinite feature forces dense')


def query_indices(count):
    if type(count) is not int or count<1:raise ValueError('Empty question')
    n=min(16,count)
    return [0] if n==1 else [i*(count-1)//(n-1) for i in range(n)]


def summaries(values):
    a=np.asarray(values,dtype=np.float64)
    if not a.size or not np.isfinite(a).all():return dict.fromkeys(STATS)
    if (a<0).any():raise ValueError('Negative attention statistic')
    return dict(min=float(a.min()),p10=float(np.percentile(a,10,method='linear')),median=float(np.median(a)),std=float(a.std()))


def reduce_statistics(rank_stats,required=None):
    wanted=set(GPU_FEATURES if required is None else required);out={}
    if not wanted<=set(GPU_FEATURES):raise ValueError('Unknown GPU feature')
    if sorted(r['rank'] for r in rank_stats)!=list(range(4)):raise ValueError('Missing or duplicate rank')
    for kind in ('head','query'):
        needed={f for f in wanted if f.startswith(kind+'_')}
        if not needed:continue
        ranks=rank_stats if kind=='head' else rank_stats[:1]
        for p in (1,5,10):
            if not any(f.startswith(f'{kind}_coverage{p}_') for f in needed):continue
            arrays=[np.asarray(r['statistics'][kind][str(p)],float) for r in ranks]
            out.update({f'{kind}_coverage{p}_{k}':v for k,v in summaries(np.concatenate(arrays)).items()})
        if any(f.startswith(kind+'_entropy') for f in needed):
            out.update({f'{kind}_entropy_{k}':v for k,v in summaries(np.concatenate([r['statistics'][kind]['entropy'] for r in ranks])).items() if k!='p10'})
    if wanted & set(CONFIDENCE):
        available=[r['confidence'] for r in rank_stats if r.get('confidence') is not None]
        if not available or any(r!=available[0] for r in available):raise ValueError('Missing/disagreeing vocabulary confidence')
        out.update(available[0])
    return {k:out[k] for k in GPU_FEATURES if k in wanted}


def dependencies(features):
    f=set(features)
    if not f<=set(REGISTERED):raise ValueError('Unregistered feature')
    return dict(head=bool(f & set(HEAD+[x for x in ENTROPY if x.startswith('head')])),
                query=bool(f & set(QUERY+[x for x in ENTROPY if x.startswith('query')])),
                confidence=bool(f & set(CONFIDENCE)))


def base_features(layers,scores,prefix,end,required):
    """Original five-feature arithmetic, reducing only requested scalars."""
    wanted=set(required)
    if not wanted<=set(FEATURES):raise ValueError('Unknown base feature')
    layers=np.asarray(layers,dtype=np.float64);scores=np.asarray(scores,dtype=np.float64)
    if layers.shape!=(64,end) or scores.shape!=(end-prefix,) or not np.isfinite(layers).all() or not np.isfinite(scores).all() or (layers<0).any() or (scores<0).any():raise ValueError('Corrupt native attention')
    if scores.sum()<=0:return dict.fromkeys(required)
    out={};n=len(scores)
    if any(f.startswith('top') for f in wanted):
        order=np.argsort(-scores,kind='stable');p=scores/scores.sum()
        for percent in (1,20,50):
            name=f'top{percent}_mass'
            if name in wanted:out[name]=float(p[order[:math.floor(n*percent/100)]].sum())
    if wanted&{'group_agreement','group_top20_jaccard'}:
        a,b=layers[:32,prefix:].mean(0),layers[32:,prefix:].mean(0)
        if 'group_agreement' in wanted:
            denominator=np.linalg.norm(a)*np.linalg.norm(b);out['group_agreement']=float(np.dot(a,b)/denominator) if denominator>0 else None
        if 'group_top20_jaccard' in wanted:
            count=math.floor(n*.2);left,right=np.argsort(-a,kind='stable')[:count],np.argsort(-b,kind='stable')[:count]
            out['group_top20_jaccard']=len(np.intersect1d(left,right))/len(np.union1d(left,right)) if count else None
    return out
