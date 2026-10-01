"""Frozen scalar candidates for the offline attention feature study (no outcomes)."""
import math
import numpy as np

CONCENTRATION = ['top5_mass', 'top10_mass', 'mass_1_to_5', 'mass_5_to_10',
                 'mass_10_to_20', 'normalized_entropy', 'effective_support',
                 'mass90_fraction', 'mass95_fraction', 'gap1', 'gap5', 'gap10']
COVERAGE = [f'coverage{p}_{s}' for p in (1, 5, 10) for s in ('min', 'p10', 'median', 'std')]
COVERAGE += [f'band{b}_coverage1' for b in range(4)]
DISAGREEMENT = [f'band{b}_{b+1}_{s}' for b in range(3) for s in ('jaccard1', 'jsd')]
GEOMETRY = [f'geometry{p}_{s}' for p in (1, 5, 10) for s in ('position_std', 'adjacent_fraction')]
ADDITIONS = CONCENTRATION + COVERAGE + DISAGREEMENT + GEOMETRY
DEFINITIONS = dict(
    version='attention-study-features-v1', additions=ADDITIONS,
    region='[first chunk end, fresh suffix start); N eligible positions',
    global_distribution='native FP32 scores cast to float64 and divided by eligible sum; stable descending native score, ascending position ties; k=floor(N*percent/100)',
    layer_distribution='equal-rank float64 mean, then each layer normalized over eligible positions',
    band_distribution='mean of 16 equal-rank float64 unnormalized layers, then normalize within eligible region',
    mass='sum p over native global top-k; increments are differences of masses',
    normalized_entropy='-sum(p*log(p))/log(N); zero terms omitted',
    effective_support='1/(N*sum(p*p))',
    mass_fraction='smallest count whose cumulative ranked mass reaches 0.90/0.95, divided by N',
    boundary_gap='(p[k-1]-p[k])/p[k-1]; zero denominator gives zero; absent boundary undefined',
    coverage='each normalized layer mass on global mask; min, linear p10/median, population std; bands use same global 1% mask',
    disagreement='adjacent fixed bands [0:16],[16:32],[32:48],[48:64]; stable top1% Jaccard; Jensen-Shannon divergence in natural-log units (not square root)',
    geometry='population std of selected relative positions/(N-1); adjacency=count(diff(sorted positions)==1)/(k-1); k<2 undefined',
    invalid='nonfinite, negative, malformed source arrays are corruption errors; zero-mass/undefined features are null and force dense for policies requiring them')


def additional_features(layers, scores, prefix, end, required=None):
    """Input is resident equal-rank float64 (64,end) and native FP32 (end-prefix,)."""
    wanted=set(ADDITIONS if required is None else required)
    if not wanted <= set(ADDITIONS):
        raise ValueError('Unknown feature')
    n=end-prefix
    if (layers.shape != (64,end) or scores.shape != (n,) or scores.dtype != np.float32
            or not np.isfinite(layers).all() or not np.isfinite(scores).all()
            or (layers<0).any() or (scores<0).any()):
        raise ValueError('Corrupt resident attention')
    result=dict.fromkeys(wanted)
    if n<2 or scores.sum(dtype=np.float64)<=0:
        return result
    p=scores.astype(np.float64);p/=p.sum()
    order=np.argsort(-scores,kind='stable')
    counts={q:math.floor(n*q/100) for q in (1,5,10,20)}
    masks={q:order[:k] for q,k in counts.items()}
    masses={q:float(p[m].sum()) for q,m in masks.items()}
    values={f'top{q}_mass':masses[q] for q in (5,10)}
    values.update({f'mass_{a}_to_{b}':masses[b]-masses[a] for a,b in ((1,5),(5,10),(10,20))})
    if wanted & set(CONCENTRATION):
        nz=p[p>0]
        values['normalized_entropy']=float(-np.dot(nz,np.log(nz))/np.log(n))
        values['effective_support']=float(1/(n*np.dot(p,p)))
        cumulative=np.cumsum(p[order])
        for q in (90,95):values[f'mass{q}_fraction']=float((np.searchsorted(cumulative,q/100)+1)/n)
        for q in (1,5,10):
            k=counts[q]
            values[f'gap{q}']=None if not 0<k<n else (float((p[order[k-1]]-p[order[k]])/p[order[k-1]]) if p[order[k-1]] else 0.)
    if wanted & set(COVERAGE+DISAGREEMENT):
        eligible=layers[:,prefix:end]
        if wanted & set(COVERAGE):
            totals=eligible.sum(1)
            for q in (1,5,10):
                if (totals>0).all():
                    mass=eligible[:,masks[q]].sum(1)/totals
                    values.update({f'coverage{q}_min':float(mass.min()),f'coverage{q}_p10':float(np.percentile(mass,10,method='linear')),
                                   f'coverage{q}_median':float(np.median(mass)),f'coverage{q}_std':float(mass.std(ddof=0))})
        bands=[]
        for b in range(4):
            a=eligible[b*16:(b+1)*16].mean(0);total=a.sum();a=a/total if total>0 else None;bands.append(a)
            values[f'band{b}_coverage1']=float(a[masks[1]].sum()) if a is not None else None
        if wanted & set(DISAGREEMENT):
            for b,(a,c) in enumerate(zip(bands,bands[1:])):
                if a is None or c is None:continue
                k=counts[1];left=np.argsort(-a,kind='stable')[:k];right=np.argsort(-c,kind='stable')[:k]
                values[f'band{b}_{b+1}_jaccard1']=float(len(np.intersect1d(left,right))/len(np.union1d(left,right))) if k else None
                mid=(a+c)/2
                values[f'band{b}_{b+1}_jsd']=float(.5*sum(np.sum(v[v>0]*np.log(v[v>0]/mid[v>0])) for v in (a,c)))
    if wanted & set(GEOMETRY):
        for q in (1,5,10):
            selected=np.sort(masks[q]);k=len(selected)
            values[f'geometry{q}_position_std']=float(selected.std(ddof=0)/(n-1)) if k>=2 else None
            values[f'geometry{q}_adjacent_fraction']=float(np.mean(np.diff(selected)==1)) if k>=2 else None
    return {k:values.get(k) for k in ADDITIONS if k in wanted}
