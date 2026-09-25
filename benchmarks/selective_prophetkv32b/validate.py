"""Validate measured inference, saved score replay, priming comparison and retirement."""
import sys,math
from pathlib import Path
import numpy as np
from prepare import H,D,sha
sys.path.insert(0,str(H/'runtime'))
from common import load,dump
import suite

def select(scores,ratio,prefix=0):
    return np.sort(np.argsort(-np.asarray(scores),kind='stable')[:math.floor(len(scores)*ratio)])+prefix

def validate(m,p,path):
    path=Path(path);r=suite.validate(D,p,m,m['case'],path)
    assert sha(m['input_path'])==m['input_sha256']
    assert r['max_model_len']==65920 and r['max_output_tokens']==256
    for k in ['cache_construction_seconds','cache_readiness_seconds','cache_build_seconds',
              'prime_seconds','prime_export_seconds','artifact_export_seconds','retirement_seconds']:
        assert math.isfinite(r[k]) and r[k]>=0,k
    files=[path,path.with_suffix('.log'),path.with_suffix('.diagnostics.json')]
    arithmetic={}
    if m['case']!='baseline':
        ds=sorted(load(path.with_suffix('.diagnostics.json')),key=lambda x:x['rank'])
        assert len(r['priming_comparison'])==4
        for w,comp in zip(ds,sorted(r['priming_comparison'],key=lambda x:x['rank'])):
            assert comp['prior_close'] and comp['observer_removed'] and comp['layers']==[45,48,50,56,58]
            for event in w['diagnostics']:
                if event['kind']=='layer_counts': assert event['selected_set_verified']
            e=next(e for e in w['diagnostics'] if e['kind']=='prophetkv_selection')
            assert e['scoring_layers']==[45,48,50,56,58] and e['sequential_project_layers']==59 and e['suffix_forward_layers']==58
            assert e['normalization']=='all_context_keys' and e['fusion']=='sum_layers_fp32'
            prime=Path(comp['path']);files.append(prime)
            with np.load(prime) as z:
                assert np.array_equal(z['native_scores'],np.asarray(e['scores'],dtype=np.float32)), 'Prime/measured scores differ'
                assert np.array_equal(z['selected_positions'],e['selected_positions'])
        prefix=m['boundaries'][1];ratio=int(m['case'].split('-')[-1])/100
        # Reconstruct the exact prior offline equal-head float64 objective.
        a=[np.load(m['prior_capture_stem']+f'.rank{i}.npz')['layers'][[45,48,50,56,58]].astype(np.float64) for i in range(4)]
        prior64=np.mean(np.stack(a),axis=0).mean(axis=0)[prefix:]
        fp32=np.asarray(e['scores'],dtype=np.float32);chosen=np.asarray(e['selected_positions'])
        assert np.array_equal(select(fp32,ratio,prefix),chosen)
        offline=select(prior64,ratio,prefix)
        arithmetic=dict(fp32_vs_prior_float64_symmetric_difference=int(len(np.setxor1d(chosen,offline))),
            selected_count=len(chosen),fp32_vs_prior_float64_max_abs_mean_score_difference=float(np.max(np.abs(fp32.astype(np.float64)/5-prior64))))
        replay=path.with_suffix('.scores.npz')
        np.savez_compressed(replay,scores=fp32,selected_positions=chosen,prior_float64_scores=prior64,
                            prior_float64_selected_positions=offline)
        files.append(replay)
    receipt=dict(complete=True,case=m['case'],sample=m['id'],arithmetic=arithmetic,
                 sha256={str(f):sha(f) for f in files})
    dump(path.with_suffix('.validated.json'),receipt)
    return r
