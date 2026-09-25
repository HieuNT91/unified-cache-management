"""Fresh result validation: exact replay, TP agreement, lifecycle and scope provenance."""
import sys,math
from pathlib import Path
import numpy as np
from prepare import H,D,sha
sys.path.insert(0,str(H/'runtime'))
from common import load,dump
from query_config import configuration,positions
import suite

def select(scores,ratio,prefix=0):
    return np.sort(np.argsort(-np.asarray(scores),kind='stable')[:math.floor(len(scores)*ratio)])+prefix

def validate(m,p,path):
    path=Path(path);cfg=configuration(m['case']);query=positions(m,m['case'])
    r=suite.validate(D,p,m|dict(query={'positions':query}),m['case'],path)
    assert sha(m['input_path'])==m['input_sha256']
    assert r['max_model_len']==65920 and r['max_output_tokens']==256
    assert r['query_scope']==cfg['query_scope'] and r['layer_scope']==cfg['layer_scope'] and r['query_positions']==query
    for k in ['cache_construction_seconds','cache_readiness_seconds','cache_build_seconds','prime_seconds',
              'prime_export_seconds','artifact_export_seconds','retirement_seconds']:
        assert math.isfinite(r[k]) and r[k]>=0,k
    session=load(r['session_path']);ready=session['warmup_readiness']
    assert ready['complete'] and ready['verified_shards']==8
    files=[path,path.with_suffix('.log'),path.with_suffix('.diagnostics.json')]
    ds=sorted(load(path.with_suffix('.diagnostics.json')),key=lambda x:x['rank'])
    assert len(r['priming_comparison'])==4
    for w,comp in zip(ds,sorted(r['priming_comparison'],key=lambda x:x['rank'])):
        assert comp['rank']==w['rank'] and comp['observer_removed'] and comp['layers']==cfg['scoring_layers']
        assert comp['query_scope']==cfg['query_scope'] and comp['layer_scope']==cfg['layer_scope']
        if cfg['query_scope']=='full_question':assert comp['prior_close'] and comp['historical_match_required']
        for event in w['diagnostics']:
            if event['kind']=='layer_counts':assert event['selected_set_verified']
        e=next(e for e in w['diagnostics'] if e['kind']=='prophetkv_selection')
        assert e['scoring_layers']==cfg['scoring_layers'] and e['query_scope']==cfg['query_scope'] and e['layer_scope']==cfg['layer_scope']
        assert e['sequential_project_layers']==(64 if cfg['layer_scope']=='all64' else 59)
        assert e['suffix_forward_layers']==(64 if cfg['layer_scope']=='all64' else 58)
        assert e['normalization']=='all_context_keys' and e['fusion']==('mean_all64_fp32' if cfg['layer_scope']=='all64' else 'sum_layers_fp32')
        prime=Path(comp['path']);files.append(prime)
        with np.load(prime) as z:
            assert z['layers'].shape==(len(cfg['scoring_layers']),m['boundaries'][-2])
            assert np.isfinite(z['layers']).all() and (z['layers']>=0).all()
            assert np.allclose(z['layers'].sum(1,dtype=np.float64),1,rtol=2e-4,atol=2e-4)
            assert np.array_equal(z['native_scores'],np.asarray(e['scores'],dtype=np.float32)),'Prime/measured scores differ'
            assert np.array_equal(z['selected_positions'],e['selected_positions'])
            assert np.array_equal(z['query_positions'],query)
    prefix=m['boundaries'][1];ratio=cfg['budget']/100
    fp32=np.asarray(e['scores'],dtype=np.float32);chosen=np.asarray(e['selected_positions'])
    assert np.array_equal(select(fp32,ratio,prefix),chosen)
    if cfg['query_scope']=='full_question':
        # Historical captures contain per-rank local means and the native all-layer reduction.
        if cfg['layer_scope']=='all64':
            with np.load(m['prior_capture_stem']+'.rank0.npz') as z:historical=z['native_scores']/np.float32(64)
            assert np.allclose(fp32,historical,rtol=2e-5,atol=2e-7)
            assert np.array_equal(select(historical,ratio,prefix),chosen),'Historical all64 mask changed'
        else:
            # Compare exact retained selected-layer scores at another budget: scores are budget independent.
            from prepare import SELECTIVE
            historical=load(SELECTIVE/'records'/m['id']/'selective_prophetkv-20.diagnostics.json')
            scores=next(d['scores'] for d in historical[0]['diagnostics'] if d['kind']=='prophetkv_selection')
            assert np.array_equal(fp32,np.asarray(scores,dtype=np.float32))
            assert np.array_equal(select(scores,ratio,prefix),chosen)
    replay=path.with_suffix('.scores.npz')
    if replay.exists():
        with np.load(replay) as saved:
            assert np.array_equal(saved['scores'],fp32) and np.array_equal(saved['selected_positions'],chosen)
    else:np.savez_compressed(replay,scores=fp32,selected_positions=chosen)
    files.append(replay)
    receipt=dict(complete=True,case=m['case'],sample=m['id'],query_scope=cfg['query_scope'],layer_scope=cfg['layer_scope'],
        sha256={str(f):sha(f) for f in files})
    certificate=path.with_suffix('.validated.json')
    if certificate.exists():assert load(certificate)==receipt,'Previously accepted validation changed'
    else:dump(certificate,receipt)
    return r
