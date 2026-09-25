"""Validate exact native/union/oracle masks, actual budgets and all-rank lifecycle."""
import math
from pathlib import Path
import numpy as np
from prepare import H,D,sha
from common import load,dump
from query_config import configuration,positions,oracle_positions
import suite

def select(scores,ratio,prefix=0):
    return np.sort(np.argsort(-np.asarray(scores),kind='stable')[:math.floor(len(scores)*ratio)])+prefix

def expected_mask(scores,m,case):
    cfg=configuration(case);oracle=np.asarray(oracle_positions(m,case),dtype=np.int64)
    native=np.empty(0,dtype=np.int64) if cfg['selection_mode']=='target_only' else select(scores,cfg['budget']/100,m['boundaries'][1])
    return native,np.union1d(native,oracle)

def validate(root,m,p,path):
    root=Path(root);path=Path(path);cfg=configuration(m['case']);query=positions(m,m['case'])
    r=suite.validate(root,p,m,m['case'],path)
    assert sha(m['input_path'])==m['input_sha256']
    assert r['max_model_len']==65920 and r['max_output_tokens']==256
    for k in ['query_scope','layer_scope','selection_mode','budget']:assert r[k]==cfg[k],k
    assert r['query_positions']==query
    for k in ['cache_construction_seconds','cache_readiness_seconds','cache_build_seconds','prime_seconds','prime_export_seconds','artifact_export_seconds','retirement_seconds']:
        assert math.isfinite(r[k]) and r[k]>=0,k
    session=load(r['session_path']);ready=session['warmup_readiness'];assert ready['complete'] and ready['verified_shards']==8
    files=[path,path.with_suffix('.log'),path.with_suffix('.diagnostics.json')]
    ds=sorted(load(path.with_suffix('.diagnostics.json')),key=lambda x:x['rank'])
    assert len(r['priming_comparison'])==4
    for w,comp in zip(ds,sorted(r['priming_comparison'],key=lambda x:x['rank'])):
        assert comp['rank']==w['rank'] and comp['observer_removed'] and comp['layers']==cfg['scoring_layers']
        assert comp['selection_mode']==cfg['selection_mode'] and comp['layer_scope']==cfg['layer_scope']
        if m.get('prior_capture_stem') and cfg['scoring_layers']:assert comp['prior_close'] and comp['historical_match_required']
        for event in w['diagnostics']:
            if event['kind']=='layer_counts':assert event['selected_set_verified']
        e=next(e for e in w['diagnostics'] if e['kind']=='prophetkv_selection')
        for k in ['query_scope','layer_scope','selection_mode','scoring_layers']:assert e[k]==cfg[k],k
        only=cfg['selection_mode']=='target_only'
        assert e['sequential_project_layers']==(0 if only else 64 if cfg['layer_scope']=='all64' else 59)
        assert e['suffix_forward_layers']==(0 if only else 64 if cfg['layer_scope']=='all64' else 58)
        assert e['normalization']==('not_applicable' if only else 'all_context_keys')
        assert e['fusion']==('oracle_only' if only else 'mean_all64_fp32' if cfg['layer_scope']=='all64' else 'sum_layers_fp32')
        prime=Path(comp['path']);files.append(prime)
        with np.load(prime) as z:
            assert z['layers'].shape==(len(cfg['scoring_layers']),m['boundaries'][-2])
            assert np.isfinite(z['layers']).all() and (z['layers']>=0).all()
            if not only:assert np.allclose(z['layers'].sum(1,dtype=np.float64),1,rtol=2e-4,atol=2e-4)
            assert np.array_equal(z['native_scores'],np.asarray(e['scores'],dtype=np.float32)),'Prime/measured scores differ'
            for k in ['selected_positions','base_selected_positions']:assert np.array_equal(z[k],e[k]),k
            assert np.array_equal(z['query_positions'],query)
    scores=np.asarray(e['scores'],dtype=np.float32);native,chosen=expected_mask(scores,m,m['case'])
    assert np.array_equal(chosen,e['selected_positions']) and np.array_equal(native,e['base_selected_positions'])
    oracle=oracle_positions(m,m['case']);eligible=m['boundaries'][-2]-m['boundaries'][1]
    assert set(oracle)<=set(chosen.tolist())
    expected=dict(selected_context_tokens=len(chosen),eligible_context_tokens=eligible,effective_recompute_ratio=len(chosen)/eligible,
        native_selected_tokens=len(native),oracle_eligible_tokens=len(oracle),oracle_added_tokens=len(chosen)-len(native),
        oracle_exact_prefix_tokens=len(m.get('oracle',{}).get('already_exact_prefix_positions',[])))
    for k,v in expected.items():assert r[k]==v,(k,r[k],v)
    if not cfg['selection_mode']=='target_only' and m.get('prior_capture_stem'):
        if cfg['layer_scope']=='all64':
            with np.load(m['prior_capture_stem']+'.rank0.npz') as z:historical=z['native_scores']/np.float32(64)
            assert np.allclose(scores,historical,rtol=2e-5,atol=2e-7)
            assert np.array_equal(select(historical,cfg['budget']/100,m['boundaries'][1]),native),'Historical native mask changed'
    replay=path.with_suffix('.scores.npz');values=dict(scores=scores,selected_positions=chosen,base_selected_positions=native,oracle_positions=np.asarray(oracle,dtype=np.int64))
    if replay.exists():
        with np.load(replay) as saved:
            for k,v in values.items():assert np.array_equal(saved[k],v)
    else:np.savez_compressed(replay,**values)
    files.append(replay)
    receipt=dict(complete=True,case=m['case'],sample=m['id'],stage=p['stage'],actual_selection=expected,sha256={str(f):sha(f) for f in files})
    certificate=path.with_suffix('.validated.json')
    if certificate.exists():assert load(certificate)==receipt,'Accepted artifacts changed'
    else:dump(certificate,receipt)
    return r
