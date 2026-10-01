#!/usr/bin/env python3
"""Independently verify final overall winners across primary and ablation grids."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from runner import attention_study as a
from runner.deeper_router_study import DEFAULT
from runner.attention_features import ADDITIONS
from runner.setups import atomic_json,file_hash


def verify(output):
    root=output/'finalized';report=a.read(root/'report.json');matrix=a.read(a.EXT/'matrix.json')
    independent=a.read(output/'independent-validation.json')
    if not independent['resume_hashes_and_mtimes_unchanged']:raise ValueError('Primary independent audit required')
    def rank(c):return (c['ttft'],-c['accuracy'],len(c['extras']),c['leaves'],c['setting']['depth'],tuple(ADDITIONS.index(f) for f in c['extras']),c['index'])
    old=a.sealed(output/'incumbent.json');best={str(old['setting']['depth']):old};count=0
    for p in (output/'search').glob('*.json'):
        for c in a.sealed(p)['candidates']:
            count+=1
            if not c['eligible']:continue
            d=str(c['setting']['depth'])
            if d not in best or rank(c)<rank(best[d]):best[d]=c
    candidates={'overall-winner':min(best.values(),key=rank),**{'depth-'+d:c for d,c in best.items()}}
    summaries={r['policy']:r for r in report['policies']}
    for label,c in candidates.items():
        saved=summaries[label]
        if saved['features']!=c['extras'] or saved['setting']!=c['setting'] or saved['actions']!=c['actions'] or saved['leaves']!=c['leaves'] or abs(saved['ttft_seconds']-c['ttft'])>1e-10:raise ValueError('Final global ranking differs')
    rows=[a.sealed(output/'features'/f'{pid}.json')['features'] for pid in matrix['ids']];names=[v['id'] for v in matrix['actions']];checks=0
    def reference(node,row):
        if 'action' in node:return node['action']
        return reference(node['le'] if row[node['feature']]<=node['threshold'] else node['gt'],row)
    for p in (root/'trees').glob('*.json'):
        policy=a.sealed(p);choices=[]
        for row in rows:
            predicted=reference(policy['tree'],row)
            if a.decide(policy,row)!=predicted:raise ValueError('Decision mismatch')
            choices.append(names.index(predicted));checks+=1
        if p.stem in candidates and choices!=candidates[p.stem]['choices']:raise ValueError('Winning choices mismatch')
        def boundaries(node,subset):
            nonlocal checks
            if 'action' in node:return
            k=node['feature'];t=node['threshold']
            for value in (float(np.nextafter(t,-np.inf)),t,float(np.nextafter(t,np.inf))):
                row=dict(subset[0],**{k:value})
                if a.decide(policy,row)!=reference(policy['tree'],row):raise ValueError('Boundary mismatch')
                checks+=1
            boundaries(node['le'],[r for r in subset if r[k]<=t]);boundaries(node['gt'],[r for r in subset if r[k]>t])
        boundaries(policy['tree'],rows)
        for k in policy['feature_names']:
            for value in (None,float('nan'),float('inf'),-float('inf')):
                if a.decide(policy,dict(rows[0],**{k:value}))!='nocache':raise ValueError('Invalid fallback failed')
                checks+=1
            row=dict(rows[0]);del row[k]
            if a.decide(policy,row)!='nocache':raise ValueError('Missing fallback failed')
            checks+=1
        score=math.fsum(matrix['scores'][i][j] for i,j in enumerate(choices))/260
        floor=math.fsum(r[0] for r in matrix['scores'])/260-.02
        cost=math.fsum(matrix['answer_ttft'][i][j]+(matrix['probe_ttft'][i] if j==0 else 0) for i,j in enumerate(choices))/260+report['primary_traversal_seconds']
        saved=summaries[p.stem]
        if score<floor-1e-12 or abs(100*score-saved['accuracy_percent'])>1e-10 or abs(cost-saved['ttft_seconds'])>1e-10 or {v:choices.count(j) for j,v in enumerate(names)}!=saved['actions']:raise ValueError('Final metrics differ')
    def snapshot():return {str(p.relative_to(root)):(file_hash(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    before=snapshot()
    subprocess.run([sys.executable,str(Path(__file__).with_name('deeper_router_results.py')),'--output',str(output)],check=True)
    if before!=snapshot():raise ValueError('Finalization rewrote completed results')
    receipt=dict(all_candidate_rankings_checked=count,final_export_checks=checks,final_decisions=8*260,
        primary_independent_validation_sha256=file_hash(output/'independent-validation.json'),
        final_completion_sha256=file_hash(root/'complete.json'),finalization_hashes_and_mtimes_unchanged=True,
        original_study_unchanged=independent['prior_study_unchanged'],verifier_sha256=file_hash(__file__))
    atomic_json(output/'delivery-validation.json',receipt);print(receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    args=p.parse_args();verify(args.output.resolve())
