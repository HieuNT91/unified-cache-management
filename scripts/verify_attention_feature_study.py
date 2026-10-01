#!/usr/bin/env python3
"""Independent record-level audit and finished-study resume check, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from runner.attention_study import DEFAULT, EXT, BASE, SCHEMA, sealed, decide
from runner.setups import atomic_json, file_hash


def read(p):return json.loads(Path(p).read_text())


def audit_sources(output):
    matrix=read(EXT/'matrix.json');names=[r['id'] for r in matrix['actions']];pins={};checked=0
    for i,pid in enumerate(matrix['ids']):
        for j,action in enumerate(names):
            root=EXT if action in ('prophetkv-5','prophetkv-10') else BASE
            folder=root/'records'/action/pid;receipt=folder/'validated.json'
            if file_hash(receipt)!=matrix['acceptance_hashes'][str(receipt)]:raise ValueError('Matrix receipt mismatch')
            p=folder/'result.json';expected=read(receipt)['files']['result.json']
            if file_hash(p)!=expected:raise ValueError('Source outcome changed')
            r=read(p);pins[str(p)]=expected
            if (r['prompt_id']!=pid or r['input_sha256']!=matrix['input_hashes'][i] or r['accuracy']!=matrix['scores'][i][j]
                or r['timings']['answer_engine_ttft_seconds']!=matrix['answer_ttft'][i][j] or r['output_cap_reached']!=matrix['output_caps'][i][j]):
                raise ValueError('Matrix differs from accepted outcome')
            checked+=1
        folder=BASE/'records/probe'/pid;p=folder/'result.json';expected=read(folder/'validated.json')['files']['result.json']
        if file_hash(p)!=expected:raise ValueError('Probe source changed')
        r=read(p);pins[str(p)]=expected
        if r['features']!=matrix['features'][i] or r['timings']['probe_ttft_seconds']!=matrix['probe_ttft'][i]:raise ValueError('Probe matrix mismatch')
    return dict(matrix_sha256=file_hash(EXT/'matrix.json'),answers_checked=checked,probes_checked=260,source_pins=pins)


def audit_exports(output):
    matrix=read(EXT/'matrix.json');names=[r['id'] for r in matrix['actions']]
    rows=[sealed(output/'features'/f'{pid}.json')['features'] for pid in matrix['ids']]
    report=read(output/'report.json');summaries={r['policy']:r for r in report['policies']};checks=0
    def reference(node,row):
        if 'action' in node:return node['action']
        if row[node['feature']]<=node['threshold']:return reference(node['le'],row)
        return reference(node['gt'],row)
    for p in sorted((output/'trees').glob('*.json')):
        policy=sealed(p);label=p.stem;node=policy['tree'];choices=[]
        for r in rows:
            expected=reference(node,r)
            if decide(policy,r)!=expected:raise ValueError('Decision differs')
            choices.append(names.index(expected));checks+=1
        # Every threshold, including retained baseline: ancestor-compatible witness.
        def walk(n,subset):
            nonlocal checks
            if 'action' in n:return
            k=n['feature'];t=n['threshold'];witness=subset[0]
            for v in (float(np.nextafter(t,-np.inf)),t,float(np.nextafter(t,np.inf))):
                r=dict(witness,**{k:v})
                if decide(policy,r)!=reference(node,r):raise ValueError('Boundary mismatch')
                checks+=1
            walk(n['le'],[r for r in subset if r[k]<=t]);walk(n['gt'],[r for r in subset if r[k]>t])
        walk(node,rows)
        for k in policy['feature_names']:
            for value in (None,float('nan'),float('inf'),-float('inf')):
                r=dict(rows[0],**{k:value})
                if decide(policy,r)!='nocache':raise ValueError('Fallback mismatch')
                checks+=1
            r=dict(rows[0]);del r[k]
            if decide(policy,r)!='nocache':raise ValueError('Absent fallback mismatch')
            checks+=1
        accuracy=math.fsum(matrix['scores'][i][j] for i,j in enumerate(choices))/260
        baseline=math.fsum(r[0] for r in matrix['scores'])/260
        ttft=math.fsum(matrix['answer_ttft'][i][j]+(matrix['probe_ttft'][i] if j==0 else 0) for i,j in enumerate(choices))/260+report['pinned_primary_traversal_seconds']
        saved=summaries[label]
        if abs(100*accuracy-saved['accuracy_percent'])>1e-10 or abs(ttft-saved['ttft_seconds'])>1e-10 or baseline-accuracy>.02+1e-12:raise ValueError('Policy metric audit failed')
        if {a:choices.count(j) for j,a in enumerate(names)}!=saved['actions']:raise ValueError('Policy counts audit failed')
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--sources-only',action='store_true');args=parser.parse_args()
    output=args.output.resolve();sources=audit_sources(output)
    if args.sources_only:
        sealed(output/'record-source-validation.json',sources);print('Verified 1560 original outcome records and 260 probe records');return
    before={str(p.relative_to(output)):(file_hash(p),p.stat().st_mtime_ns) for folder in ('features','search') for p in (output/folder).glob('*.json')}
    checks=audit_exports(output)
    subprocess.run([sys.executable,str(Path(__file__).with_name('attention_feature_study.py')),'resume','--output',str(output)],check=True)
    after={str(p.relative_to(output)):(file_hash(p),p.stat().st_mtime_ns) for folder in ('features','search') for p in (output/folder).glob('*.json')}
    if before!=after:raise ValueError('Resume replaced a completed feature/search unit')
    atomic_json(output/'resume-validation.json',dict(source_record_checks=sources['answers_checked']+sources['probes_checked'],export_checks=checks,
        completed_units=len(before),hashes_and_mtimes_unchanged=True,full_resume_passed=True,verifier_sha256=file_hash(__file__)))
    print(f'Independent source/export audit and full resume passed; {len(before)} units unchanged.')

if __name__=='__main__':main()
