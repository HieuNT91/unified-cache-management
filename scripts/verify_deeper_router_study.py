#!/usr/bin/env python3
"""Independent saved-record, export, tree-shape and immutable-resume audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_attention_feature_study import audit_sources, audit_exports
from runner.deeper_router_study import DEFAULT, SOURCE, GRID
from runner import attention_study as prior
from runner.setups import atomic_json, file_hash


def snapshot(output):
    return {str(p.relative_to(output)):(file_hash(p),p.stat().st_mtime_ns)
            for p in output.rglob('*') if p.is_file() and p.name not in ('study.lock','run.log','independent-validation.json')}


def audit(output):
    sources=audit_sources(output);checks=audit_exports(output)
    matrix=prior.read(prior.EXT / 'matrix.json');report=prior.read(output / 'report.json')
    floor=math.fsum(r[0] for r in matrix['scores'])/260-.02
    if abs(100*floor-report['accuracy_floor_percent'])>1e-10:raise ValueError('Floor differs')
    summaries={r['policy']:r for r in report['policies']}
    for p in (output / 'trees').glob('*.json'):
        policy=prior.sealed(p);h=policy['setting']
        def walk(n,indices):
            if 'action' in n:
                if len(indices)<h['min_leaf'] or len(indices)!=n['training_samples']:raise ValueError('Leaf count/minimum failed')
                return 0,1
            feature=n['feature'];threshold=n['threshold']
            left=[i for i in indices if rows[i][feature]<=threshold];right=[i for i in indices if rows[i][feature]>threshold]
            ld,ll=walk(n['le'],left);rd,rl=walk(n['gt'],right)
            return 1+max(ld,rd),ll+rl
        rows=[prior.sealed(output / 'features' / f'{pid}.json')['features'] for pid in matrix['ids']]
        depth,leaves=walk(policy['tree'],list(range(260)))
        if depth>h['depth'] or leaves>2**h['depth'] or depth!=summaries[p.stem]['realized_depth'] or leaves!=summaries[p.stem]['leaves']:raise ValueError('Depth/leaf maximum failed')
    # Independently rank full search candidates by the documented exact order.
    from runner.attention_features import ADDITIONS
    def rank(c):return (c['ttft'],-c['accuracy'],len(c['extras']),c['leaves'],c['setting']['depth'],tuple(ADDITIONS.index(f) for f in c['extras']),c['index'])
    selection=prior.sealed(output / 'selection.json')
    best={};counts={'reused':0,'fitted':0};total=0
    for p in (output / 'search').glob('*.json'):
        unit=prior.sealed(p)
        if len(unit['candidates'])!=4320:raise ValueError('Incomplete grid')
        for k in counts:counts[k]+=unit[k]
        for i,c in enumerate(unit['candidates']):
            if c['index']!=i or c['setting']!=GRID[i]:raise ValueError('Setting order differs')
            choices=c['choices']
            score=math.fsum(matrix['scores'][j][a] for j,a in enumerate(choices))/260
            elapsed=math.fsum(matrix['answer_ttft'][j][a]+(matrix['probe_ttft'][j] if a==0 else 0) for j,a in enumerate(choices))/260+report['pinned_primary_traversal_seconds']
            if abs(score-c['accuracy'])>1e-12 or abs(elapsed-c['ttft'])>1e-10 or c['eligible']!=(score>=floor-1e-12):raise ValueError('Independent candidate metric differs')
            if c['eligible'] and p.stem in selection['primary_subsets']:
                d=str(c['setting']['depth'])
                if d not in best or rank(c)<rank(best[d]):best[d]=c
            total+=1
    old=prior.sealed(output / 'incumbent.json');d=str(old['setting']['depth'])
    if rank(old)<rank(best[d]):best[d]=old
    selection=prior.sealed(output / 'selection.json')
    for d,c in best.items():
        if rank(c)!=rank(selection['best_by_depth'][d]) or c['choices']!=selection['best_by_depth'][d]['choices']:raise ValueError('Per-depth selection differs')
    winner=min([old,*best.values()],key=rank)
    if rank(winner)!=rank(selection['winner']):raise ValueError('Overall selection differs')
    if total!=report['candidate_checks'] or counts['reused']!=report['reused_settings'] or counts['fitted']!=report['new_fits']:raise ValueError('Search accounting differs')
    return dict(source_answers=sources['answers_checked'],source_probes=sources['probes_checked'],export_checks=checks,
                independently_checked_candidates=total,accuracy_floor_percent=100*floor,per_depth_and_overall_selection_verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args()
    output=args.output.resolve();before=snapshot(output);source_before=snapshot(SOURCE)
    result=audit(output)
    subprocess.run([sys.executable,str(Path(__file__).with_name('deeper_router_study.py')),'resume','--output',str(output)],check=True)
    if before!=snapshot(output) or source_before!=snapshot(SOURCE):raise ValueError('Resume changed completed artifacts')
    result.update(resume_hashes_and_mtimes_unchanged=True,prior_study_unchanged=True,completed_files=len(before),verifier_sha256=file_hash(__file__))
    atomic_json(output / 'independent-validation.json',result)
    print(result)

if __name__=='__main__':main()
