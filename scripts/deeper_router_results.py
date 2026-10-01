#!/usr/bin/env python3
"""Finalize overall ranking including the bounded, one-round drop-one diagnostics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
import fcntl
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from runner import attention_study as a
from runner.deeper_router_study import DEFAULT,tree_shape
from runner.setups import atomic_json,file_hash


def finalize(output):
    out=output/'finalized';out.mkdir(exist_ok=True)
    source=a.sealed(output/'complete.json')
    for p,h in source['files'].items():
        if file_hash(output/p)!=h:raise ValueError('Study artifact changed')
    if (out/'complete.json').exists():
        done=a.sealed(out/'complete.json')
        if done['source_complete_sha256']!=file_hash(output/'complete.json') or done['script_sha256']!=file_hash(__file__):raise ValueError('Finalizer source changed')
        for p,h in done['files'].items():
            if file_hash(out/p)!=h:raise ValueError('Final report changed')
        print('Final overall ranking already verified; artifacts unchanged.');return
    matrix=a.read(a.EXT/'matrix.json');names=[v['id'] for v in matrix['actions']]
    rows=[a.sealed(output/'features'/f'{pid}.json')['features'] for pid in matrix['ids']]
    old=a.sealed(output/'incumbent.json');selection=a.sealed(output/'selection.json')
    by_depth={str(old['setting']['depth']):old};total=reused=fitted=0
    source_reused=completed_fits=expanded_reused=0
    primary_tags=set(selection['primary_subsets'])
    categories={};subsets=[]
    for p in sorted((output/'search').glob('*.json')):
        unit=a.sealed(p);total+=unit['searched'];reused+=unit['reused'];fitted+=unit['fitted']
        source_reused+=unit.get('first_pass_reused',unit['reused'])
        completed_fits+=unit.get('first_pass_fitted',unit['fitted'])
        if 'preserved_source_sha256' in unit:expanded_reused+=unit['searched']
        for d,c in unit['best_by_depth'].items():
            if d not in by_depth or a.ranking(c)<a.ranking(by_depth[d]):by_depth[d]=c
        categories[tuple(unit['extras'])]='primary search' if p.stem in primary_tags else 'one-round drop-one refit'
        subsets.append(dict(tag=p.stem,additions=unit['extras'],settings=unit['searched'],reused=unit['reused'],fitted=unit['fitted']))
    winner=min([old,*by_depth.values()],key=a.ranking)
    policies=[('incumbent',old),('primary-winner',selection['winner']),('overall-winner',winner)]+[('depth-'+d,c) for d,c in sorted(by_depth.items())]
    timing=a.sealed(output/'cpu-overhead.json');oracle=a.read(a.ORACLE)['results']['0.02']['ttft_seconds']
    floor=math.fsum(r[0] for r in matrix['scores'])/260-.02
    summaries=[];changes=[];task_rows=[];checks=0
    for label,c in policies:
        export=dict(schema=a.SCHEMA,offline_only=True,feature_names=c['feature_names'],tree=c['tree'],actions=names,
            additions=c['extras'],setting=c['setting'],training_ids=matrix['ids'],heldout_ids=[],
            protocol_sha256=file_hash(output/'protocol.json'),missing_required_feature='nocache')
        checks+=a.check_export(export,c.get('raw_tree'),rows,names)
        # Separate scalar recursive traversal and fsum accounting, independent of
        # the fitting implementation and candidate metric reducer.
        def reference(node,row):
            if 'action' in node:return node['action']
            return reference(node['le'] if row[node['feature']]<=node['threshold'] else node['gt'],row)
        choices=[names.index(reference(c['tree'],row)) for row in rows]
        if choices!=c['choices'] or any(reference(c['tree'],r)!=a.decide(export,r) for r in rows):raise ValueError('Export choices differ')
        accuracy=math.fsum(matrix['scores'][i][j] for i,j in enumerate(choices))/260
        costs=[matrix['answer_ttft'][i][j]+(matrix['probe_ttft'][i] if j==0 else 0)+old['traversal_seconds'] for i,j in enumerate(choices)]
        ttft=math.fsum(costs)/260
        if accuracy<floor-1e-12 or abs(accuracy-c['accuracy'])>1e-12 or abs(ttft-c['ttft'])>1e-10:raise ValueError('Final metric/floor mismatch')
        counts={v:choices.count(j) for j,v in enumerate(names)}
        if counts!=c['actions']:raise ValueError('Final action counts differ')
        depth,leaves=tree_shape(c['raw_tree'],c['setting']['min_leaf'],c['setting']['depth'])
        a.sealed(out/'trees'/f'{label}.json',export)
        (out/'trees'/f'{label}.txt').write_text('OFFLINE ONLY. Missing/undefined required feature -> nocache.\n'+'\n'.join(a.rules(c['tree']))+'\n')
        repeats=[]
        for _ in range(9):
            start=time.perf_counter()
            for _ in range(20):
                for row in rows:
                    node=c['tree']
                    while 'feature' in node:node=node['le'] if row[node['feature']]<=node['threshold'] else node['gt']
            repeats.append((time.perf_counter()-start)/5200)
        traversal=float(np.median(repeats))
        incremental=timing['feature_sets'][','.join(c['extras'])]['mean_seconds'] if c['extras'] else 0
        sub=[]
        for i,(pid,j) in enumerate(zip(matrix['ids'],choices)):
            before=old['choices'][i]
            sub.append(dict(policy=label,prompt_id=pid,task=matrix['tasks'][i],old_action=names[before],new_action=names[j],changed=j!=before,
                accuracy=matrix['scores'][i][j],dense_accuracy=matrix['scores'][i][0],old_accuracy=matrix['scores'][i][before],ttft_seconds=costs[i],
                newly_one=j==1 and before!=1,newly_one_at_least_dense=j==1 and before!=1 and matrix['scores'][i][j]>=matrix['scores'][i][0]-1e-12))
        changes+=sub
        for task in sorted(set(matrix['tasks'])):
            rs=[r for r in sub if r['task']==task]
            task_rows.append(dict(policy=label,task=task,accuracy_percent=100*math.fsum(r['accuracy'] for r in rs)/20,
                loss_pp=100*math.fsum(r['dense_accuracy']-r['accuracy'] for r in rs)/20,ttft_seconds=math.fsum(r['ttft_seconds'] for r in rs)/20,
                **{v:sum(r['new_action']==v for r in rs) for v in names}))
        dense50=math.fsum(cost for cost,j in zip(costs,choices) if j in (0,5))/260
        summaries.append(dict(policy=label,origin=categories.get(tuple(c['extras']),'incumbent'),features=c['extras'],setting=c['setting'],
            maximum_depth=c['setting']['depth'],realized_depth=depth,leaves=leaves,accuracy_percent=100*accuracy,loss_pp=100*(floor+.02-accuracy),
            ttft_seconds=ttft,actions=counts,improvement_seconds=old['ttft']-ttft,improvement_percent=100*(old['ttft']-ttft)/old['ttft'],
            remaining_hindsight_gap_seconds=ttft-oracle,gap_closed_percent=100*(old['ttft']-ttft)/(old['ttft']-oracle),
            incremental_cpu_seconds=incremental,measured_traversal_seconds=traversal,cpu_adjusted_ttft_seconds=ttft-old['traversal_seconds']+traversal+incremental,
            dense50_mean_ttft_contribution=dense50,dense50_ttft_percent=100*dense50/ttft,
            dense_reduction=old['actions']['nocache']-counts['nocache'],fifty_reduction=old['actions']['prophetkv-50']-counts['prophetkv-50'],
            changed_decisions=sum(r['changed'] for r in sub),newly_one=sum(r['newly_one'] for r in sub),newly_one_at_least_dense=sum(r['newly_one_at_least_dense'] for r in sub)))
    ablations=[]
    anchor=selection['winner']
    for r in selection['drop_one']:
        c=r['best']
        ablations.append(dict(dropped=r['dropped'],anchor='primary-winner',anchor_ttft=anchor['ttft'],ttft_seconds=c['ttft'],
            change_seconds=c['ttft']-anchor['ttft'],accuracy_percent=100*c['accuracy'],depth=c['setting']['depth'],leaves=c['leaves']))
    a.csv_write(out/'drop-one-ablations.csv',ablations)
    a.csv_write(out/'per-task.csv',task_rows);a.csv_write(out/'decision-changes.csv',changes)
    report=dict(training_only=True,training_samples=260,heldout_samples=0,settings_per_subset=4320,total_candidate_checks=total,
        reused_settings_in_final_pass=reused,new_fits_in_final_pass=fitted,
        original_depth1_3_reused=source_reused,completed_new_fits_across_passes=completed_fits,expanded_checkpoint_settings_reused=expanded_reused,
        fit_accounting="Completed subset settings only; interrupted uncommitted work is not included.",subsets=subsets,accuracy_floor_percent=100*floor,
        incumbent_ttft_seconds=old['ttft'],oracle_ttft_seconds=oracle,policies=summaries,
        ablation_scope='One round of full-grid drop-one refits of the primary-search winner. All their candidates are included in final per-depth and overall ranking. No recursive feature elimination.',
        primary_traversal_seconds=old['traversal_seconds'],export_checks=checks,
        limitations='All260 prompts used for fitting, adaptive selection and scoring. Larger trees increase overfitting risk; training-set findings only. Overall accuracy constraint, not per-task. Finite greedy search. Mixed measured outcome sessions and assumed integrated probe reuse. Live transfer/synchronization/switching costs excluded. Offline-only; no new inference, runtime integration or deployment.')
    inc=summaries[0]
    for r in summaries:r['cpu_adjusted_improvement_seconds']=inc['cpu_adjusted_ttft_seconds']-r['cpu_adjusted_ttft_seconds']
    w=next(r for r in summaries if r['policy']=='overall-winner')
    report['result']='improvement found' if w['ttft_seconds']<old['ttft']-1e-9 else 'no further improvement found'
    atomic_json(out/'report.json',report)
    lines=[report['result'].upper(),report['limitations'],report['ablation_scope'],
        'Primary accounting: selected answer TTFT + frozen traversal allowance + saved probe TTFT only for dense.',
        f'Incumbent {old["ttft"]:.6f}s; hindsight {oracle:.6f}s; overall accuracy floor {100*floor:.7f}%.',
        f'{len(subsets)} distinct subsets x4320 = {total} evaluated settings; {source_reused} original depth1-3 results reused, {completed_fits} completed new fits across passes.',
        'Policy | accuracy% | loss pp | TTFT s | max/actual depth | leaves | dense/1/5/10/20/50']
    for r in summaries:lines.append(f'{r["policy"]} | {r["accuracy_percent"]:.4f} | {r["loss_pp"]:.4f} | {r["ttft_seconds"]:.6f} | {r["maximum_depth"]}/{r["realized_depth"]} | {r["leaves"]} | '+ '/'.join(str(r['actions'][v]) for v in names))
    lines+=['',f'Improvement {w["improvement_seconds"]:.6f}s ({w["improvement_percent"]:.2f}%). Remaining hindsight gap {w["remaining_hindsight_gap_seconds"]:.6f}s; closes {w["gap_closed_percent"]:.2f}% of incumbent gap.',
        f'Overhead-adjusted estimate {w["cpu_adjusted_ttft_seconds"]:.6f}s, including {1000*w["incremental_cpu_seconds"]:.3f}ms incremental resident-array features and {1e6*w["measured_traversal_seconds"]:.3f}us traversal.',
        f'Dense {old["actions"]["nocache"]} -> {w["actions"]["nocache"]}; 50% {old["actions"]["prophetkv-50"]} -> {w["actions"]["prophetkv-50"]}. Dense/50% contribution {inc["dense50_ttft_percent"]:.2f}% -> {w["dense50_ttft_percent"]:.2f}%.',
        f'{w["changed_decisions"]} decisions changed; {w["newly_one"]} newly select1%, {w["newly_one_at_least_dense"]} at least dense on the saved training outcomes.',
        '', 'Overall winner rules:',*a.rules(winner['tree'])]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    a.sealed(out/'validation.json',dict(export_checks=checks,export_decisions=len(policies)*260,all_exports_feasible=True,all_candidate_subset_winners_ranked=True,
        maximum_depth_and_leaf_counts_verified=True,minimum_leaf_counts_verified=True,independent_scalar_decisions_and_metrics=True))
    a.sealed(out/'complete.json',dict(source_complete_sha256=file_hash(output/'complete.json'),script_sha256=file_hash(__file__),
        files={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print('\n'.join(lines[:21]))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args();output=args.output.resolve()
    with (output/'finalize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finalize(output)
