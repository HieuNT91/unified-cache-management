"""Unchanged 2592-setting all-training search over an expanded action matrix."""
import csv
import html
import itertools
import json
import shutil
import time
import uuid
from collections import Counter
from pathlib import Path
import numpy as np
from runner.corpus_train import fit_tree,leaf,leaf_count
from runner.router_policy import FEATURES
from runner.tree_policy import SCHEMA,FEATURE_DEFINITIONS,export,load,decide,actions
from runner.setups import atomic_json,file_hash,fingerprint

WEIGHTS=[0.,.1,.25,.5,1.,2.,3.,4.,5.,6.,8.,10.,12.,16.,20.,24.,32.,48.,64.,96.,128.,256.,512.,1024.]
GRID=[dict(objective=o,loss_mode=l,depth=d,min_leaf=m,penalty=p,weight=w)
      for o,l,d,m,p,w in itertools.product(['maximize-one','minimize-time'],['positive','signed'],[1,2,3],[5,10,20],[0.,.005,.02],WEIGHTS)]


def traverse(tree,features):
    while 'feature' in tree:tree=tree['le'] if features[tree['feature']]<=tree['threshold'] else tree['gt']
    return tree['action']


def compile_node(raw,names):
    if 'feature' not in raw:return dict(action=names[raw['action']],training_samples=raw['n'])
    return dict(feature=FEATURES[raw['feature']],threshold=raw['threshold'],le=compile_node(raw['left'],names),gt=compile_node(raw['right'],names))


def traversal_time(node,features):
    measurements=[]
    for _ in range(3):
        start=time.perf_counter()
        for f in features:traverse(node,f)
        measurements.append((time.perf_counter()-start)/len(features))
    return float(np.median(measurements))


def check_export(tree,raw,names,features):
    checks=list(features)
    def boundaries(n,values):
        if 'feature' not in n:return
        k=FEATURES[n['feature']];t=n['threshold']
        checks.extend(dict(values,**{k:float(v)}) for v in [np.nextafter(t,-np.inf),t,np.nextafter(t,np.inf)])
        boundaries(n['left'],dict(values,**{k:t}));boundaries(n['right'],dict(values,**{k:float(np.nextafter(t,np.inf))}))
    if raw is not None:boundaries(raw,dict.fromkeys(FEATURES,.5))
    else:
        def visit(n):
            if 'feature' not in n:return
            for v in (np.nextafter(n['threshold'],-np.inf),n['threshold'],np.nextafter(n['threshold'],np.inf)):
                checks.append(dict.fromkeys(FEATURES,.5)|{n['feature']:float(v)})
            visit(n['le']);visit(n['gt'])
        visit(tree['tree'])
    for f in checks:
        expected=names[leaf(raw,[f[k] for k in FEATURES])['action']] if raw is not None else traverse(tree['tree'],f)
        if decide(tree,f)['action']!=expected:raise ValueError('Export decision mismatch')
    for k in FEATURES:
        for value in (None,float('nan'),float('inf')):
            f=dict(features[0]);f[k]=value
            if decide(tree,f)['action']!='nocache':raise ValueError('Missing-feature fallback mismatch')
        f=dict(features[0]);del f[k]
        if decide(tree,f)['action']!='nocache':raise ValueError('Missing-feature fallback mismatch')
    return len(checks)+4*len(FEATURES)


def validate_matrix(matrix):
    names=list(actions(matrix['actions']));ids=matrix['ids'];n=len(ids)
    if n!=260 or len(set(ids))!=n or len(matrix['input_hashes'])!=n or len(set(matrix['input_hashes']))!=n:
        raise ValueError('Expected 260 distinct joined prompt identities and input hashes')
    if len(set(matrix['tasks']))!=13 or set(Counter(matrix['tasks']).values())!={20}:raise ValueError('Expected 20 prompts per task')
    x=np.asarray([[f[k] for k in FEATURES] for f in matrix['features']]);scores=np.asarray(matrix['scores']);times=np.asarray(matrix['answer_ttft']);probe=np.asarray(matrix['probe_ttft'])
    if x.shape!=(n,5) or scores.shape!=(n,len(names)) or times.shape!=scores.shape or probe.shape!=(n,):raise ValueError('Incomplete outcome matrix')
    if not all(np.isfinite(a).all() for a in (x,scores,times,probe)) or (scores<0).any() or (scores>1).any() or (times<=0).any() or (probe<0).any():raise ValueError('Invalid outcome values')
    return names,x,scores,times,probe


def train(output,matrix,previous,matrix_sha256,grid=None):
    output=Path(output);names,x,scores,times,probe=validate_matrix(matrix);grid=GRID if grid is None else grid
    if output.exists():
        done=json.loads((output/'complete.json').read_text())
        if done['matrix_sha256']!=matrix_sha256:raise ValueError('Completed training matrix changed')
        for name,digest in done['files'].items():
            if file_hash(output/name)!=digest:raise ValueError('Completed training artifact changed')
        return
    stage=output.with_name(output.name+'-staging-'+uuid.uuid4().hex);stage.mkdir(parents=True)
    started=time.time();n=len(x);features=matrix['features'];ids=matrix['ids'];dense=names.index('nocache');one=names.index('prophetkv-1')
    corrected=times.copy();corrected[:,dense]+=probe;base=float(scores[:,dense].mean());baseline_time=float(times[:,dense].mean())
    signed=scores[:,[dense]]-scores;positive=np.maximum(0,signed);normalized=corrected/float(np.median(times[:,dense]))
    non_one=np.ones_like(scores);non_one[:,one]=0
    costs={'maximize-one':non_one+1e-4*normalized,'minimize-time':normalized}
    protocol=dict(schema='expanded-all260-search-v1',matrix_sha256=matrix_sha256,previous_sha256=file_hash(previous),grid=grid,
        features=list(FEATURES),loss_limit_pp=2,evaluation='All 260 used for fitting, tuning and scoring; no held-out set',
        timing='Selected answer TTFT + tree traversal + saved probe TTFT only for dense',
        limitations='Training-set simulation with assumed integrated probe reuse. New 5%/10% timings are from a later measurement session. Incremental feature, synchronization and switching costs remain unmeasured. Finite greedy search, not a global optimum.')
    atomic_json(stage/'protocol.json',protocol)
    def metrics(choices,traversal):
        score=float(scores[np.arange(n),choices].mean());ttft=float(corrected[np.arange(n),choices].mean())+traversal
        return dict(accuracy=score,loss=base-score,ttft=ttft,traversal_seconds=traversal,one_count=int((choices==one).sum()),
            eligible=bool(base-score<=.02+1e-12),choices=choices.tolist(),actions={a:int((choices==j).sum()) for j,a in enumerate(names)})
    old=load(previous);old_choices=np.array([names.index(decide(old,f)['action']) for f in features])
    old_metrics=metrics(old_choices,traversal_time(old['tree'],features))
    old_candidate=dict(index=-1,setting=old['setting'],leaves=None,**old_metrics)
    if not old_candidate['eligible']:raise ValueError('Previous router no longer feasible')
    candidates=[];raw_trees={};nodes={-1:old['tree']}
    for index,h in enumerate(grid):
        labels=costs[h['objective']]+h['weight']*(positive if h['loss_mode']=='positive' else signed)
        raw=fit_tree(x,positive[:,[j for j in range(len(names)) if j!=dense]],h,cost_labels=labels)
        choices=np.asarray([leaf(raw,row)['action'] for row in x]);node=compile_node(raw,names)
        candidate=dict(index=index,setting=h,leaves=leaf_count(raw),**metrics(choices,traversal_time(node,features)))
        candidates.append(candidate);raw_trees[index]=raw;nodes[index]=node
        if (index+1)%200==0:print(f'Fitted {index+1}/{len(grid)}',flush=True)
    eligible=[c for c in [*candidates,old_candidate] if c['eligible']]
    complexity=lambda c:(c['setting']['depth'],c['leaves'] or 0,c['index'])
    fastest=min(eligible,key=lambda c:(c['ttft'],-c['accuracy'],-c['one_count'],*complexity(c)))
    maxone=min(eligible,key=lambda c:(-c['one_count'],c['ttft'],-c['accuracy'],*complexity(c)))
    if fastest['ttft']>old_metrics['ttft']:raise ValueError('Best feasible router regressed')
    atomic_json(stage/'search.json',candidates);atomic_json(stage/'selection.json',dict(min_ttft=fastest,max_one=maxone,previous=old_candidate))
    decisions=[];summary=[];checks=0
    for name,c in [('previous-router',old_candidate),('min-ttft',fastest),('max-one',maxone)]:
        tree=dict(schema=SCHEMA,actions=matrix['actions'],feature_names=list(FEATURES),feature_definitions=FEATURE_DEFINITIONS,
            id=name,primary=name=='min-ttft',tree=nodes[c['index']],training_costs=corrected.mean(0).tolist(),
            trainer=protocol['schema'],setting=c['setting'],training_ids=ids,heldout_ids=[],heldout_hashes={},
            provenance=dict(snapshot_sha256=matrix_sha256,training_run_sha256=file_hash(stage/'protocol.json'),corpus_protocol_sha256=matrix['protocol_sha256']))
        tree=export(stage/(name+'.json'),tree);checks+=check_export(tree,raw_trees.get(c['index']),names,features)
        for i,pid in enumerate(ids):
            action=decide(tree,features[i])['action'];j=names.index(action)
            if j!=c['choices'][i]:raise ValueError('Export changed selected choices')
            decisions.append(dict(policy=name,prompt_id=pid,input_sha256=matrix['input_hashes'][i],task=matrix['tasks'][i],action=action,
                accuracy=float(scores[i,j]),baseline_accuracy=float(scores[i,dense]),answer_ttft=float(times[i,j]),
                dense_probe=float(probe[i] if j==dense else 0),traversal=c['traversal_seconds'],
                ttft=float(times[i,j]+(probe[i] if j==dense else 0)+c['traversal_seconds']),output_cap=matrix['output_caps'][i][j]))
        selected=[r for r in decisions if r['policy']==name]
        # Independent scalar recomputation from exported decisions, not candidate indices.
        accuracy=sum(r['accuracy'] for r in selected)/n;latency=sum(r['ttft'] for r in selected)/n;counts=Counter(r['action'] for r in selected)
        if abs(accuracy-c['accuracy'])>1e-12 or abs(latency-c['ttft'])>1e-10 or any(counts[a]!=c['actions'][a] for a in names):raise ValueError('Independent policy recomputation failed')
        summary.append(dict(policy=name,accuracy_percent=100*accuracy,loss_pp=100*(base-accuracy),ttft_seconds=latency,
            speedup=baseline_time/latency,actions=c['actions'],output_caps=sum(r['output_cap'] for r in selected),setting=c['setting']))
    for c in candidates:
        rows=[(float(matrix['scores'][i][j]),float(matrix['answer_ttft'][i][j])+(float(matrix['probe_ttft'][i]) if j==dense else 0)) for i,j in enumerate(c['choices'])]
        score=sum(r[0] for r in rows)/n;latency=sum(r[1] for r in rows)/n+c['traversal_seconds']
        if abs(score-c['accuracy'])>1e-12 or abs(latency-c['ttft'])>1e-10 or c['eligible']!=(base-score<=.02+1e-12):raise ValueError('Independent candidate check failed')
    per_task=[]
    for policy in summary:
        for task in sorted(set(matrix['tasks'])):
            rows=[r for r in decisions if r['policy']==policy['policy'] and r['task']==task];counts=Counter(r['action'] for r in rows)
            latency=sum(r['ttft'] for r in rows)/20
            base_t=sum(float(times[i,dense]) for i,t in enumerate(matrix['tasks']) if t==task)/20
            per_task.append(dict(policy=policy['policy'],task=task,samples=20,accuracy_percent=5*sum(r['accuracy'] for r in rows),
                baseline_accuracy_percent=5*sum(r['baseline_accuracy'] for r in rows),ttft_seconds=latency,speedup=base_t/latency,
                output_caps=sum(r['output_cap'] for r in rows),**{a:counts[a] for a in names}))
    for stem,rows in [('decisions',decisions),('per-task',per_task)]:
        with (stage/(stem+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    methods=[dict(action=a,accuracy_percent=float(scores[:,j].mean()*100),answer_ttft_seconds=float(times[:,j].mean()),
        output_caps=sum(row[j] for row in matrix['output_caps'])) for j,a in enumerate(names)]
    report=dict(training_samples=n,heldout_samples=0,training_only=True,searched=len(grid),feasible_candidates=sum(c['eligible'] for c in candidates),
        previous_retained=True,baseline_accuracy_percent=base*100,baseline_ttft_seconds=baseline_time,policies=summary,methods=methods,
        timing=protocol['timing'],limitations=protocol['limitations'],elapsed_seconds=time.time()-started)
    atomic_json(stage/'report.json',report)
    text='\n'.join([protocol['evaluation'],protocol['limitations'],protocol['timing'],
        f'Dense: {100*base:.4f}% / {baseline_time:.4f}s',f'Searched {len(grid)} settings; previous router retained as a candidate.',
        'Policy | accuracy % | loss pp | TTFT s | speedup | dense/1/5/10/20/50']+
        [f"{r['policy']} | {r['accuracy_percent']:.4f} | {r['loss_pp']:.4f} | {r['ttft_seconds']:.4f} | {r['speedup']:.4f}x | "+'/'.join(str(r['actions'][a]) for a in names) for r in summary])+'\n'
    (stage/'report.txt').write_text(text);(stage/'report.html').write_text('<meta charset="utf-8"><pre>'+html.escape(text)+'</pre>')
    atomic_json(stage/'independent-validation.json',dict(export_checks=checks,policy_decisions=len(decisions),candidate_recomputations=len(candidates),
        all_training_loss_bounds=True,previous_nonregression=True,matrix_sha256=matrix_sha256))
    atomic_json(stage/'complete.json',dict(complete=True,matrix_sha256=matrix_sha256,files={p.name:file_hash(p) for p in stage.iterdir() if p.is_file()}))
    stage.rename(output);print(text,flush=True)
