"""ruler13-v1: pooled five-fold positive-loss/cost trees, CPU only."""
import itertools
from collections import Counter
from pathlib import Path
import numpy as np
from runner.router_policy import FEATURES, number, digest
from runner.tree_policy import actions, SCHEMA, FEATURE_DEFINITIONS, export, decide
from runner.setups import atomic_json, file_hash
from scripts.corpus_inputs import TASKS

LOSS_GRID=[dict(family='loss',depth=d,min_leaf=l,shrinkage=s,threshold=t)
    for d,l,s,t in itertools.product((1,2,3),(10,20),(0,5,20),(0.,.02,.05,.10,.20))]
COST_GRID=[dict(family='cost',depth=d,min_leaf=l,lam=w,penalty=p)
    for d,l,w,p in itertools.product((1,2,3),(10,20),(1,2,5,10,20,50,100),(0.,.005,.02))]
GRID=LOSS_GRID+COST_GRID


def split(samples,n,seed):
    if len({r['id'] for r in samples})!=len(samples):raise ValueError('Duplicate snapshot sample')
    groups={t:[] for t in TASKS}
    for row in samples:
        if row['task'] not in groups:raise ValueError('Unknown task')
        groups[row['task']].append(row['id'])
    if any(len(g)<2 for g in groups.values()):raise ValueError('Train/test requires at least two complete samples for all 13 tasks')
    if type(n) is not int or not 13<=n<=len(samples)-13:raise ValueError('train-samples must be between 13 and complete_samples - 13')
    capacity={t:len(g)-2 for t,g in groups.items()};remaining=n-13;total=sum(capacity.values())
    # Integer arithmetic avoids floating-point largest-remainder ties.
    quotas={t:1+(remaining*capacity[t]//total if total else 0) for t in TASKS}
    order=sorted(TASKS,key=lambda t:(-(remaining*capacity[t]%total if total else 0),t))
    for t in order[:n-sum(quotas.values())]:quotas[t]+=1
    train=[];test=[]
    for t in TASKS:
        ordered=sorted(groups[t],key=lambda pid:(digest([seed,t,pid]),pid))
        train.extend(ordered[:quotas[t]]);test.extend(ordered[quotas[t]:])
    return dict(train_ids=train,heldout_ids=test,allocation=quotas,seed=seed,requested_train_samples=n)


def fold_ids(samples,seed):
    if set(r['task'] for r in samples)!=set(TASKS):raise ValueError('Pooled training must represent all 13 tasks')
    result=np.empty(len(samples),int);offset=0
    for task in TASKS:
        indices=sorted((i for i,r in enumerate(samples) if r['task']==task),key=lambda i:digest([seed,'fold',samples[i]['id']]))
        for j,i in enumerate(indices):result[i]=(offset+j)%5
        offset=(offset+len(indices))%5
    if set(result)!=set(range(5)):raise ValueError('Five nonempty prompt folds required')
    return result


def fit_tree(x,y,setting,cost_labels=None):
    """Ordered CART splits; shrink toward this fitting fold's pooled loss mean."""
    prior=y.mean(0);total=len(x);minimum=setting['min_leaf'];shrink=setting.get('shrinkage',0)
    def build(indices,depth):
        values=y[indices]
        node=dict(n=len(indices),loss=((values.sum(0)+shrink*prior)/(len(indices)+shrink)).tolist())
        if cost_labels is not None:node['action']=int(np.argmin(cost_labels[indices].mean(0)))
        if depth==setting['depth'] or len(indices)<2*minimum:return node
        best=None;best_order=None
        for feature in range(len(FEATURES)):
            order=indices[np.argsort(x[indices,feature],kind='stable')];v=x[order,feature]
            z=y[order] if cost_labels is None else cost_labels[order]
            sums=z.cumsum(0);squares=(z*z).cumsum(0)
            for k in range(minimum,len(order)-minimum+1):
                if v[k-1]==v[k]:continue
                if cost_labels is None:
                    error=float((squares[k-1]-sums[k-1]**2/k).sum()+
                        (squares[-1]-squares[k-1]-(sums[-1]-sums[k-1])**2/(len(order)-k)).sum())
                else:error=float((sums[k-1].min()+(sums[-1]-sums[k-1]).min())/total+setting['penalty'])
                threshold=float(v[k-1]+(v[k]-v[k-1])/2)
                if threshold>=v[k]:threshold=float(v[k-1])
                candidate=(error,feature,threshold,k)
                if best is None or candidate[:3]<best[:3]:best=candidate;best_order=order
        base=float(((values-values.mean(0))**2).sum()) if cost_labels is None else float(cost_labels[indices].sum(0).min()/total)
        if best is not None and best[0]<base-1e-12:
            _,feature,threshold,k=best
            node.update(feature=feature,threshold=threshold,left=build(best_order[:k],depth+1),right=build(best_order[k:],depth+1))
        return node
    return build(np.arange(total),0)


def leaf(tree,x):
    while 'feature' in tree:tree=tree['left'] if x[tree['feature']]<=tree['threshold'] else tree['right']
    return tree


def leaf_count(tree):
    return 1 if 'feature' not in tree else leaf_count(tree['left'])+leaf_count(tree['right'])


def choose(node,costs,setting,dense,sparse):
    if setting['family']=='cost':return node['action']
    eligible=[index for index,loss in zip(sparse,node['loss']) if loss<=setting['threshold']]+[dense]
    return min(eligible,key=lambda i:(costs[i],i))


def fit(x,y,times,overhead,h,dense,sparse):
    totals=times+overhead[:,None];costs=totals.mean(0);normalizer=float(np.median(times[:,dense]))
    labels=None
    if h['family']=='cost':
        loss=np.zeros_like(times);loss[:,sparse]=y
        labels=totals/normalizer+h['lam']*loss
    return fit_tree(x,y,h,labels),costs,normalizer


def macro(values,samples):
    return float(np.mean([np.mean([v for v,r in zip(values,samples) if r['task']==t]) for t in sorted({r['task'] for r in samples})]))


def rank(table,count=3):
    good=sorted((r for r in table if r['feasible']),key=lambda r:(r['mean_total_ttft'],-r['macro_score'],r['depth'],r['leaf_count'],r['index']))
    bad=sorted((r for r in table if not r['feasible']),key=lambda r:(-r['macro_score'],r['mean_total_ttft'],r['depth'],r['leaf_count'],r['index']))
    chosen=[];vectors=set()
    for r in good+bad:
        vector=tuple(r['oof_actions'])
        if vector not in vectors:chosen.append(r);vectors.add(vector)
        if len(chosen)==count:return chosen
    for r in good+bad:
        if r['index'] not in {p['index'] for p in chosen}:chosen.append(r)
        if len(chosen)==count:return chosen
    raise ValueError('Requested more policies than settings')


def search(samples,features,scores,times,overhead,inventory,seed=42,count=3):
    definitions=actions(inventory);names=list(definitions);dense=names.index('nocache');sparse=[i for i in range(len(names)) if i!=dense]
    scores=np.asarray(scores,float);times=np.asarray(times,float);overhead=np.asarray(overhead,float);n=len(samples)
    if (scores.shape!=times.shape or scores.shape!=(n,len(names)) or overhead.shape!=(n,) or len(features)!=n or
            not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any() or not np.isfinite(times).all() or
            (times<=0).any() or not np.isfinite(overhead).all() or (overhead<0).any()):raise ValueError('Invalid training outcome matrix')
    missing=np.array([any(not number(row.get(f)) for f in FEATURES) for row in features])
    x=np.array([[row[f] if number(row.get(f)) else 0. for f in FEATURES] for row in features])
    y=np.maximum(0,scores[:,[dense]]-scores[:,sparse]);folds=fold_ids(samples,seed)
    baseline=macro(scores[:,dense],samples);table=[];cache={}
    for index,h in enumerate(GRID):
        decisions=np.empty(n,int);statistics=[];leaves=0
        for fold in range(5):
            tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
            key=(tuple((k,v) for k,v in h.items() if k!='threshold'),fold)
            if key not in cache:cache[key]=fit(x[tr],y[tr],times[tr],overhead[tr],h,dense,sparse)
            tree,costs,norm=cache[key];leaves+=leaf_count(tree)
            statistics.append(dict(fold=fold,train_ids=[samples[i]['id'] for i in tr],validation_ids=[samples[i]['id'] for i in va],
                                   costs=costs.tolist(),dense_median=norm,shrinkage_prior=y[tr].mean(0).tolist()))
            for i in va:decisions[i]=dense if missing[i] else choose(leaf(tree,x[i]),costs,h,dense,sparse)
        score=macro(scores[np.arange(n),decisions],samples)
        elapsed=times[np.arange(n),decisions]+overhead
        speed=float(times[:,dense].mean()/elapsed.mean())
        table.append(dict(h,index=index,macro_score=score,macro_loss=baseline-score,mean_total_ttft=float(elapsed.mean()),
            speedup=speed,leaf_count=leaves,feasible=bool(baseline-score<=.02+1e-12 and speed>=4.-1e-12),
            oof_actions=decisions.tolist(),fold_statistics=statistics))
    selected=rank(table,count);policies=[]
    for index,winner in enumerate(selected):
        h=GRID[winner['index']];raw,costs,norm=fit(x,y,times,overhead,h,dense,sparse)
        def compile_node(node):
            if 'feature' not in node:return dict(action=names[choose(node,costs,h,dense,sparse)],training_samples=node['n'])
            return dict(feature=FEATURES[node['feature']],threshold=node['threshold'],le=compile_node(node['left']),gt=compile_node(node['right']))
        policies.append(dict(id=f'router{index+1}',primary=index==0,tree=compile_node(raw),training_costs=costs.tolist(),
            trainer='ruler13-v1',setting=h,dense_median=norm,oof={k:v for k,v in winner.items() if k!='fold_statistics'},
            duplicate_of=[f'router{j+1}' for j,r in enumerate(selected[:index]) if r['oof_actions']==winner['oof_actions']],
            raw_tree=raw))
    return policies,table,folds.tolist()


def _compiled_leaves(tree):
    return 1 if 'action' in tree else _compiled_leaves(tree['le'])+_compiled_leaves(tree['gt'])


def train(corpus,n,output,seed=42,trainer='ruler13-v1',policy_count=3,
          action_scope='all',evaluation='heldout'):
    if corpus.protocol.get('feature_profile'):
        raise ValueError('Coverage-five data requires the separate portable-dataset trainer')
    from runner.corpus_training_data import training_view
    corpus=training_view(corpus,action_scope)
    if corpus.protocol.get('dataset')!='ruler' or corpus.protocol.get('kind')!='collection':raise ValueError('Only RULER corpus collections may train routers')
    if trainer!='ruler13-v1':raise ValueError('Unknown trainer version')
    if type(policy_count) is not int or not 1<=policy_count<=len(GRID):raise ValueError('Invalid policy count')
    if evaluation not in ('heldout','training'):raise ValueError('Unknown evaluation mode')
    snapshot=corpus.snapshot()
    if evaluation=='training':
        total=len(snapshot['samples'])
        if n is not None and n!=total:
            raise ValueError('Training-set evaluation uses all complete samples; omit --train-samples or give their exact count')
        n=total
        ids=[r['id'] for r in snapshot['samples']]
        if not ids or len(set(ids))!=len(ids):raise ValueError('Empty or duplicate training cohort')
        membership=dict(train_ids=ids,heldout_ids=[],evaluation_ids=list(ids),evaluation=evaluation,
            training_overlap=True,seed=seed,requested_train_samples=n,
            allocation=dict(Counter(r['task'] for r in snapshot['samples'])))
    else:
        membership=split(snapshot['samples'],n,seed)
        membership.update(evaluation_ids=membership['heldout_ids'],evaluation=evaluation,training_overlap=False)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'snapshot.json',snapshot);atomic_json(output/'split.json',membership)
    by_id={r['id']:r for r in snapshot['samples']};samples=[by_id[i] for i in membership['train_ids']]
    features=[];scores=[];times=[];overhead=[]
    for row in samples:
        pid=row['id'];features.append(corpus.features(pid))
        outcomes=[corpus.outcome(pid,a) for a in corpus.actions]
        if any(o is None for o in outcomes):raise ValueError('Frozen training sample lost an accepted outcome')
        scores.append([o['accuracy'] for o in outcomes]);times.append([o['timings']['answer_engine_ttft_seconds'] for o in outcomes])
        overhead.append(corpus.probe(pid)['timings']['routing_overhead_seconds'])
    atomic_json(output/'matrices.json',dict(ids=membership['train_ids'],features=features,scores=scores,answer_ttft=times,probe_overhead=overhead))
    policies,table,folds=search(samples,features,scores,times,overhead,corpus.protocol['actions'],seed,policy_count)
    settings=dict(trainer=trainer,seed=seed,policy_count=policy_count,primary='router1',grid=GRID,folds=folds,
        action_scope=action_scope,evaluation=evaluation,evaluation_ids=membership['evaluation_ids'],
        train_ids=membership['train_ids'],targets=dict(macro_loss=.02,speedup=4),actions=corpus.protocol['actions'],
        training_hardware=corpus.protocol.get('hardware',{}),snapshot_sha256=file_hash(output/'snapshot.json'))
    atomic_json(output/'settings.json',settings);atomic_json(output/'search.json',table)
    trees=[]
    for policy in policies:
        raw=policy.pop('raw_tree');atomic_json(output/(policy['id']+'-fit.json'),raw)
        tree=dict(policy,schema=SCHEMA,feature_names=list(FEATURES),feature_definitions=FEATURE_DEFINITIONS,
            actions=corpus.protocol['actions'],training_ids=membership['train_ids'],heldout_ids=membership['heldout_ids'],
            heldout_hashes={k:v for k,v in snapshot['acceptance_hashes'].items() if k.split('/')[2] in membership['heldout_ids']},
            provenance=dict(snapshot_sha256=file_hash(output/'snapshot.json'),training_run_sha256=file_hash(output/'settings.json'),
                corpus_protocol_sha256=snapshot['protocol_sha256'],plan_sha256=corpus.protocol['plan_sha256'],
                hardware=corpus.protocol.get('hardware',{})))
        tree['evaluation']=evaluation
        if evaluation=='training':
            tree['training_hashes']=dict(snapshot['acceptance_hashes'])
        tree=export(output/(policy['id']+'.json'),tree);trees.append(tree)
        # Check compiler equivalence at training values and exact split boundaries.
        xrows=[{f:row.get(f) for f in FEATURES} for row in features]
        def boundaries(node,values):
            if 'feature' not in node:return
            name=FEATURES[node['feature']];t=node['threshold']
            for v in (np.nextafter(t,-np.inf),t,np.nextafter(t,np.inf)):xrows.append(dict(values,**{name:float(v)}))
            boundaries(node['left'],dict(values,**{name:t}));boundaries(node['right'],dict(values,**{name:float(np.nextafter(t,np.inf))}))
        boundaries(raw,dict.fromkeys(FEATURES,.5))
        names=list(corpus.actions);dense=names.index('nocache');sparse=[i for i in range(len(names)) if i!=dense]
        for row in xrows+[{}]:
            missing=any(not number(row.get(f)) for f in FEATURES)
            expected='nocache' if missing else names[choose(leaf(raw,[row[f] for f in FEATURES]),policy['training_costs'],policy['setting'],dense,sparse)]
            if decide(tree,row)['action']!=expected:raise ValueError('Export decision mismatch')
    from runner.corpus_eval import test_tree
    reports={tree['id']:test_tree(corpus,tree,output/(tree['id']+'-'+evaluation),evaluation=evaluation)
             for tree in trees}
    summary=dict(training_samples=n,heldout_samples=len(membership['heldout_ids']),primary='router1',
        action_scope=action_scope,evaluation=evaluation,evaluation_samples=len(membership['evaluation_ids']),
        training_overlap=evaluation=='training',
        interpretation='Training-set resubstitution; no held-out performance claim' if evaluation=='training' else 'Held-out evaluation',
        trainer=trainer,actions=corpus.protocol['actions'],settings_searched=len(GRID),
        timing='Estimated probe-inclusive OOF TTFT; not live router latency',
        policies=[dict(id=t['id'],feasible=t['oof']['feasible'],oof_macro_loss=t['oof']['macro_loss'],
            oof_speedup=t['oof']['speedup'],leaves=_compiled_leaves(t['tree']),duplicate_of=t['duplicate_of']) for t in trees])
    for policy in summary['policies']:
        report=reports[policy['id']];overall=report['overall']
        policy.update(accuracy_percent=100*report['macro_accuracy'],
            baseline_accuracy_percent=100*report['macro_baseline_accuracy'],
            accuracy_loss_pp=100*(report['macro_baseline_accuracy']-report['macro_accuracy']),
            answer_ttft_seconds=overall['answer_ttft_seconds'],
            estimated_router_ttft_seconds=overall['estimated_router_ttft_seconds'],
            baseline_ttft_seconds=overall['baseline_ttft_seconds'],
            estimated_speedup=overall['estimated_speedup'],actions=overall['actions'])
    if snapshot.get('source')=='completed-add5-10-extension':
        summary['data_source']=snapshot['source']
        summary['timing']+='; original and added fixed actions were measured in different sessions'
    atomic_json(output/'summary.json',summary)
    import json
    (output/'summary.txt').write_text(json.dumps(summary,indent=2)+'\n')
    atomic_json(output/'complete.json',dict(complete=True,primary='router1',training=n,heldout=len(membership['heldout_ids']),
        evaluation=evaluation,evaluated=len(membership['evaluation_ids']),training_overlap=evaluation=='training',
        settings_searched=len(GRID),policies=policy_count,estimated_timing=True,
        files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}))
    return summary
