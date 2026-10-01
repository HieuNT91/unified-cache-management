"""ruler13-v1: pooled five-fold positive-loss/cost trees, CPU only."""
import itertools
import math
from collections import Counter
from pathlib import Path
import numpy as np
from runner import corpus_progress as progress
from runner.router_policy import FEATURES, number, digest
from runner.tree_policy import actions, SCHEMA, FEATURE_DEFINITIONS, export, decide
from runner.setups import atomic_json, file_hash
from scripts.corpus_inputs import TASKS

LOSS_GRID=[dict(family='loss',depth=d,min_leaf=l,shrinkage=s,threshold=t)
    for d,l,s,t in itertools.product((1,2,3),(10,20),(0,5,20),(0.,.02,.05,.10,.20))]
COST_GRID=[dict(family='cost',depth=d,min_leaf=l,lam=w,penalty=p)
    for d,l,w,p in itertools.product((1,2,3),(10,20),(1,2,5,10,20,50,100),(0.,.005,.02))]
GRID=LOSS_GRID+COST_GRID


def grid_values(name,values,default,integer=False,upper=None):
    values=tuple(default if values is None else values)
    if not values or any((type(v) is not int or v<1) if integer else
            (not number(v) or not math.isfinite(v) or v<0) for v in values):
        raise ValueError(name+' requires '+('positive integers' if integer else 'finite nonnegative numbers'))
    if upper is not None and any(v>upper for v in values):raise ValueError(name+' exceeds '+str(upper))
    return sorted(set(values))


def search_grid(depths=None,accuracy_weight=None,selection_objective='legacy',max_accuracy_loss_pp=2.,
                min_leaf=None,lambdas=None,loss_thresholds=None,shrinkages=None,leaf_penalties=None,min_speedup=None):
    if selection_objective not in ('legacy','min-budget','min-ttft'):raise ValueError('Unknown selection objective')
    if not number(max_accuracy_loss_pp) or not 0<=max_accuracy_loss_pp<=100:
        raise ValueError('max-accuracy-loss-pp must be between 0 and 100')
    if min_speedup is not None:
        grid_values('min-speedup',[min_speedup],())
        if selection_objective!='legacy':raise ValueError('--min-speedup applies only to legacy selection; min-ttft/min-budget have no speedup gate')
    depths=grid_values('depths',depths,(1,2,3),integer=True,upper=32)
    leaves=grid_values('min-leaf',min_leaf,(10,20),integer=True)
    penalties=grid_values('leaf-penalties',leaf_penalties,(0.,.005,.02))
    if accuracy_weight is not None:
        grid_values('accuracy-weight',[accuracy_weight],())
        if lambdas is not None:raise ValueError('Use --lambdas or --accuracy-weight, not both')
    cost_only=selection_objective=='min-budget' or accuracy_weight is not None
    if cost_only and (loss_thresholds is not None or shrinkages is not None):
        raise ValueError('Loss thresholds/shrinkages require loss trees; omit them with min-budget or --accuracy-weight')
    default_weights=(0.,.1,.25,.5,1.,2.,5.,10.,20.,50.,100.) if selection_objective=='min-budget' else (1,2,5,10,20,50,100)
    weights=grid_values('lambdas',lambdas,default_weights) if accuracy_weight is None else [float(accuracy_weight)]
    costs=[dict(family='cost',depth=d,min_leaf=l,lam=w,penalty=p)
           for d,l,w,p in itertools.product(depths,leaves,weights,penalties)]
    if selection_objective=='min-budget':
        return [dict(h,cost_basis='budget') for h in costs]
    if cost_only:return costs
    thresholds=grid_values('loss-thresholds',loss_thresholds,(0.,.02,.05,.10,.20),upper=1.)
    shrink=grid_values('shrinkages',shrinkages,(0,5,20))
    losses=[dict(family='loss',depth=d,min_leaf=l,shrinkage=r,threshold=t)
            for d,l,r,t in itertools.product(depths,leaves,shrink,thresholds)]
    return losses+costs


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


def fit(x,y,times,overhead,h,dense,sparse,budgets=None):
    totals=times+overhead[:,None];costs=totals.mean(0);normalizer=float(np.median(times[:,dense]))
    labels=None
    if h['family']=='cost':
        loss=np.zeros_like(times);loss[:,sparse]=y
        base=np.asarray(budgets)[None,:] if h.get('cost_basis')=='budget' else totals/normalizer
        labels=base+h['lam']*loss
    return fit_tree(x,y,h,labels),costs,normalizer


def macro(values,samples):
    return float(np.mean([np.mean([v for v,r in zip(values,samples) if r['task']==t]) for t in sorted({r['task'] for r in samples})]))


def rank(table,count=3,accuracy_weight=None,selection_objective='legacy'):
    good=sorted((r for r in table if r['feasible']),key=lambda r:(r['mean_total_ttft'],-r['macro_score'],r['depth'],r['leaf_count'],r['index']))
    bad=sorted((r for r in table if not r['feasible']),key=lambda r:(-r['macro_score'],r['mean_total_ttft'],r['depth'],r['leaf_count'],r['index']))
    ordered=good+bad
    if accuracy_weight is not None and selection_objective=='legacy':
        ordered=sorted(table,key=lambda r:(r['weighted_objective'],r['mean_total_ttft'],
            -r['macro_score'],r['depth'],r['leaf_count'],r['index']))
    if selection_objective!='legacy':
        eligible=[r for r in table if r['feasible']]
        if not eligible:raise ValueError('No candidate meets the OOF accuracy-loss limit; increase --max-accuracy-loss-pp or broaden --depths/weight search')
        key='mean_selected_budget' if selection_objective=='min-budget' else 'mean_total_ttft'
        ordered=sorted(eligible,key=lambda r:(r[key],r['mean_total_ttft'],-r['macro_score'],r['depth'],r['leaf_count'],r['index']))
        count=min(count,len(ordered))
    chosen=[];vectors=set()
    for r in ordered:
        vector=tuple(r['oof_actions'])
        if vector not in vectors:chosen.append(r);vectors.add(vector)
        if len(chosen)==count:return chosen
    for r in ordered:
        if r['index'] not in {p['index'] for p in chosen}:chosen.append(r)
        if len(chosen)==count:return chosen
    raise ValueError('Requested more policies than settings')


def search(samples,features,scores,times,overhead,inventory,seed=42,count=3,
           depths=None,accuracy_weight=None,selection_objective='legacy',max_accuracy_loss_pp=2.,
           min_leaf=None,lambdas=None,loss_thresholds=None,shrinkages=None,leaf_penalties=None,min_speedup=None):
    grid=search_grid(depths,accuracy_weight,selection_objective,max_accuracy_loss_pp,
                     min_leaf,lambdas,loss_thresholds,shrinkages,leaf_penalties,min_speedup)
    definitions=actions(inventory);names=list(definitions);dense=names.index('nocache');sparse=[i for i in range(len(names)) if i!=dense]
    budgets=np.array([1. if name=='nocache' else definitions[name]['ratio'] for name in names])
    scores=np.asarray(scores,float);times=np.asarray(times,float);overhead=np.asarray(overhead,float);n=len(samples)
    if (scores.shape!=times.shape or scores.shape!=(n,len(names)) or overhead.shape!=(n,) or len(features)!=n or
            not np.isfinite(scores).all() or (scores<0).any() or (scores>1).any() or not np.isfinite(times).all() or
            (times<=0).any() or not np.isfinite(overhead).all() or (overhead<0).any()):raise ValueError('Invalid training outcome matrix')
    missing=np.array([any(not number(row.get(f)) for f in FEATURES) for row in features])
    x=np.array([[row[f] if number(row.get(f)) else 0. for f in FEATURES] for row in features])
    y=np.maximum(0,scores[:,[dense]]-scores[:,sparse]);folds=fold_ids(samples,seed)
    baseline=macro(scores[:,dense],samples);table=[];cache={}
    for index,h in progress.track(list(enumerate(grid)), 'Searching trees', 'settings', lambda item: f'setting {item[0]+1}/{len(grid)} ({item[1]["family"]})'):
        decisions=np.empty(n,int);statistics=[];leaves=0;normalizers=np.empty(n,float)
        for fold in range(5):
            progress.detail(f'setting {index+1}/{len(grid)} ({h["family"]}); fold {fold+1}/5')
            tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
            key=(tuple((k,v) for k,v in h.items() if k!='threshold'),fold)
            if key not in cache:cache[key]=fit(x[tr],y[tr],times[tr],overhead[tr],h,dense,sparse,budgets)
            tree,costs,norm=cache[key];leaves+=leaf_count(tree);normalizers[va]=norm
            statistics.append(dict(fold=fold,train_ids=[samples[i]['id'] for i in tr],validation_ids=[samples[i]['id'] for i in va],
                                   costs=costs.tolist(),dense_median=norm,shrinkage_prior=y[tr].mean(0).tolist()))
            for i in va:decisions[i]=dense if missing[i] else choose(leaf(tree,x[i]),costs,h,dense,sparse)
        score=macro(scores[np.arange(n),decisions],samples)
        elapsed=times[np.arange(n),decisions]+overhead
        speed=float(times[:,dense].mean()/elapsed.mean())
        table.append(dict(h,index=index,macro_score=score,macro_loss=baseline-score,mean_total_ttft=float(elapsed.mean()),
            speedup=speed,leaf_count=leaves,
            feasible=bool(baseline-score<=max_accuracy_loss_pp/100+1e-12 and (selection_objective!='legacy' or speed>=(4. if min_speedup is None else min_speedup)-1e-12)),
            mean_selected_budget=float(budgets[decisions].mean()),oof_action_counts=dict(Counter(names[i] for i in decisions)),
            oof_actions=decisions.tolist(),fold_statistics=statistics))
        if accuracy_weight is not None and selection_objective=='legacy':
            positive_loss=np.maximum(0,scores[:,dense]-scores[np.arange(n),decisions])
            table[-1]['weighted_objective']=macro(elapsed/normalizers+accuracy_weight*positive_loss,samples)
    selected=rank(table,count,accuracy_weight,selection_objective);policies=[]
    for index,winner in progress.track(list(enumerate(selected)), 'Fitting selected trees', 'policies', lambda item: f'router{item[0]+1}'):
        h=grid[winner['index']];raw,costs,norm=fit(x,y,times,overhead,h,dense,sparse,budgets)
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
          action_scope='all',evaluation='heldout',depths=None,accuracy_weight=None,
          selection_objective='legacy',max_accuracy_loss_pp=2.,
          min_leaf=None,lambdas=None,loss_thresholds=None,shrinkages=None,leaf_penalties=None,min_speedup=None):
    from runner.corpus_training_data import training_view
    corpus=training_view(corpus,action_scope)
    if corpus.protocol.get('dataset')!='ruler' or corpus.protocol.get('kind')!='collection':raise ValueError('Only RULER corpus collections may train routers')
    if trainer!='ruler13-v1':raise ValueError('Unknown trainer version')
    grid_options=dict(min_leaf=min_leaf,lambdas=lambdas,loss_thresholds=loss_thresholds,
                      shrinkages=shrinkages,leaf_penalties=leaf_penalties,min_speedup=min_speedup)
    grid=search_grid(depths,accuracy_weight,selection_objective,max_accuracy_loss_pp,**grid_options)
    if type(policy_count) is not int or not 1<=policy_count<=len(grid):raise ValueError('Invalid policy count')
    if evaluation not in ('heldout','training'):raise ValueError('Unknown evaluation mode')
    unchecked=getattr(corpus,'skip_validation',False)
    with progress.stage('Reading saved sample inventory' if unchecked else 'Creating validated snapshot'):
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
    for row in progress.track(samples, 'Extracting features and outcomes', 'samples', lambda row: row['id']):
        pid=row['id'];features.append(corpus.features(pid))
        outcomes=[corpus.outcome(pid,a) for a in corpus.actions]
        if any(o is None for o in outcomes):raise ValueError('Frozen training sample lost an accepted outcome')
        scores.append([o['accuracy'] for o in outcomes]);times.append([o['timings']['answer_engine_ttft_seconds'] for o in outcomes])
        overhead.append(corpus.probe(pid)['timings']['routing_overhead_seconds'])
    progress.detail('writing feature/outcome matrices')
    atomic_json(output/'matrices.json',dict(ids=membership['train_ids'],features=features,scores=scores,answer_ttft=times,probe_overhead=overhead))
    policies,table,folds=search(samples,features,scores,times,overhead,corpus.protocol['actions'],seed,policy_count,depths,accuracy_weight,selection_objective,max_accuracy_loss_pp,**grid_options)
    objective=dict(mode='legacy' if accuracy_weight is None else 'weighted',accuracy_weight=accuracy_weight,
        depths=sorted({h['depth'] for h in grid}),
        formula='task-macro mean(total_ttft / fitting_fold_dense_median + accuracy_weight * max(0, dense_score - action_score))'
            if accuracy_weight is not None else 'feasibility first; TTFT then accuracy if feasible, accuracy then TTFT otherwise')
    speedup_target=(4. if min_speedup is None else min_speedup) if selection_objective=='legacy' else None
    objective.update(selection_objective=selection_objective,max_accuracy_loss_pp=max_accuracy_loss_pp,
        min_speedup=speedup_target,search_space={key:sorted({h[key] for h in grid if key in h})
            for key in ('depth','min_leaf','lam','threshold','shrinkage','penalty')})
    if selection_objective!='legacy':
        objective.update(mode=selection_objective,
            formula=('mean selected budget (nocache=1); tie: estimated TTFT, accuracy' if selection_objective=='min-budget'
                else 'mean estimated TTFT; tie: accuracy')+'; subject to OOF macro accuracy-loss limit',
            fitting_cost='budget + lambda * positive_loss' if selection_objective=='min-budget' else 'normalized TTFT + lambda * positive_loss')
    settings=dict(trainer=trainer,seed=seed,policy_count=policy_count,primary='router1',grid=grid,folds=folds,objective=objective,
        action_scope=action_scope,evaluation=evaluation,evaluation_ids=membership['evaluation_ids'],
        train_ids=membership['train_ids'],targets=dict(macro_loss=max_accuracy_loss_pp/100,speedup=speedup_target),actions=corpus.protocol['actions'],
        training_hardware=corpus.protocol.get('hardware',{}),snapshot_sha256=file_hash(output/'snapshot.json'))
    if unchecked:settings['dataset_validation']='skipped'
    with progress.stage('Writing search results'):
        atomic_json(output/'settings.json',settings);atomic_json(output/'search.json',table)
    trees=[]
    for policy in progress.track(policies, 'Exporting trees', 'policies', lambda policy: policy['id']):
        raw=policy.pop('raw_tree');atomic_json(output/(policy['id']+'-fit.json'),raw)
        tree=dict(policy,schema=SCHEMA,max_depth=policy['setting']['depth'],objective=objective,feature_names=list(FEATURES),feature_definitions=FEATURE_DEFINITIONS,
            actions=corpus.protocol['actions'],training_ids=membership['train_ids'],heldout_ids=membership['heldout_ids'],
            heldout_hashes={k:v for k,v in snapshot['acceptance_hashes'].items() if k.split('/')[2] in membership['heldout_ids']},
            provenance=dict(snapshot_sha256=file_hash(output/'snapshot.json'),training_run_sha256=file_hash(output/'settings.json'),
                corpus_protocol_sha256=snapshot['protocol_sha256'],plan_sha256=corpus.protocol['plan_sha256'],
                hardware=corpus.protocol.get('hardware',{})))
        tree['evaluation']=evaluation
        if unchecked:tree['dataset_validation']='skipped'
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
    reports={}
    for tree in progress.track(trees, 'Evaluating routers offline', 'policies', lambda tree: tree['id']):
        reports[tree['id']]=test_tree(corpus,tree,output/(tree['id']+'-'+evaluation),evaluation=evaluation)
    summary=dict(training_samples=n,heldout_samples=len(membership['heldout_ids']),primary='router1',
        action_scope=action_scope,evaluation=evaluation,evaluation_samples=len(membership['evaluation_ids']),
        training_overlap=evaluation=='training',
        interpretation='Training-set resubstitution; no held-out performance claim' if evaluation=='training' else 'Held-out evaluation',
        trainer=trainer,actions=corpus.protocol['actions'],settings_searched=len(grid),objective=objective,
        timing='Estimated probe-inclusive OOF TTFT; not live router latency',
        requested_policy_count=policy_count,exported_policy_count=len(trees),
        policies=[dict(id=t['id'],oof_mean_budget_percent=100*t['oof']['mean_selected_budget'],
            oof_action_counts=t['oof']['oof_action_counts'],feasible=t['oof']['feasible'],oof_macro_loss=t['oof']['macro_loss'],
            oof_speedup=t['oof']['speedup'],leaves=_compiled_leaves(t['tree']),duplicate_of=t['duplicate_of']) for t in trees])
    budget_by_action={a['id']:1. if a['id']=='nocache' else a['ratio'] for a in corpus.protocol['actions']}
    for policy in summary['policies']:
        report=reports[policy['id']];overall=report['overall']
        policy.update(accuracy_percent=100*report['macro_accuracy'],
            baseline_accuracy_percent=100*report['macro_baseline_accuracy'],
            accuracy_loss_pp=100*(report['macro_baseline_accuracy']-report['macro_accuracy']),
            answer_ttft_seconds=overall['answer_ttft_seconds'],
            estimated_router_ttft_seconds=overall['estimated_router_ttft_seconds'],
            baseline_ttft_seconds=overall['baseline_ttft_seconds'],
            estimated_speedup=overall['estimated_speedup'],actions=overall['actions'],
            mean_selected_budget_percent=100*sum(budget_by_action[a]*c for a,c in overall['actions'].items())/sum(overall['actions'].values()))
    if unchecked:summary['dataset_validation']='skipped'
    if snapshot.get('source') in ('completed-add5-10-extension','add5-10-saved-records','completed-multi-extension','multi-extension-saved-records'):
        summary['data_source']=snapshot['source']
        summary['timing']+='; original and added fixed actions were measured in different sessions'
    progress.detail('writing summary and completion receipt: '+str(output))
    atomic_json(output/'summary.json',summary)
    import json
    (output/'summary.txt').write_text(json.dumps(summary,indent=2)+'\n')
    atomic_json(output/'complete.json',dict(complete=True,primary='router1',training=n,heldout=len(membership['heldout_ids']),
        evaluation=evaluation,evaluated=len(membership['evaluation_ids']),training_overlap=evaluation=='training',
        settings_searched=len(grid),objective=objective,policies=len(trees),requested_policies=policy_count,estimated_timing=True,
        **({'dataset_validation':'skipped'} if unchecked else {}),
        files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file()}))
    return summary
