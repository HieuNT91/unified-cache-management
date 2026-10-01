"""Outcome-cost CART search over 76 singles, 28 pairs and six triples."""
import itertools
import time
from pathlib import Path
import numpy as np
from runner.gpu_study_features import ADDITIONS,CPU_FEATURES,GPU_FEATURES,FEATURES,dependencies,additional_features
from runner.attention_study import fit_ordered,compile_tree,metrics,rules,csv_write
from runner.corpus_train import leaf,leaf_count
from runner.extension_train import GRID,validate_matrix
from runner.gpu_study_policy import make,decide,check_boundaries
from runner.setups import file_hash,atomic_json


def used_features(tree):
    if 'action' in tree:return set()
    return {tree['feature']}|used_features(tree['le'])|used_features(tree['gt'])


def rank(c):
    return (c['cost_aware_ttft'],-c['accuracy'],len(c['extras']),c['leaves'],c['setting']['depth'],tuple(ADDITIONS.index(f) for f in c['extras']),c['index'])


def cost_estimate(required,costs):
    """Measured shared dependencies counted once; conservative GPU full-stat bound.

    GPU collection times both head/query branches together. For any requested GPU
    attention feature, charge that entire measured shared pass/reduction. Live
    execution omits unused branches. This is explicitly an upper-cost estimate,
    rather than silently amortizing or extrapolating unmeasured branch costs.
    """
    need=dependencies(required);value=costs['native_transfer_rpc']+costs['aggregation']
    if set(required)&set(FEATURES):value+=costs['base_cpu']
    cpu=tuple(f for f in CPU_FEATURES if f in required)
    if cpu:value+=costs['cpu_subsets'][','.join(cpu)]
    if need['head'] or need['query']:value+=costs['attention_statistics']+costs['statistics_reduce_transfer_sync']+costs['gpu_scalar_reduction']
    if need['confidence']:value+=costs['confidence']
    return value


def benchmark_subsets(output,corpus,ids,subsets,sealed):
    dest=output/'costs'/('cpu-'+str(len(subsets))+'-'+__import__('hashlib').sha256(repr(subsets).encode()).hexdigest()[:12]+'.json')
    if dest.exists():return sealed(dest)
    # 2 fixed prompt ordinals/task, chosen independently of outcomes.
    timings={','.join(fs):[] for fs in subsets}
    for pid in ids:
        a=corpus.base.attention(pid);s=corpus.inputs(pid);layers=a['layers'].astype(np.float64).mean(0)
        for fs in subsets:
            durations=[]
            for _ in range(3):
                start=time.perf_counter();additional_features(layers,a['scores'],s['boundaries'][1],s['boundaries'][-2],fs);durations.append(time.perf_counter()-start)
            timings[','.join(fs)].append(float(np.median(durations)))
    return sealed(dest,{k:float(np.mean(v)) for k,v in timings.items()})


def search(output,corpus,matrix,old,sealed,progress):
    names,_,scores,times,probe=validate_matrix(matrix)
    data=[sealed(output/'features'/f'{p}.json') for p in matrix['ids']];rows=[r['features'] for r in data]
    records=[sealed(output/'records/diagnostic'/p/'record.json') for p in matrix['ids']]
    def avg(fn):return float(np.mean([fn(r) for r in records]))
    costs=dict(native_transfer_rpc=avg(lambda r:r['native_transfer_rpc_seconds']),aggregation=avg(lambda r:r['timings']['aggregation_seconds']),
        base_cpu=avg(lambda r:r['timings']['base_cpu_seconds']),attention_statistics=avg(lambda r:max(x['attention_statistics_seconds'] for x in r['rank_costs'])),
        statistics_reduce_transfer_sync=avg(lambda r:max(x['statistics_reduce_transfer_sync_seconds'] for x in r['rank_costs'])),
        confidence=avg(lambda r:max(x['confidence_seconds'] for x in r['rank_costs'])),gpu_scalar_reduction=avg(lambda r:r['timings']['gpu_scalar_reduction_seconds']),cpu_subsets={})
    bench_ids=[pid for pid in matrix['ids'] if corpus.rows[pid]['ordinal']<2]
    if len(bench_ids)!=26:raise ValueError('Expected fixed 26-row timing cohort')
    def ensure_costs(subsets):
        combos=set()
        for fs in subsets:
            cpu=[f for f in CPU_FEATURES if f in fs]
            for count in range(1,len(cpu)+1):
                combos.update(itertools.combinations(cpu,count))
        missing=sorted(c for c in combos if ','.join(c) not in costs['cpu_subsets'])
        if missing:costs['cpu_subsets'].update(benchmark_subsets(output,corpus,bench_ids,missing,sealed))
    ensure_costs([[f] for f in CPU_FEATURES]);progress('single-feature costs measured')
    corrected=times.copy();corrected[:,0]+=probe
    signed=scores[:,[0]]-scores;positive=np.maximum(0,signed);normalized=corrected/np.median(times[:,0]);non_one=np.ones_like(scores);non_one[:,1]=0
    labels_by_objective={'maximize-one':non_one+1e-4*normalized,'minimize-time':normalized}
    def finish_candidate(c):
        required=used_features(c['tree']);c['required_features']=sorted(required)
        c['feature_cost_seconds']=cost_estimate(required,costs);c['cost_aware_ttft']=c['ttft']+c['feature_cost_seconds']
        return c
    old=finish_candidate(dict(old,extras=[]));old['index']=old.get('index',-1)
    original=__import__('json').loads((corpus.root/'training/search.json').read_text())
    def one(extras):
        extras=sorted(extras,key=ADDITIONS.index);ensure_costs([extras])
        tag='base' if not extras else '-'.join(f'{ADDITIONS.index(f):02d}' for f in extras)
        dest=output/'search'/f'{tag}.json'
        if dest.exists():return sealed(dest)['best']
        columns=list(FEATURES)+extras;x=np.asarray([[r.get(f) if r.get(f) is not None else np.nan for f in columns] for r in rows]);valid=np.isfinite(x).all(1)
        candidates=[];best=None
        for index,h in enumerate(GRID):
            labels=labels_by_objective[h['objective']]+h['weight']*(positive if h['loss_mode']=='positive' else signed)
            raw=fit_ordered(x[valid],positive[valid,1:],h,labels[valid])
            tree=compile_tree(raw,columns,names)
            # Fallback depends on features actually used by the compiled tree.
            policy=make(tree,names,{'study':file_hash(output/'protocol.json')})
            choices=[names.index(decide(policy,row)) for row in rows]
            c=finish_candidate(dict(index=index,setting=h,leaves=leaf_count(raw),extras=extras,tree=tree,
                **metrics(matrix,choices,old['traversal_seconds'])))
            if not extras and choices!=original[index]['choices']:raise ValueError('Original 2592-setting reproduction failed')
            candidates.append(c)
            if c['eligible'] and (best is None or rank(c)<rank(best)):best=c
        if best is None:raise ValueError('No feasible tree, including dense')
        sealed(dest,dict(extras=extras,candidates=candidates,best=best,searched=len(GRID)))
        progress('searched '+tag);return best
    base=one([]);singles=[one([f]) for f in ADDITIONS]
    shortlist=[c['extras'][0] for c in sorted(singles,key=rank)[:8]];sealed(output/'shortlist.json',dict(features=shortlist))
    pairs=[one(list(p)) for p in itertools.combinations(shortlist,2)];best_pair=min(pairs,key=rank)
    triples=[one(best_pair['extras']+[f]) for f in shortlist if f not in best_pair['extras']]
    all_candidates=[old,base,*singles,*pairs,*triples];winner=min(all_candidates,key=rank)
    drop=[dict(dropped=f,best=one([x for x in winner['extras'] if x!=f])) for f in winner['extras']]
    selection=dict(current=old,winner=winner,best_single=min(singles,key=rank),best_pair=best_pair,best_triple=min(triples,key=rank),drop_one=drop,
        strongest_simulated=min(all_candidates,key=lambda c:(c['ttft'],-c['accuracy'])),costs=costs,
        cost_scope='Measured extraction/transfer/sync plus CPU resident reductions. GPU attention dependencies charged full collected joint pass once (conservative upper-cost estimate); archive I/O excluded. Live measures actual required branches.')
    sealed(output/'selection.json',selection)
    policies={label:make(c['tree'],names,dict(protocol_sha256=file_hash(output/'protocol.json'),training_ids=matrix['ids'],heldout_ids=[])) for label,c in [('current',old),('winner',winner)]}
    for label,p in policies.items():
        if [names.index(decide(p,row)) for row in rows]!=selection[label]['choices']:raise ValueError('Export decision mismatch')
        check_boundaries(p,rows)
        sealed(output/'policies'/f'{label}.json',p)
        (output/'policies'/f'{label}.txt').write_text('\n'.join(rules(p['tree']))+'\n')
    sealed(output/'policy-lock.json',dict(policies={k:v['sha256'] for k,v in policies.items()},selection_sha256=file_hash(output/'selection.json'),no_refit_after_live=True))
    csv_write(output/'ranked-features.csv',[dict(rank=i+1,feature=c['extras'][0],accuracy_percent=100*c['accuracy'],loss_pp=100*c['loss'],
        comparable_ttft=c['ttft'],feature_cost_seconds=c['feature_cost_seconds'],cost_aware_ttft=c['cost_aware_ttft']) for i,c in enumerate(sorted(singles,key=rank))])
    decisions=[];tasks=[]
    for label,c in [('current',old),('winner',winner),('strongest-simulated',selection['strongest_simulated'])]:
        for task in sorted(set(matrix['tasks'])):
            indices=[i for i,t in enumerate(matrix['tasks']) if t==task]
            tasks.append(dict(policy=label,task=task,accuracy_percent=100*np.mean([matrix['scores'][i][c['choices'][i]] for i in indices]),
                comparable_ttft=float(np.mean([corrected[i,c['choices'][i]] for i in indices]))+c['traversal_seconds'],
                **{n:sum(c['choices'][i]==j for i in indices) for j,n in enumerate(names)}))
        for i,pid in enumerate(matrix['ids']):
            decisions.append(dict(policy=label,prompt_id=pid,task=matrix['tasks'][i],old_action=names[old['choices'][i]],action=names[c['choices'][i]],
                changed=c['choices'][i]!=old['choices'][i],accuracy=matrix['scores'][i][c['choices'][i]]))
    csv_write(output/'per-task-simulation.csv',tasks);csv_write(output/'decision-changes.csv',decisions)
    validate_search(output,matrix,sealed)
    atomic_json(output/'search-complete.json',dict(policy_lock_sha256=file_hash(output/'policy-lock.json'),features=76,settings=2592,training_samples=260,heldout_samples=0))
    return selection


def validate_search(output,matrix,sealed):
    names,_,scores,times,probe=validate_matrix(matrix);times=times.copy();times[:,0]+=probe;checked=0
    for path in (output/'search').glob('*.json'):
        unit=sealed(path)
        if unit['searched']!=2592 or len(unit['candidates'])!=2592:raise ValueError('Incomplete grid')
        for c in unit['candidates']:
            choices=np.asarray(c['choices']);accuracy=float(scores[np.arange(260),choices].mean());ttft=float(times[np.arange(260),choices].mean())+c['traversal_seconds']
            if abs(c['accuracy']-accuracy)>1e-12 or abs(c['ttft']-ttft)>1e-10 or c['eligible']!=(float(scores[:,0].mean())-accuracy<=.02+1e-12):raise ValueError('Candidate metric audit failed')
            if abs(c['cost_aware_ttft']-c['ttft']-c['feature_cost_seconds'])>1e-10:raise ValueError('Candidate cost audit failed')
            checked+=1
    sealed(output/'search-validation.json',dict(candidate_checks=checked,original_settings_replayed=2592,feature_count=76))
