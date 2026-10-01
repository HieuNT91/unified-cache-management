"""Resumable CPU-only attention feature attribution; never a runtime policy export."""
import csv
import fcntl
import itertools
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
from runner.attention_features import ADDITIONS, DEFINITIONS, additional_features
from runner.corpus import Corpus
from runner.corpus_train import leaf, leaf_count
from runner.extension_train import GRID, validate_matrix
from runner.router_policy import FEATURES, features_from_arrays
from runner.setups import atomic_json, file_hash
from runner.tree_policy import load, decide as runtime_decide

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'outputs/ruler13-attention-feature-study-20260930'
EXT=ROOT/'outputs/ruler13-router-add5-10-20260930/results'
BASE=ROOT/'outputs/ruler13-corpus20-clean-yarn4-20260929/results'
ORACLE=ROOT/'outputs/ruler13-router-oracle-bound-20260930/six-action-report.json'
SCHEMA='offline-attention-feature-study-tree-v1'


def read(path):return json.loads(Path(path).read_text())


def sealed(path, value=None):
    """Never overwrite a completed unit. Atomic payload plus digest envelope."""
    path=Path(path)
    if path.exists():
        envelope=read(path)
        import hashlib
        if hashlib.sha256(json.dumps(envelope['data'],sort_keys=True,allow_nan=False).encode()).hexdigest()!=envelope['sha256']:
            raise ValueError('Corrupt completed study unit: '+str(path))
        if value is not None and envelope['data']!=value:raise ValueError('Completed unit differs: '+str(path))
        return envelope['data']
    if value is None:raise FileNotFoundError(path)
    import hashlib
    atomic_json(path,dict(sha256=hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest(),data=value))
    return value


def pin_file(pins,path,expected=None):
    path=Path(path).resolve();actual=file_hash(path)
    if expected is not None and actual!=expected:raise ValueError('Source integrity failure: '+str(path))
    pins[str(path)]=actual


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    path=output/'protocol.json'
    if path.exists():
        protocol=sealed(path)
        for p,h in protocol['code_pins'].items():
            if file_hash(p)!=h:raise ValueError('Study implementation changed: '+p)
        for p,h in protocol['source_pins'].items():
            if file_hash(p)!=h:raise ValueError('Pinned source changed: '+p)
        return protocol
    matrix=read(EXT/'matrix.json');validate_matrix(matrix)
    pins={}
    for p in [EXT/'matrix.json',EXT/'sources.json',EXT/'protocol.json',EXT/'complete.json',ORACLE,
              EXT/'training/min-ttft.json',EXT/'training/min-ttft.json.sha256',EXT/'training/selection.json',EXT/'training/search.json',EXT/'training/complete.json']:
        pin_file(pins,p)
    completion=read(EXT/'training/complete.json')
    if completion['matrix_sha256']!=pins[str((EXT/'matrix.json').resolve())]:raise ValueError('Training matrix pin mismatch')
    for name,h in completion['files'].items():pin_file(pins,EXT/'training'/name,h)
    sources=read(EXT/'sources.json')['files']
    corpus=Corpus(BASE)
    archive_pins={}
    for pid in matrix['ids']:
        folder=BASE/'records/probe'/pid
        receipt=read(folder/'validated.json')
        pin_file(pins,folder/'validated.json',sources[str(folder/'validated.json')])
        for name,h in receipt['files'].items():
            p=folder/name
            if p.suffix=='.npz':archive_pins[str(p)]=h
            else:pin_file(pins,p,h)
        sample_path=corpus.prepared/corpus.rows[pid]['prepared']
        pin_file(pins,sample_path,corpus.rows[pid]['sha256'])
    for p,h in matrix['acceptance_hashes'].items():
        pin_file(pins,p,h)
        for name,digest in read(p)['files'].items():
            # Full answers/diagnostics are already validated by the completed collection;
            # pin their accepted digests, verify outcome records here (large diagnostic
            # arrays are not inputs to this CPU study).
            if name=='record.json':pin_file(pins,Path(p).parent/name,digest)
    code={str(ROOT/p):file_hash(ROOT/p) for p in ['runner/attention_features.py','runner/attention_study.py','scripts/attention_feature_study.py',
        'runner/corpus_train.py','runner/extension_train.py','runner/router_policy.py','runner/corpus.py','runner/tree_policy.py']}
    protocol=dict(schema='attention-feature-study-v1',source_pins=pins,archive_pins=archive_pins,code_pins=code,
        definitions=DEFINITIONS,base_features=list(FEATURES),grid=GRID,loss_limit=.02,
        shortlist=8,pairs=28,triples=6,training_samples=260,heldout_samples=0,
        tie_order='TTFT, higher accuracy, fewer additions, fewer leaves, shallower tree, stable feature indices, setting index',
        timing='selected answer TTFT + fixed measured baseline traversal allowance; saved probe TTFT only on dense; incremental CPU features reported separately',
        invalid_training='Undefined required feature forces dense; excluded from fitting splits, full cohort remains in scoring. No undefined values in the real cohort expected.')
    return sealed(path,protocol)


def baseline(output,matrix):
    names,x,scores,times,probe=validate_matrix(matrix)
    old=load(EXT/'training/min-ttft.json');saved=read(EXT/'training/selection.json')['min_ttft']
    choices=[names.index(runtime_decide(old,f)['action']) for f in matrix['features']]
    if choices!=saved['choices']:raise ValueError('Original baseline decisions changed')
    score=float(scores[np.arange(len(x)),choices].mean())
    elapsed=float(np.mean([times[i,j]+(probe[i] if j==0 else 0) for i,j in enumerate(choices)]))+saved['traversal_seconds']
    if abs(score-saved['accuracy'])>1e-12 or abs(elapsed-saved['ttft'])>1e-12:raise ValueError('Original metrics changed')
    oracle=read(ORACLE)
    if oracle['matrix_sha256']!=file_hash(EXT/'matrix.json'):raise ValueError('Oracle matrix mismatch')
    oc=oracle['results']['0.02']['choices']
    if abs(np.mean([times[i,j]+(probe[i] if j==0 else 0) for i,j in enumerate(oc)])-oracle['results']['0.02']['ttft_seconds'])>1e-12:raise ValueError('Oracle metric mismatch')
    candidate=dict(saved,extras=[],tree=old['tree'],feature_names=list(FEATURES),source='retained-current-router')
    return sealed(output/'baseline.json',candidate)


def extract(output,protocol,matrix):
    corpus=Corpus(BASE)
    for i,pid in enumerate(matrix['ids']):
        dest=output/'features'/f'{pid}.json'
        folder=BASE/'records/probe'/pid
        for p,h in protocol['archive_pins'].items():
            if Path(p).parent==folder and file_hash(p)!=h:raise ValueError('Archive changed: '+p)
        if dest.exists():sealed(dest);continue
        sample=corpus.inputs(pid);attention=corpus.attention(pid)
        layers=attention['layers'].astype(np.float64).mean(0)
        prefix,end=sample['boundaries'][1],sample['boundaries'][-2]
        original=features_from_arrays(layers,attention['scores'],prefix,end)
        if original!=matrix['features'][i]:raise ValueError('Five-feature exact replay failed: '+pid)
        start=time.perf_counter();extra=additional_features(layers,attention['scores'],prefix,end);elapsed=time.perf_counter()-start
        sealed(dest,dict(id=pid,input_sha256=matrix['input_hashes'][i],features=original|extra,seconds_all40_resident=elapsed,
            original_exact=True,eligible_positions=end-prefix))
        if (i+1)%10==0:print(f'Extracted and exactly replayed {i+1}/260',flush=True)
    sealed(output/'extraction-complete.json',dict(files={p.name:file_hash(p) for p in sorted((output/'features').glob('*.json'))},rows=260))


def fit_ordered(x,y,h,labels):
    """Original cost CART arithmetic/order, vectorized across candidate boundaries.

    Only the number of feature columns differs; no sklearn dtype conversions.
    """
    total=len(x);minimum=h['min_leaf']
    def build(indices,depth):
        values=y[indices]
        node=dict(n=len(indices),loss=values.mean(0).tolist(),action=int(np.argmin(labels[indices].mean(0))))
        if depth==h['depth'] or len(indices)<2*minimum:return node
        best=None;best_order=None
        for feature in range(x.shape[1]):
            order=indices[np.argsort(x[indices,feature],kind='stable')];v=x[order,feature]
            sums=labels[order].cumsum(0)
            ks=np.arange(minimum,len(order)-minimum+1)
            ks=ks[v[ks-1]!=v[ks]]
            if not len(ks):continue
            errors=(sums[ks-1].min(1)+(sums[-1]-sums[ks-1]).min(1))/total+h['penalty']
            pos=int(np.argmin(errors));k=int(ks[pos]);threshold=float(v[k-1]+(v[k]-v[k-1])/2)
            if threshold>=v[k]:threshold=float(v[k-1])
            candidate=(float(errors[pos]),feature,threshold,k)
            if best is None or candidate[:3]<best[:3]:best=candidate;best_order=order
        base=float(labels[indices].sum(0).min()/total)
        if best is not None and best[0]<base-1e-12:
            _,feature,threshold,k=best
            node.update(feature=feature,threshold=threshold,left=build(best_order[:k],depth+1),right=build(best_order[k:],depth+1))
        return node
    return build(np.arange(total),0)


def compile_tree(raw,columns,names):
    if 'feature' not in raw:return dict(action=names[raw['action']],training_samples=raw['n'])
    return dict(feature=columns[raw['feature']],threshold=raw['threshold'],le=compile_tree(raw['left'],columns,names),gt=compile_tree(raw['right'],columns,names))


def decide(tree,features):
    if tree['schema']!=SCHEMA or tree['offline_only'] is not True:raise ValueError('Not a study tree')
    if any(not isinstance(features.get(k),(int,float)) or isinstance(features.get(k),bool) or not math.isfinite(features[k]) for k in tree['feature_names']):return 'nocache'
    node=tree['tree']
    while 'feature' in node:node=node['le'] if features[node['feature']]<=node['threshold'] else node['gt']
    return node['action']


def ranking(c):
    return (c['ttft'],-c['accuracy'],len(c['extras']),c['leaves'],c['setting']['depth'],tuple(ADDITIONS.index(f) for f in c['extras']),c['index'])


def metrics(matrix,choices,traversal):
    n=len(choices);names=[a['id'] for a in matrix['actions']]
    # Scalar independent recomputation: task counts are equal, so overall == macro.
    accuracy=sum(float(matrix['scores'][i][j]) for i,j in enumerate(choices))/n
    elapsed=sum(float(matrix['answer_ttft'][i][j])+(float(matrix['probe_ttft'][i]) if j==0 else 0) for i,j in enumerate(choices))/n+traversal
    base=sum(row[0] for row in matrix['scores'])/n
    return dict(accuracy=accuracy,loss=base-accuracy,ttft=elapsed,eligible=base-accuracy<=.02+1e-12,
        actions={a:choices.count(j) for j,a in enumerate(names)},one_count=choices.count(1),choices=choices,traversal_seconds=traversal)


def search_one(output,matrix,rows,extras,old):
    extras=sorted(extras,key=ADDITIONS.index);tag='base' if not extras else '-'.join(str(ADDITIONS.index(k)).zfill(2) for k in extras)
    dest=output/'search'/f'{tag}.json'
    if dest.exists():return sealed(dest)['best']
    names,_,scores,times,probe=validate_matrix(matrix);columns=list(FEATURES)+extras
    x=np.asarray([[r.get(k) if r.get(k) is not None else np.nan for k in columns] for r in rows])
    valid=np.isfinite(x).all(1)
    corrected=times.copy();corrected[:,0]+=probe
    signed=scores[:,[0]]-scores;positive=np.maximum(0,signed);normalized=corrected/float(np.median(times[:,0]))
    non_one=np.ones_like(scores);non_one[:,1]=0
    costs={'maximize-one':non_one+1e-4*normalized,'minimize-time':normalized}
    candidates=[];winner=None;original=read(EXT/'training/search.json') if not extras else None
    started=time.perf_counter()
    for index,h in enumerate(GRID):
        labels=costs[h['objective']]+h['weight']*(positive if h['loss_mode']=='positive' else signed)
        raw=fit_ordered(x[valid],positive[valid,1:],h,labels[valid])
        choices=[int(leaf(raw,row)['action']) if ok else 0 for row,ok in zip(x,valid)]
        c=dict(index=index,setting=h,leaves=leaf_count(raw),extras=extras,**metrics(matrix,choices,old['traversal_seconds']))
        if original is not None and (choices!=original[index]['choices'] or abs(c['accuracy']-original[index]['accuracy'])>1e-12):
            raise ValueError('Ordered CART original grid replay differs at '+str(index))
        candidates.append(c)
        if c['eligible'] and (winner is None or ranking(c)<ranking(winner)):
            winner=dict(c,tree=compile_tree(raw,columns,names),raw_tree=raw,feature_names=columns)
    # A feature addition can hurt a greedy fit; retain baseline only for overall selection.
    data=dict(extras=extras,searched=len(GRID),best=winner,candidates=candidates,seconds=time.perf_counter()-started,
        original_grid_exact=not extras,invalid_rows=int((~valid).sum()))
    sealed(dest,data)
    print(f'Searched {tag}: {winner["ttft"]:.6f}s, accuracy {100*winner["accuracy"]:.4f}%, {data["seconds"]:.1f}s CPU wall',flush=True)
    return winner


def search(output,matrix,old):
    rows=[sealed(output/'features'/f'{pid}.json')['features'] for pid in matrix['ids']]
    base=search_one(output,matrix,rows,[],old)
    singles=[search_one(output,matrix,rows,[f],old) for f in ADDITIONS]
    shortlist=[c['extras'][0] for c in sorted(singles,key=ranking)[:8]]
    sealed(output/'shortlist.json',dict(features=shortlist))
    pairs=[search_one(output,matrix,rows,pair,old) for pair in itertools.combinations(shortlist,2)]
    best_pair=min(pairs,key=ranking)
    triples=[search_one(output,matrix,rows,best_pair['extras']+[f],old) for f in shortlist if f not in best_pair['extras']]
    winner=min([old,base,*singles,*pairs,*triples],key=ranking)
    ablations=[]
    for f in winner['extras']:
        remaining=[k for k in winner['extras'] if k!=f]
        result=search_one(output,matrix,rows,remaining,old)
        ablations.append(dict(dropped=f,best=result))
    sealed(output/'selection.json',dict(winner=winner,best_single=min(singles,key=ranking),best_pair=best_pair,
        best_triple=min(triples,key=ranking),shortlist=shortlist,drop_one=ablations))
    return rows


def check_export(policy,raw,rows,names):
    count=0
    for row in rows:
        expected=names[leaf(raw,[row[k] for k in policy['feature_names']])['action']] if raw is not None else None
        actual=decide(policy,row)
        if expected is not None and actual!=expected:raise ValueError('Export training mismatch')
        count+=1
    def visit(node,subset):
        nonlocal count
        if 'feature' not in node:return
        k=policy['feature_names'][node['feature']];t=node['threshold']
        witness=next(r for r in subset)
        for v in (np.nextafter(t,-np.inf),t,np.nextafter(t,np.inf)):
            row=dict(witness,**{k:float(v)})
            expected=names[leaf(raw,[row[f] for f in policy['feature_names']])['action']]
            if decide(policy,row)!=expected:raise ValueError('Threshold boundary mismatch')
            count+=1
        visit(node['left'],[r for r in subset if r[k]<=t]);visit(node['right'],[r for r in subset if r[k]>t])
    if raw is not None:visit(raw,rows)
    for k in policy['feature_names']:
        for value in (None,float('nan'),float('inf'),-float('inf')):
            row=dict(rows[0],**{k:value})
            if decide(policy,row)!='nocache':raise ValueError('Invalid fallback failed')
            count+=1
        row=dict(rows[0]);del row[k]
        if decide(policy,row)!='nocache':raise ValueError('Missing fallback failed')
        count+=1
    return count


def csv_write(path,rows):
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def rules(node,indent=''):
    if 'action' in node:return [indent+'return '+node['action']]
    return [indent+f'if {node["feature"]} <= {node["threshold"]!r}:']+rules(node['le'],indent+'    ')+[indent+'else:']+rules(node['gt'],indent+'    ')


def benchmark(output,matrix,selection):
    dest=output/'cpu-overhead.json'
    if dest.exists():return sealed(dest)
    corpus=Corpus(BASE);timings={};chosen=[c for c in [selection['best_single'],selection['winner']] if c['extras']]
    feature_sets={','.join(c['extras']):c['extras'] for c in chosen}
    samples=[]
    for task in sorted(set(matrix['tasks'])):
        samples += [pid for pid,t in zip(matrix['ids'],matrix['tasks']) if t==task][:2]
    for pid in samples:
        a=corpus.attention(pid);s=corpus.inputs(pid);layers=a['layers'].astype(np.float64).mean(0)
        prefix,end=s['boundaries'][1],s['boundaries'][-2]
        for key,fs in feature_sets.items():
            additional_features(layers,a['scores'],prefix,end,fs)
            durations=[]
            for _ in range(5):
                start=time.perf_counter();additional_features(layers,a['scores'],prefix,end,fs);durations.append(time.perf_counter()-start)
            timings.setdefault(key,[]).append(dict(id=pid,median_seconds=float(np.median(durations)),repeats=durations))
    summary={k:dict(mean_seconds=float(np.mean([r['median_seconds'] for r in v])),median_seconds=float(np.median([r['median_seconds'] for r in v])),
        p95_seconds=float(np.percentile([r['median_seconds'] for r in v],95)),max_seconds=max(r['median_seconds'] for r in v),rows=v) for k,v in timings.items()}
    return sealed(dest,dict(feature_sets=summary,samples=26,repeats=5,
        scope='Resident equal-rank float64 layer arrays and native FP32 scores; incremental function including validation/sort. Equal-rank aggregation already needed by original five features. Excludes archive I/O, transfer and synchronization. Single BLAS/OMP thread.'))


def report(output,matrix,old):
    if (output/'complete.json').exists():
        done=sealed(output/'complete.json')
        for p,h in done['files'].items():
            if file_hash(output/p)!=h:raise ValueError('Final report changed: '+p)
        return
    selection=sealed(output/'selection.json');winner=selection['winner'];rows=[sealed(output/'features'/f'{pid}.json')['features'] for pid in matrix['ids']]
    timing=benchmark(output,matrix,selection);names=[a['id'] for a in matrix['actions']]
    oracle=read(ORACLE)['results']['0.02']['ttft_seconds'];checks=0;decisions=[];task_rows=[];summaries=[]
    policies=[('baseline',old),('best-single',selection['best_single']),('best-pair',selection['best_pair']),('best-triple',selection['best_triple']),('winner',winner)]
    policies += [('drop-'+r['dropped'],r['best']) for r in selection['drop_one']]
    for label,c in policies:
        export=dict(schema=SCHEMA,offline_only=True,feature_names=c['feature_names'],tree=c['tree'],actions=names,
            additions=c['extras'],setting=c['setting'],training_ids=matrix['ids'],heldout_ids=[],protocol_sha256=file_hash(output/'protocol.json'),
            missing_required_feature='nocache')
        checks+=check_export(export,c.get('raw_tree'),rows,names)
        choices=[names.index(decide(export,r)) for r in rows]
        recomputed=metrics(matrix,choices,c['traversal_seconds'])
        if choices!=c['choices'] or abs(recomputed['ttft']-c['ttft'])>1e-10 or abs(recomputed['accuracy']-c['accuracy'])>1e-12 or not recomputed['eligible']:
            raise ValueError('Independent export metrics failed')
        sealed(output/'trees'/f'{label}.json',export)
        (output/'trees'/f'{label}.txt').write_text('OFFLINE STUDY ONLY. Missing/undefined required feature -> nocache.\n'+'\n'.join(rules(c['tree']))+'\n')
        for i,pid in enumerate(matrix['ids']):
            j=choices[i];prior=old['choices'][i]
            decisions.append(dict(policy=label,prompt_id=pid,task=matrix['tasks'][i],old_action=names[prior],new_action=names[j],changed=j!=prior,
                accuracy=matrix['scores'][i][j],baseline_accuracy=matrix['scores'][i][0],old_accuracy=matrix['scores'][i][prior],
                answer_ttft=matrix['answer_ttft'][i][j],dense_probe=matrix['probe_ttft'][i] if j==0 else 0,
                ttft=matrix['answer_ttft'][i][j]+(matrix['probe_ttft'][i] if j==0 else 0)+c['traversal_seconds'],
                newly_safe_one=j==1 and prior!=1 and matrix['scores'][i][1]>=matrix['scores'][i][0]-1e-12))
        for task in sorted(set(matrix['tasks'])):
            sub=[r for r in decisions if r['policy']==label and r['task']==task]
            task_rows.append(dict(policy=label,task=task,accuracy_percent=100*sum(r['accuracy'] for r in sub)/len(sub),
                loss_pp=100*sum(r['baseline_accuracy']-r['accuracy'] for r in sub)/len(sub),
                ttft_seconds=sum(r['ttft'] for r in sub)/len(sub),**{a:sum(r['new_action']==a for r in sub) for a in names}))
        incremental=timing['feature_sets'].get(','.join(c['extras']),{}).get('mean_seconds')
        summaries.append(dict(policy=label,features=c['extras'],accuracy_percent=100*c['accuracy'],loss_pp=100*c['loss'],ttft_seconds=c['ttft'],
            actions=c['actions'],gap_closed_percent=100*(old['ttft']-c['ttft'])/(old['ttft']-oracle),
            maximum_added_overhead_seconds=old['ttft']-c['ttft'],incremental_cpu_seconds=incremental,
            cpu_adjusted_ttft_seconds=c['ttft']+incremental if incremental is not None else None,
            dense_reduction=old['actions']['nocache']-c['actions']['nocache'],fifty_reduction=old['actions']['prophetkv-50']-c['actions']['prophetkv-50'],
            newly_safe_one=sum(r['newly_safe_one'] for r in decisions if r['policy']==label)))
    candidates_checked=0;ranked=[]
    _,_,scores,times,probe=validate_matrix(matrix);times=times.copy();times[:,0]+=probe
    for path in sorted((output/'search').glob('*.json')):
        result=sealed(path)
        for c in result['candidates']:
            choices=np.asarray(c['choices']);accuracy=float(scores[np.arange(260),choices].mean());ttft=float(times[np.arange(260),choices].mean())+c['traversal_seconds']
            if abs(accuracy-c['accuracy'])>1e-12 or abs(ttft-c['ttft'])>1e-10 or c['eligible']!=(float(scores[:,0].mean())-accuracy<=.02+1e-12):raise ValueError('Candidate vectorized audit failed')
            if c['actions']!={a:int((choices==j).sum()) for j,a in enumerate(names)}:raise ValueError('Candidate counts failed')
            candidates_checked+=1
        if len(result['extras'])==1:ranked.append(result['best'])
    ranked.sort(key=ranking)
    table=[dict(rank=i+1,feature=c['extras'][0],ttft_seconds=c['ttft'],accuracy_percent=100*c['accuracy'],loss_pp=100*c['loss'],
        reduction_seconds=old['ttft']-c['ttft'],gap_closed_percent=100*(old['ttft']-c['ttft'])/(old['ttft']-oracle),leaves=c['leaves'],setting_index=c['index'],**c['actions']) for i,c in enumerate(ranked)]
    csv_write(output/'ranked-features.csv',table);csv_write(output/'per-task.csv',task_rows);csv_write(output/'decision-changes.csv',decisions)
    if winner['ttft']>old['ttft']+1e-12:raise ValueError('Regression')
    # Measure final compiled traversal separately, without allowing scheduler noise to
    # break exact candidate ties. Primary uses the pinned baseline's ~0.2us allowance.
    traversal={}
    for label,c in policies:
        durations=[]
        for _ in range(5):
            start=time.perf_counter()
            for r in rows:
                node=c['tree']
                while 'feature' in node:node=node['le'] if r[node['feature']]<=node['threshold'] else node['gt']
            durations.append((time.perf_counter()-start)/260)
        traversal[label]=float(np.median(durations))
    summary=dict(training_only=True,training_samples=260,heldout_samples=0,settings_per_subset=2592,candidate_checks=candidates_checked,
        original_grid_replay=2592,feature_replays=260,export_checks=checks,oracle_ttft_seconds=oracle,policies=summaries,
        best_features=winner['extras'],result='improvement found' if winner['ttft']<old['ttft']-1e-9 else 'no improvement found',
        measured_traversal_seconds=traversal,pinned_primary_traversal_seconds=old['traversal_seconds'],
        timing='Selected answer TTFT + pinned baseline traversal allowance; saved probe TTFT charged only to dense. Additional resident-array CPU feature work reported separately.',
        limitations='All260 prompts used for fitting, selection and scoring. No held-out guarantee; many adaptive comparisons and finite greedy search. Mixed measurement sessions; assumed integrated probe reuse, not live measured router TTFT. Transfer, synchronization and switching costs unmeasured. Safe1 means observed accuracy at least dense on this training prompt, not a future guarantee. Hindsight actions are diagnostics only.')
    atomic_json(output/'report.json',summary)
    lines=[summary['result'].upper(),summary['limitations'],summary['timing'],
        f'Original trained router: {old["ttft"]:.6f}s. Hindsight optimum: {oracle:.6f}s.',
        'Policy | accuracy% | loss pp | TTFT s | gap closed% | dense/1/5/10/20/50']
    for r in summaries:
        lines.append(f'{r["policy"]} | {r["accuracy_percent"]:.4f} | {r["loss_pp"]:.4f} | {r["ttft_seconds"]:.6f} | {r["gap_closed_percent"]:.2f} | '+ '/'.join(str(r['actions'][a]) for a in names))
    lines+=['','Winning rules (offline only):',*rules(winner['tree']),'','Drop-one ablations refit all2592 settings on remaining additions; compare with best single to distinguish interactions.',
        'Primary traversal is held at the original measured allowance to keep exact decision-vector ties deterministic; separately measured traversal is in report.json.',
        'Resident-array CPU overhead and tolerable overhead (baseline TTFT minus policy TTFT) are in report.json/cpu-overhead.json.', '', 'Individual additions ranked by fastest feasible tree:']
    lines += [f'{r["rank"]:2d} {r["feature"]}: {r["ttft_seconds"]:.6f}s / {r["accuracy_percent"]:.4f}%' for r in table]
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    sealed(output/'validation.json',dict(original_features_exact=260,original_decisions_exact=260,original_grid_exact=2592,
        exported_checks=checks,candidate_metrics_checked=candidates_checked,all_exports_feasible=True,nonregression=True,
        source_archives=1040,feature_units_immutable=True,search_units_immutable=True))
    sealed(output/'complete.json',dict(complete=True,files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name not in ('study.lock','run.log','resume-validation.json')}))
    print('\n'.join(lines[:12]),flush=True)


def main(stage,output=DEFAULT):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('OPENBLAS_NUM_THREADS')!='1' or os.environ.get('OMP_NUM_THREADS')!='1':
        raise ValueError('Require CUDA_VISIBLE_DEVICES empty and OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=prepare(output);matrix=read(EXT/'matrix.json');old=baseline(output,matrix)
        if stage in ('extract','all','resume'):extract(output,protocol,matrix)
        if stage in ('search','all','resume'):search(output,matrix,old)
        if stage in ('report','all','resume'):report(output,matrix,old)
