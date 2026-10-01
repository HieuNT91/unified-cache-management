"""Offline depth-1..5 continuation of the frozen attention feature study."""
import fcntl
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from runner import attention_study as prior
from runner.attention_features import ADDITIONS, DEFINITIONS
from runner.extension_train import GRID as OLD_GRID, WEIGHTS, validate_matrix
from runner.router_policy import FEATURES
from runner.corpus_train import leaf, leaf_count
from runner.setups import atomic_json, file_hash

DEFAULT = prior.ROOT / 'outputs/ruler13-deeper-router-study-20260930'
SOURCE = prior.DEFAULT
PARTIAL = DEFAULT.with_name(DEFAULT.name + "-partial-first-pass")
GRID = [dict(objective=o, loss_mode=l, depth=d, min_leaf=m, penalty=p, weight=w)
        for o,l,d,m,p,w in itertools.product(['maximize-one','minimize-time'],
        ['positive','signed'], range(1,6), [5,10,20], [0.,.005,.02], WEIGHTS)]


def key(setting):
    return json.dumps(setting, sort_keys=True)


def tag(extras):
    return '-'.join(f'{ADDITIONS.index(f):02d}' for f in extras) or 'base'


def prepare(output):
    """Verify immutable source study and exact inputs before reusing any fits."""
    protocol = prior.sealed(SOURCE / 'protocol.json')
    if protocol['grid'] != OLD_GRID or protocol['definitions'] != DEFINITIONS or protocol['base_features'] != list(FEATURES):
        raise ValueError('Source feature/grid contract changed')
    pins = dict(protocol['source_pins'])
    pins.update(protocol['code_pins'])
    done = prior.sealed(SOURCE / 'complete.json')
    pins.update({str(SOURCE / p): h for p,h in done['files'].items()})
    pins[str(SOURCE / 'complete.json')] = file_hash(SOURCE / 'complete.json')
    if PARTIAL.exists():
        preserved = prior.read(PARTIAL / 'preservation.json')
        pins.update({str(PARTIAL / p):h for p,h in preserved['files'].items()})
        pins[str(PARTIAL / 'preservation.json')] = file_hash(PARTIAL / 'preservation.json')
    for p,h in pins.items():
        if file_hash(p) != h:
            raise ValueError('Source changed: ' + p)
    code_paths = ['runner/deeper_router_study.py','scripts/deeper_router_study.py',
                  'scripts/verify_deeper_router_study.py','runner/attention_study.py',
                  'runner/attention_features.py','runner/extension_train.py','runner/corpus_train.py',
                  'runner/router_policy.py','runner/corpus.py','runner/tree_policy.py',
                  'scripts/verify_attention_feature_study.py']
    value = dict(schema='offline-deeper-router-study-v1', source_pins=pins,
        code_pins={str(prior.ROOT / p): file_hash(prior.ROOT / p) for p in code_paths},
        grid=GRID, base_features=list(FEATURES), definitions=DEFINITIONS,
        loss_limit=.02, training_samples=260, heldout_samples=0,
        shortlist=8, pairs=28, triples=6, workers=4,
        per_depth='Best feasible among settings with each exact maximum depth; realized depth is reported separately.',
        timing=protocol['timing'], tie_order=protocol['tie_order'],
        reuse='Exact matrix, extracted columns, fitter source and setting equality. Reindex old settings into depth-1..5 grid.')
    return prior.sealed(output / 'protocol.json', value)


def extract(output, matrix):
    for i,pid in enumerate(matrix['ids']):
        unit = prior.sealed(SOURCE / 'features' / f'{pid}.json')
        if unit['id'] != pid or unit['input_sha256'] != matrix['input_hashes'][i]:
            raise ValueError('Extraction identity mismatch')
        if {f: unit['features'][f] for f in FEATURES} != matrix['features'][i]:
            raise ValueError('Original feature mismatch')
        prior.sealed(output / 'features' / f'{pid}.json', unit)
    prior.sealed(output / 'extraction-complete.json', dict(reused=260,
        files={p.name: file_hash(p) for p in sorted((output / 'features').glob('*.json'))}))


def incumbent(output, matrix):
    c = dict(prior.sealed(SOURCE / 'selection.json')['winner'])
    c['index'] = next(i for i,h in enumerate(GRID) if h == c['setting'])
    policy = prior.sealed(SOURCE / 'trees/winner.json')
    names = [a['id'] for a in matrix['actions']]
    rows = [prior.sealed(output / 'features' / f'{pid}.json')['features'] for pid in matrix['ids']]
    choices = [names.index(prior.decide(policy,r)) for r in rows]
    actual = prior.metrics(matrix, choices, c['traversal_seconds'])
    if choices != c['choices'] or not actual['eligible']:
        raise ValueError('Incumbent decisions changed')
    for k in ('accuracy','ttft','loss'):
        if abs(actual[k]-c[k]) > 1e-12:
            raise ValueError('Incumbent metric mismatch')
    if abs(c['ttft']-8.605654) > 0.0000005:
        raise ValueError('Unexpected incumbent')
    oracle = prior.read(prior.ORACLE)
    if oracle['matrix_sha256'] != file_hash(prior.EXT / 'matrix.json'):
        raise ValueError('Hindsight matrix mismatch')
    oc = oracle['results']['0.02']
    if abs(prior.metrics(matrix,oc['choices'],0)['ttft']-oc['ttft_seconds']) > 1e-12:
        raise ValueError('Hindsight metric mismatch')
    prior.sealed(output / 'incumbent.json',c)
    return c


def tree_shape(raw, minimum, maximum):
    def visit(n):
        if 'feature' not in n:
            if n['n'] < minimum:
                raise ValueError('Undersized leaf')
            return 0, 1
        ld,ll = visit(n['left']); rd,rl = visit(n['right'])
        if n['n'] != n['left']['n'] + n['right']['n']:
            raise ValueError('Invalid sample counts')
        return 1+max(ld,rd), ll+rl
    depth,leaves = visit(raw)
    if depth > maximum or leaves > 2**maximum:
        raise ValueError('Tree exceeds depth/leaf bound')
    return depth,leaves


def search_one(args):
    output, extras = args
    extras = sorted(extras,key=ADDITIONS.index)
    dest = output / 'search' / f'{tag(extras)}.json'
    if dest.exists():
        return prior.sealed(dest)['best']
    partial = PARTIAL / 'search' / dest.name
    if partial.exists():
        saved = prior.sealed(partial)
        if saved['extras'] != extras or [c['setting'] for c in saved['candidates']] != GRID:
            raise ValueError('Preserved grid mismatch')
        data = dict(saved, reused=len(GRID), fitted=0, preserved_source_sha256=file_hash(partial),
                    first_pass_reused=saved['reused'], first_pass_fitted=saved['fitted'])
        prior.sealed(dest,data)
        print(f'Reused completed expanded subset {tag(extras)}',flush=True)
        return data['best']
    matrix = prior.read(prior.EXT / 'matrix.json')
    old = prior.sealed(output / 'incumbent.json')
    rows = [prior.sealed(output / 'features' / f'{pid}.json')['features'] for pid in matrix['ids']]
    names,_,scores,times,probe = validate_matrix(matrix)
    columns = list(FEATURES)+extras
    x = np.asarray([[r.get(k) if r.get(k) is not None else np.nan for k in columns] for r in rows])
    valid = np.isfinite(x).all(1)
    corrected = times.copy(); corrected[:,0] += probe
    signed = scores[:,[0]]-scores; positive = np.maximum(0,signed)
    normalized = corrected / float(np.median(times[:,0]))
    non_one = np.ones_like(scores); non_one[:,1] = 0
    costs = {'maximize-one':non_one+1e-4*normalized, 'minimize-time':normalized}
    reuse = {}; source = SOURCE / 'search' / dest.name
    if source.exists():
        saved = prior.sealed(source)
        if saved['extras'] != extras or saved['searched'] != len(OLD_GRID):
            raise ValueError('Reuse column/grid mismatch')
        for index,c in enumerate(saved['candidates']):
            if c['index'] != index or c['setting'] != OLD_GRID[index] or c['extras'] != extras:
                raise ValueError('Reuse setting mismatch')
            reuse[key(c['setting'])] = c
    started = time.perf_counter(); candidates = []; best_depth = {}
    def fit(h):
        labels = costs[h['objective']]+h['weight']*(positive if h['loss_mode']=='positive' else signed)
        return prior.fit_ordered(x[valid],positive[valid,1:],h,labels[valid])
    for index,h in enumerate(GRID):
        cached = reuse.get(key(h))
        if cached is not None:
            c = dict(cached,index=index)
            actual = prior.metrics(matrix,c['choices'],old['traversal_seconds'])
            if any(abs(c[k]-actual[k]) > 1e-12 for k in ('accuracy','ttft','loss')) or c['actions'] != actual['actions'] or c['eligible'] != actual['eligible']:
                raise ValueError('Reused candidate metrics differ')
        else:
            raw = fit(h)
            tree_shape(raw,h['min_leaf'],h['depth'])
            choices = [int(leaf(raw,row)['action']) if ok else 0 for row,ok in zip(x,valid)]
            c = dict(index=index,setting=h,leaves=leaf_count(raw),extras=extras,
                     **prior.metrics(matrix,choices,old['traversal_seconds']))
        candidates.append(c)
        d = str(h['depth'])
        if c['eligible'] and (d not in best_depth or prior.ranking(c) < prior.ranking(best_depth[d])):
            best_depth[d] = c
    for d,c in list(best_depth.items()):
        raw = fit(c['setting'])
        realized,leaves = tree_shape(raw,c['setting']['min_leaf'],c['setting']['depth'])
        choices = [int(leaf(raw,row)['action']) if ok else 0 for row,ok in zip(x,valid)]
        if choices != c['choices'] or leaves != c['leaves']:
            raise ValueError('Rebuilt winner differs')
        best_depth[d] = dict(c,raw_tree=raw,tree=prior.compile_tree(raw,columns,names),
                             feature_names=columns,realized_depth=realized)
    winner = min(best_depth.values(),key=prior.ranking)
    prior.sealed(dest,dict(extras=extras,columns=columns,searched=len(GRID),
        reused=len(reuse),fitted=len(GRID)-len(reuse),source_sha256=file_hash(source) if reuse else None,
        best=winner,best_by_depth=best_depth,candidates=candidates,seconds=time.perf_counter()-started))
    print(f'{tag(extras)}: {winner["ttft"]:.6f}s, {100*winner["accuracy"]:.4f}%, depth {winner["setting"]["depth"]}; reused {len(reuse)}, fitted {len(GRID)-len(reuse)}',flush=True)
    return winner


def search(output, old):
    with ProcessPoolExecutor(max_workers=4) as pool:
        def batch(subsets):
            return list(pool.map(search_one,[(output,s) for s in subsets]))
        controls = batch([[],old['extras'],ADDITIONS])
        singles = batch([[f] for f in ADDITIONS])
        shortlist = [c['extras'][0] for c in sorted(singles,key=prior.ranking)[:8]]
        prior.sealed(output / 'shortlist.json',dict(features=shortlist))
        pairs = batch(list(itertools.combinations(shortlist,2)))
        pair = min(pairs,key=prior.ranking)
        triples = batch([pair['extras']+[f] for f in shortlist if f not in pair['extras']])
        winner = min([old,*controls,*singles,*pairs,*triples],key=prior.ranking)
        ablations = batch([[f for f in winner['extras'] if f != dropped] for dropped in winner['extras']])
    # As in the original study, drop-one refits are attribution diagnostics of
    # the selected winner, not another recursively adaptive selection stage.
    primary = [*controls,*singles,*pairs,*triples]
    by_depth = {}
    for result in primary:
        unit = prior.sealed(output / 'search' / f"{tag(result['extras'])}.json")
        for d,c in unit['best_by_depth'].items():
            if d not in by_depth or prior.ranking(c) < prior.ranking(by_depth[d]):
                by_depth[d] = c
    d = str(old['setting']['depth'])
    if prior.ranking(old) < prior.ranking(by_depth[d]): by_depth[d] = old
    winner = min([old,*by_depth.values()],key=prior.ranking)
    drops = [dict(dropped=f,best=c) for f,c in zip(winner['extras'],ablations)]
    prior.sealed(output / 'selection.json',dict(winner=winner,incumbent=old,
        original_five=controls[0],incumbent_seven=controls[1],all45=controls[2],
        best_single=min(singles,key=prior.ranking),best_pair=pair,best_triple=min(triples,key=prior.ranking),
        shortlist=shortlist,best_by_depth=by_depth,drop_one=drops,
        primary_subsets=sorted({tag(c['extras']) for c in primary}),
        ablation_scope='One-round diagnostic refits of the selected winner; excluded from primary selection, as in the source study.'))


def report(output,matrix,old):
    if (output / 'complete.json').exists():
        for p,h in prior.sealed(output / 'complete.json')['files'].items():
            if file_hash(output / p)!=h: raise ValueError('Completed artifact changed: '+p)
        return
    sel = prior.sealed(output / 'selection.json'); winner = sel['winner']
    policies = [('incumbent',old),('original-five',sel['original_five']),('incumbent-seven',sel['incumbent_seven']),
        ('all45',sel['all45']),('best-single',sel['best_single']),('best-pair',sel['best_pair']),
        ('best-triple',sel['best_triple']),('winner',winner)]
    policies += [('depth-'+d,c) for d,c in sorted(sel['best_by_depth'].items())]
    policies += [('drop-'+r['dropped'],r['best']) for r in sel['drop_one']]
    rows = [prior.sealed(output / 'features' / f'{pid}.json')['features'] for pid in matrix['ids']]
    names = [a['id'] for a in matrix['actions']]
    oracle = prior.read(prior.ORACLE)['results']['0.02']['ttft_seconds']
    # Benchmark every exported feature subset, sharing resident arrays across subsets.
    timing_path = output / 'cpu-overhead.json'
    if timing_path.exists(): timing = prior.sealed(timing_path)
    else:
        from runner.attention_features import additional_features
        corpus = prior.Corpus(prior.BASE); timings = {}
        sets = {','.join(c['extras']):c['extras'] for _,c in policies if c['extras']}
        samples = [i for task in sorted(set(matrix['tasks'])) for i in [j for j,t in enumerate(matrix['tasks']) if t==task][:2]]
        for i in samples:
            pid=matrix['ids'][i]; a=corpus.attention(pid); s=corpus.inputs(pid)
            layers=a['layers'].astype(np.float64).mean(0); prefix,end=s['boundaries'][1],s['boundaries'][-2]
            for k,fs in sets.items():
                got=additional_features(layers,a['scores'],prefix,end,fs)
                if got!={f:rows[i][f] for f in fs}: raise ValueError('Resident feature replay differs')
                repeats=[]
                for _ in range(5):
                    start=time.perf_counter(); additional_features(layers,a['scores'],prefix,end,fs); repeats.append(time.perf_counter()-start)
                timings.setdefault(k,[]).append(dict(id=pid,repeats=repeats,median_seconds=float(np.median(repeats))))
            print('Benchmarked resident features: '+pid,flush=True)
        timing=prior.sealed(timing_path,dict(samples=26,repeats=5,
            scope='Incremental features from resident equal-rank FP64 layers/native FP32 scores; includes validation/sort. Excludes archive I/O, equal-rank aggregation already needed by original features, transfer, TP synchronization and switching.',
            feature_sets={k:dict(mean_seconds=float(np.mean([r['median_seconds'] for r in v])),p95_seconds=float(np.percentile([r['median_seconds'] for r in v],95)),rows=v) for k,v in timings.items()}))
    summaries=[]; decisions=[]; task_rows=[]; checks=0; traversal={}
    for label,c in policies:
        export=dict(schema=prior.SCHEMA,offline_only=True,feature_names=c['feature_names'],tree=c['tree'],actions=names,
            additions=c['extras'],setting=c['setting'],training_ids=matrix['ids'],heldout_ids=[],
            protocol_sha256=file_hash(output / 'protocol.json'),missing_required_feature='nocache')
        checks+=prior.check_export(export,c.get('raw_tree'),rows,names)
        choices=[names.index(prior.decide(export,r)) for r in rows]
        actual=prior.metrics(matrix,choices,c['traversal_seconds'])
        if choices!=c['choices'] or not actual['eligible'] or abs(actual['ttft']-c['ttft'])>1e-12: raise ValueError('Export differs')
        prior.sealed(output / 'trees' / f'{label}.json',export)
        (output / 'trees' / f'{label}.txt').write_text('OFFLINE ONLY; missing required feature -> nocache.\n'+'\n'.join(prior.rules(c['tree']))+'\n')
        repeats=[]
        for _ in range(9):
            start=time.perf_counter()
            for _ in range(20):
                for row in rows:
                    node=c['tree']
                    while 'feature' in node: node=node['le'] if row[node['feature']]<=node['threshold'] else node['gt']
            repeats.append((time.perf_counter()-start)/(260*20))
        traversal[label]=float(np.median(repeats))
        for i,pid in enumerate(matrix['ids']):
            j=choices[i]; before=old['choices'][i]
            decisions.append(dict(policy=label,prompt_id=pid,task=matrix['tasks'][i],old_action=names[before],new_action=names[j],changed=j!=before,
                accuracy=matrix['scores'][i][j],baseline_accuracy=matrix['scores'][i][0],old_accuracy=matrix['scores'][i][before],
                ttft=matrix['answer_ttft'][i][j]+(matrix['probe_ttft'][i] if j==0 else 0)+c['traversal_seconds'],
                newly_one=j==1 and before!=1,newly_one_at_least_dense=j==1 and before!=1 and matrix['scores'][i][j]>=matrix['scores'][i][0]-1e-12))
        sub=[r for r in decisions if r['policy']==label]
        for task in sorted(set(matrix['tasks'])):
            rr=[r for r in sub if r['task']==task]
            task_rows.append(dict(policy=label,task=task,accuracy_percent=100*np.mean([r['accuracy'] for r in rr]),
                loss_pp=100*np.mean([r['baseline_accuracy']-r['accuracy'] for r in rr]),ttft_seconds=np.mean([r['ttft'] for r in rr]),
                **{a:sum(r['new_action']==a for r in rr) for a in names}))
        extra=timing['feature_sets'][','.join(c['extras'])]['mean_seconds'] if c['extras'] else 0.
        dense50=sum(r['ttft'] for r in sub if r['new_action'] in ('nocache','prophetkv-50'))/260
        summaries.append(dict(policy=label,features=c['extras'],maximum_depth=c['setting']['depth'],
            realized_depth=tree_shape(c['raw_tree'],c['setting']['min_leaf'],c['setting']['depth'])[0],leaves=c['leaves'],
            accuracy_percent=100*c['accuracy'],loss_pp=100*c['loss'],ttft_seconds=c['ttft'],actions=c['actions'],
            improvement_seconds=old['ttft']-c['ttft'],improvement_percent=100*(old['ttft']-c['ttft'])/old['ttft'],
            remaining_hindsight_gap_seconds=c['ttft']-oracle,gap_closed_percent=100*(old['ttft']-c['ttft'])/(old['ttft']-oracle),
            incremental_cpu_seconds=extra,measured_traversal_seconds=traversal[label],
            cpu_adjusted_ttft_seconds=c['ttft']-c['traversal_seconds']+traversal[label]+extra,
            dense_reduction=old['actions']['nocache']-c['actions']['nocache'],fifty_reduction=old['actions']['prophetkv-50']-c['actions']['prophetkv-50'],
            dense50_mean_ttft_contribution=dense50,dense50_ttft_percent=100*dense50/c['ttft'],
            changed_decisions=sum(r['changed'] for r in sub),newly_one=sum(r['newly_one'] for r in sub),
            newly_one_at_least_dense=sum(r['newly_one_at_least_dense'] for r in sub)))
    checked=reused=fitted=0; singles=[]
    _,_,scores,times,probe=validate_matrix(matrix); corrected=times.copy();corrected[:,0]+=probe
    for p in sorted((output / 'search').glob('*.json')):
        unit=prior.sealed(p); reused+=unit['reused'];fitted+=unit['fitted']
        if len(unit['extras'])==1: singles.append(unit['best'])
        for index,c in enumerate(unit['candidates']):
            choices=np.asarray(c['choices']); accuracy=float(scores[np.arange(260),choices].mean()); elapsed=float(corrected[np.arange(260),choices].mean())+old['traversal_seconds']
            if c['setting']!=GRID[index] or abs(accuracy-c['accuracy'])>1e-12 or abs(elapsed-c['ttft'])>1e-10 or c['eligible']!=(scores[:,0].mean()-accuracy<=.02+1e-12):raise ValueError('Candidate audit failed')
            if c['actions']!={a:int((choices==j).sum()) for j,a in enumerate(names)}:raise ValueError('Action count mismatch')
            checked+=1
    table=[dict(rank=i+1,feature=c['extras'][0],maximum_depth=c['setting']['depth'],leaves=c['leaves'],ttft_seconds=c['ttft'],accuracy_percent=100*c['accuracy'],improvement_seconds=old['ttft']-c['ttft']) for i,c in enumerate(sorted(singles,key=prior.ranking))]
    prior.csv_write(output / 'ranked-features.csv',table);prior.csv_write(output / 'per-task.csv',task_rows);prior.csv_write(output / 'decision-changes.csv',decisions)
    floor=100*(float(scores[:,0].mean())-.02)
    if winner['ttft']>old['ttft'] or abs(floor-89.57692307692308)>1e-10:raise ValueError('Nonregression/floor failure')
    summary=dict(training_only=True,training_samples=260,heldout_samples=0,settings_per_subset=4320,
        candidate_checks=checked,reused_settings=reused,new_fits=fitted,accuracy_floor_percent=floor,
        incumbent_ttft_seconds=old['ttft'],oracle_ttft_seconds=oracle,policies=summaries,export_checks=checks,
        pinned_primary_traversal_seconds=old['traversal_seconds'],measured_traversal_seconds=traversal,
        result='improvement found' if winner['ttft']<old['ttft']-1e-9 else 'no further improvement found',
        limitations='All260 prompts used for fitting, adaptive selection and scoring. Larger trees increase overfitting risk. Overall 2pp bound, not per-task. Finite greedy search, not global optimality. Mixed timing sessions; integrated probe reuse assumed. Transfer, synchronization and switching unmeasured. Offline only; no GPU inference, runtime integration or deployment.')
    inc=next(r for r in summaries if r['policy']=='incumbent')
    for r in summaries:r['cpu_adjusted_improvement_seconds']=inc['cpu_adjusted_ttft_seconds']-r['cpu_adjusted_ttft_seconds']
    atomic_json(output / 'report.json',summary)
    lines=[summary['result'].upper(),summary['limitations'],
        'Primary: selected answer TTFT + frozen traversal allowance + saved probe TTFT only for dense.',
        f'Incumbent {old["ttft"]:.6f}s; hindsight {oracle:.6f}s; accuracy floor {floor:.7f}%.',
        f'{checked} settings checked; {reused} reused; {fitted} new fits.',
        'Policy | accuracy% | loss pp | TTFT s | max/actual depth | leaves | dense/1/5/10/20/50']
    for r in summaries:lines.append(f'{r["policy"]} | {r["accuracy_percent"]:.4f} | {r["loss_pp"]:.4f} | {r["ttft_seconds"]:.6f} | {r["maximum_depth"]}/{r["realized_depth"]} | {r["leaves"]} | '+ '/'.join(str(r['actions'][a]) for a in names))
    w=next(r for r in summaries if r['policy']=='winner')
    lines += ['',f'Additional improvement: {w["improvement_seconds"]:.6f}s ({w["improvement_percent"]:.2f}%). Remaining hindsight gap: {w["remaining_hindsight_gap_seconds"]:.6f}s; closed {w["gap_closed_percent"]:.2f}% of incumbent gap.',
        f'CPU adjusted: {w["cpu_adjusted_ttft_seconds"]:.6f}s; incremental resident features {1000*w["incremental_cpu_seconds"]:.3f}ms; measured traversal {1e6*w["measured_traversal_seconds"]:.3f}us.',
        f'Decisions changed: {w["changed_decisions"]}; dense reduction {w["dense_reduction"]}; 50% reduction {w["fifty_reduction"]}; newly 1% {w["newly_one"]} ({w["newly_one_at_least_dense"]} at least dense on saved outcomes).',
        f'Dense/50% TTFT contribution: {inc["dense50_ttft_percent"]:.2f}% -> {w["dense50_ttft_percent"]:.2f}%.',
        '', 'Winner rules:',*prior.rules(winner['tree'])]
    (output / 'report.txt').write_text('\n'.join(lines)+'\n')
    prior.sealed(output / 'validation.json',dict(candidate_checks=checked,export_checks=checks,all_exports_feasible=True,accuracy_floor_percent=floor,nonregression=True))
    prior.sealed(output / 'complete.json',dict(files={str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name not in ('study.lock','run.log','complete.json','independent-validation.json')}))
    print('\n'.join(lines[:22]),flush=True)


def main(stage,output=DEFAULT):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('Require CPU-only and single numeric thread')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output / 'study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(output);matrix=prior.read(prior.EXT / 'matrix.json');validate_matrix(matrix)
        from scripts.verify_attention_feature_study import audit_sources
        prior.sealed(output / 'record-source-validation.json',audit_sources(output))
        extract(output,matrix);old=incumbent(output,matrix)
        print(f'Verified incumbent: {old["ttft"]:.9f}s, {100*old["accuracy"]:.7f}%',flush=True)
        if stage in ('all','resume','search'):search(output,old)
        if stage in ('all','resume','report'):report(output,matrix,old)
