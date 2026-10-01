"""Immutable data units and stages for the authorized GPU feature study."""
import fcntl
import json
import os
import time
import uuid
from pathlib import Path
import numpy as np
from runner.setups import atomic_json,file_hash,fingerprint
from runner.corpus_extension import LinkedCorpus,check_pins
from runner.attention_study import baseline as old_baseline,read
from runner.gpu_study_features import ADDITIONS,REGISTERED,DEFINITIONS,CPU_FEATURES,additional_features
from runner.router_policy import FEATURES,features_from_arrays
from runner.extension_train import validate_matrix


def sealed(path,value=None):
    path=Path(path)
    if value is None:
        v=read(path)
        if v['sha256']!=fingerprint(v['data']):raise ValueError('Corrupt study unit: '+str(path))
        return v['data']
    json.dumps(value,allow_nan=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    with (path.parent/'.publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            old=sealed(path)
            if old!=value:raise FileExistsError('Completed unit differs: '+str(path))
            return old
        atomic_json(path,dict(sha256=fingerprint(value),data=value))
    return value


def progress(root,stage,message):
    atomic_json(root/'progress.json',dict(stage=stage,message=message,at=time.time()));print(stage+': '+message,flush=True)


def prepare(root,extension,source_worktree):
    from scripts.router_control import code_hashes
    if root.exists() and any(root.iterdir()):raise FileExistsError('Use a fresh study result directory')
    root.mkdir(parents=True,exist_ok=True);corpus=LinkedCorpus(extension)
    done=read(extension/'complete.json')
    if not done['complete'] or not done['owned_engines_exited']:raise ValueError('Source GPU collection incomplete')
    matrix=read(extension/'matrix.json');validate_matrix(matrix)
    if done['matrix_sha256']!=file_hash(extension/'matrix.json'):raise ValueError('Matrix pin mismatch')
    source_pins={str(extension/name):file_hash(extension/name) for name in ('protocol.json','sources.json','complete.json','matrix.json')}
    for p in (extension/'training').rglob('*'):
        if p.is_file():source_pins[str(p)]=file_hash(p)
    # Pin accepted original outcome/probe bodies and their original validation receipts.
    for pid in matrix['ids']:
        for case in ['probe',*[a['id'] for a in matrix['actions']]]:
            base=corpus.base.root if case=='probe' or case in corpus.base.actions else extension
            folder=base/'records'/case/pid;receipt=read(folder/'validated.json')
            source_pins[str(folder/'validated.json')]=file_hash(folder/'validated.json')
            for name,h in receipt['files'].items():source_pins[str(folder/name)]=h
        for key in ('prepared','raw'):
            p=corpus.prepared/corpus.rows[pid][key];source_pins[str(p)]=file_hash(p)
    code=code_hashes();protocol=dict(corpus.protocol,schema='single-probe-gpu-study-v1',kind='gpu-feature-study',
        cache_root=str(root/'cache'),extension=str(extension),source_worktree=str(source_worktree),
        definitions=DEFINITIONS,additions=ADDITIONS,code=code,runtime_sha256=fingerprint(code),source_pins=source_pins,
        diagnostic_probes=260,live_probes=520,live_answers=520,total_scheduled_probes=780,
        loss_limit=.02,search_settings=2592,training_samples=260,heldout_samples=0,
        cost_estimate='Comparable selected answer + traversal + saved probe ONLY for dense; measured shared feature dependencies counted once. GPU attention uses conservative collected joint-pass cost.',
        ordering='diagnostic in original order, then CPU search/lock after owned engine exit, then current/winner alternating by prompt index',
        scratch_limit_bytes=256*2**20,watchdog_seconds=900,operational_retries=1)
    atomic_json(root/'protocol.json',protocol)
    progress(root,'prepared','76 additions; 260 diagnostic probes then 520 live probe/answer pairs')
    return protocol


def verify(root,full=False):
    from scripts.router_control import code_hashes
    protocol=read(root/'protocol.json')
    if protocol['code']!=code_hashes() or protocol['runtime_sha256']!=fingerprint(protocol['code']):raise ValueError('Frozen study runtime changed')
    corpus=LinkedCorpus(Path(protocol['extension']));matrix=read(corpus.root/'matrix.json');validate_matrix(matrix)
    if protocol['groups']!=corpus.protocol['groups'] or protocol['prompt_ids']!=matrix['ids']:raise ValueError('Source UUID/cohort drift')
    check_pins({k:v for k,v in protocol['source_pins'].items() if full or not k.endswith('.npz')})
    from importlib.metadata import version
    from runner.config import VERSIONS
    from scripts.corpus_inputs import spec
    if any(version(n).split('+')[0]!=v for n,v in VERSIONS.items()):raise ValueError('Runtime environment drift')
    for n,h in spec()['model_fingerprints'].items():
        if file_hash(Path(protocol['model'])/n)!=h:raise ValueError('Model/tokenizer changed')
    return protocol,corpus,matrix


def extract(root):
    protocol,corpus,matrix=verify(root)
    import runner.attention_study as old
    old.EXT=corpus.root;old.BASE=corpus.base.root;old.ORACLE=Path(protocol['source_worktree'])/'outputs/ruler13-router-oracle-bound-20260930/six-action-report.json'
    current=old_baseline(root/'baseline-replay',matrix)
    sealed(root/'baseline.json',current)
    for i,pid in enumerate(matrix['ids']):
        target=root/'cpu-features'/f'{pid}.json'
        if target.exists():sealed(target);continue
        a=corpus.base.attention(pid);s=corpus.inputs(pid);layers=a['layers'].astype(np.float64).mean(0)
        original=features_from_arrays(layers,a['scores'],s['boundaries'][1],s['boundaries'][-2])
        if original!=matrix['features'][i]:raise ValueError('Original features did not replay exactly')
        start=time.perf_counter();extra=additional_features(layers,a['scores'],s['boundaries'][1],s['boundaries'][-2])
        sealed(target,dict(features=original|extra,input_sha256=matrix['input_hashes'][i],resident_seconds=time.perf_counter()-start))
        progress(root,'extract',f'{i+1}/260 exact source replays')
    sealed(root/'extraction-complete.json',dict(rows=260,baseline_sha256=file_hash(root/'baseline.json'),
        files={p.name:file_hash(p) for p in sorted((root/'cpu-features').glob('*.json'))}))


def accepted(root,phase,pid):
    folder=root/'records'/phase/pid;target=folder/'validated.json'
    if not target.exists():return None
    receipt=sealed(target)
    if receipt['protocol_sha256']!=file_hash(root/'protocol.json'):raise ValueError('Accepted protocol changed')
    for name,h in receipt['files'].items():
        if file_hash(folder/name)!=h:raise ValueError('Accepted record corrupted')
    record=sealed(folder/'record.json')
    protocol=read(root/'protocol.json')
    if record['prompt_id']!=pid or record['gpu_uuids']!=protocol['groups'][0] or record['cache_deletion']['deleted_shards']<=0:raise ValueError('Accepted identity/lifecycle mismatch')
    if len(record['retirement'])!=4 or any(not r['quiescent'] or r['transfers']['pending'] or r['request_bookkeeping'] for r in record['retirement']):raise ValueError('Accepted retirement changed')
    if any(not a['normalized'] or a['delta_amplitude']!=1. or a['aliases_native_table'] for a in record['alignment_audit']):raise ValueError('Accepted YaRN evidence changed')
    return record


def _publish(root,phase,pid,record,diagnostics,arrays,sample,probe_ds=None):
    from runner.gpu_study_runtime import save_arrays
    folder=root/'records'/phase/pid
    if (folder/'validated.json').exists():raise FileExistsError('Duplicate publication')
    if folder.exists():
        history=root/'incomplete'/f'{phase}-{pid}-{uuid.uuid4().hex}';history.parent.mkdir(parents=True,exist_ok=True);folder.rename(history)
    folder.mkdir(parents=True)
    record['artifacts']=save_arrays(folder,arrays,sample)
    sealed(folder/'record.json',record);atomic_json(folder/'diagnostics.json',diagnostics)
    if probe_ds is not None:atomic_json(folder/'probe-diagnostics.json',probe_ds)
    sealed(folder/'validated.json',dict(protocol_sha256=file_hash(root/'protocol.json'),
        files={p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and not p.name.startswith('.')}))
    accepted(root,phase,pid)


def collect(root,phase,attempt):
    from runner.gpu_study_runtime import StudyEngine,native_validate,reference_validate
    from runner.gpu_study_policy import decide
    from runner.corpus_runtime import paired
    from runner.corpus import match_answer
    from runner.setups import check_environment
    from runner import gpu_study_worker as rpc
    protocol,corpus,matrix=verify(root)
    if check_environment(4)!=protocol['groups'][0]:raise ValueError('Wrong UUID group')
    cases=['diagnostic'] if phase=='collect' else ['current','winner']
    policies={label:sealed(root/'policies'/f'{label}.json') for label in cases} if phase=='live' else {}
    if policies:
        lock=sealed(root/'policy-lock.json')
        if {k:v['sha256'] for k,v in policies.items()}!=lock['policies']:raise ValueError('Locked policy changed')
    pending=[corpus.rows[p] for p in matrix['ids'] if any(accepted(root,c,p) is None for c in cases)]
    if not pending:return
    with (root/'group0.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine=StudyEngine(root,protocol,0,'cached',attempt,pending)
        try:
            for row in pending:
                pid=row['id'];sample=corpus.inputs(pid);saved=corpus.attention(pid,replay=True);engine.begin(row);staged=[]
                order=cases if matrix['ids'].index(pid)%2==0 else list(reversed(cases))
                for case in order:
                    if accepted(root,case,pid) is not None:continue
                    if phase=='collect':
                        reference=pid==matrix['ids'][0]
                        record,ds,ranks,layers=engine.probe_study(REGISTERED,reference)
                        native_validate(ranks,sample,saved);match_answer(ds,sample,'prophetkv-1',saved,protocol['actions'])
                        source=sealed(root/'cpu-features'/f'{pid}.json')
                        if any(record['features'][f]!=v for f,v in source['features'].items()):raise ValueError('Passive probe changed CPU features')
                        if reference:record['reference_validation']=reference_validate(ranks,sample['boundaries'][1])
                        record['statistics']=[dict(rank=r['rank'],statistics=r['statistics'],confidence=r['confidence']) for r in ranks]
                        record['native_transfer_rpc_seconds']=max(0.,record['timings']['export_rpc_seconds']-max(r['timings']['statistics_reduce_transfer_sync_seconds'] for r in ranks))
                        engine.llm.collective_rpc(rpc.discard);probe_ds=None
                    else:
                        record,ds,probe_ds,ranks=engine.routed(policies[case],case)
                        native_validate(ranks,sample,saved);action=record['executed_action']
                        control=corpus.outcome(pid,action);base=corpus.base.root if action in corpus.base.actions else corpus.root
                        control_ds=read(base/'records'/action/pid/'diagnostics.json')
                        record['paired_validation']=paired(record,ds,control,control_ds)
                        record['control_sha256']=file_hash(base/'records'/action/pid/'result.json')
                        if action!='nocache':match_answer(ds,sample,action,saved,protocol['actions'])
                    staged.append((case,record,ds,ranks,probe_ds))
                deletion=engine.end()
                for case,record,ds,ranks,probe_ds in staged:
                    record['cache_deletion']=deletion;publish(root,case,pid,record,ds,ranks,sample,probe_ds)
                    if case=='diagnostic':sealed(root/'features'/f'{pid}.json',dict(features=record['features'],input_sha256=row['sha256']))
                    progress(root,phase,case+' '+pid+' accepted')
                startup=root/('startup-verification.json' if phase=='collect' else 'live-startup-verification.json')
                if pid==matrix['ids'][0] and not startup.exists():
                    initial=read(engine.session/'initialization.json')
                    sealed(startup,dict(prompt_id=pid,phase=phase,validated=True,all_rank_native_scores_masks=True,all64_selection=True,
                        yarn_normalized=True,cache_immutable=True,retirement=True,cache_deletion=deletion,
                        warmup_shards=initial['warmup_readiness']['verified_shards'],reference_validation=staged[0][1].get('reference_validation'),
                        one_scoring_probe_per_routed_answer=phase=='live'))
        finally:engine.close()


def report(root):
    protocol,corpus,matrix=verify(root)
    selection=sealed(root/'selection.json');rows=[];summaries={};tasks=[]
    for label in ('current','winner'):
        records=[accepted(root,label,p) for p in matrix['ids']]
        if any(r is None for r in records):raise ValueError('Live comparison incomplete')
        for pid,task,r in zip(matrix['ids'],matrix['tasks'],records):
            rows.append(dict(policy=label,prompt_id=pid,task=task,action=r['executed_action'],accuracy=r['accuracy'],live_ttft=r['timings']['ttft_seconds']))
        summaries[label]=dict(accuracy_percent=100*np.mean([r['accuracy'] for r in records]),
            live_ttft_seconds=float(np.mean([r['timings']['ttft_seconds'] for r in records])),
            actions={a['id']:sum(r['executed_action']==a['id'] for r in records) for a in protocol['actions']})
        for task in sorted(set(matrix['tasks'])):
            rr=[r for r,t in zip(records,matrix['tasks']) if t==task]
            tasks.append(dict(policy=label,task=task,accuracy_percent=100*np.mean([r['accuracy'] for r in rr]),live_ttft_seconds=float(np.mean([r['timings']['ttft_seconds'] for r in rr]))))
    dense=100*np.mean([r[0] for r in matrix['scores']]);winner=summaries['winner'];current=summaries['current']
    confirmed=bool(dense-winner['accuracy_percent']<=2+1e-10 and winner['live_ttft_seconds']<current['live_ttft_seconds'])
    paired_rows=[dict(prompt_id=p,ttft_delta_seconds=rows[260+i]['live_ttft']-rows[i]['live_ttft'],accuracy_delta=rows[260+i]['accuracy']-rows[i]['accuracy']) for i,p in enumerate(matrix['ids'])]
    from runner.attention_study import csv_write
    csv_write(root/'live-decisions.csv',rows);csv_write(root/'live-per-task.csv',tasks);csv_write(root/'live-paired.csv',paired_rows)
    result=dict(confirmed_improvement=confirmed,training_only=True,heldout_samples=0,dense_accuracy_percent=dense,live=summaries,
        best_features=selection['winner']['extras'],strongest_simulated=selection['strongest_simulated'],
        diagnostic_probes=260,live_probes=520,live_answers=520,all_fixed_action_comparisons_validated=True,
        no_refit_after_live=True,claim='Specific to this training cohort; no deployment or held-out claim.')
    sealed(root/'report.json',result)
    (root/'report.txt').write_text(('CONFIRMED LIVE IMPROVEMENT' if confirmed else 'NO CONFIRMED LIVE IMPROVEMENT')+'\n'+
        f"Current: {current['accuracy_percent']:.4f}%, {current['live_ttft_seconds']:.6f}s live TTFT.\nWinner: {winner['accuracy_percent']:.4f}%, {winner['live_ttft_seconds']:.6f}s live TTFT.\n"+
        'Features: '+', '.join(result['best_features'])+'\n'+result['claim']+'\n')
    sealed(root/'complete.json',dict(report_sha256=file_hash(root/'report.json'),owned_engines_exited=True,diagnostic_probes=260,live_probes=520,answers=520))
    return result


def publish(root,phase,pid,record,diagnostics,arrays,sample,probe_ds=None):
    with (root/'record-publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _publish(root,phase,pid,record,diagnostics,arrays,sample,probe_ds)
