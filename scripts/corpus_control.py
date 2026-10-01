#!/usr/bin/env python3
"""Reusable corpus and supplied-tree control plane, always CPU-only."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.setups import atomic_json,file_hash,fingerprint
from runner.tree_policy import actions,DEFAULT_ACTIONS,load
from runner.corpus_records import accepted,record_dir,protocol_identity
from runner.router_process import identity,alive,group_alive
from scripts.router_control import code_hashes,environment,idle as legacy_idle,terminate_owned,cleanup_caches
from scripts.corpus_inputs import TASKS,prepare,verify_prepared,prepared_rows


def idle(root):
    root=Path(root);legacy_idle(root)
    for path in (root/'sessions').glob('*/ownership.json'):
        state=json.loads(path.read_text())
        if alive(state) or group_alive(state['pid']):raise ValueError('Owned worker/engine remains alive')
    for path in [root/'run.lock',*root.glob('group*.lock')]:
        if not path.exists():continue
        with path.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ValueError('Owned supervisor/worker lock is held') from error



def groups(a,b=None):
    values=[g.split(',') for g in (a,b) if g]
    if (len(values) not in (1,2) or any(len(g)!=4 for g in values) or
            len(set(sum(values,[])))!=4*len(values) or any(not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',s) for s in sum(values,[]))):
        raise ValueError('Explicit one or two disjoint TP4 groups of full GPU UUIDs required')
    return values


def read_rows(settings):
    prepared=Path(settings['prepared'])
    if settings['dataset']=='ruler':return prepared_rows(prepared)
    receipt=json.loads((prepared/'preparation.json').read_text())
    from runner.corpus import relative
    for name,digest in receipt['files'].items():
        if file_hash(relative(prepared,name))!=digest:raise ValueError('LongBench preparation changed')
    rows=[json.loads(line) for line in (prepared/'manifest.jsonl').read_text().splitlines()]
    if len(rows)!=503 or len({r['id'] for r in rows})!=503:raise ValueError('Expected all 503 LongBench v2 rows')
    for i,row in enumerate(rows):
        row.update(ordinal=i,sha256=receipt['files'][row['prepared']])
    return rows


def check_hardware(settings):
    from runner.tree_profiles import PROFILES
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    inventory={r[0].strip():dict(name=r[1].strip(),total_mib=float(r[2]),free_mib=float(r[3])) for r in csv.reader(raw.splitlines())}
    requested=sum(settings['groups'],[])
    index=json.loads((Path(settings['model'])/'model.safetensors.index.json').read_text())
    weights=index['metadata']['total_size']/4
    kv=PROFILES[settings['dataset']]['kv_tokens']*262144/4
    # Per-rank BF16 weights + full position KV + explicit activation/runtime reserve.
    required=(weights+kv+8*2**30)/.9/2**20
    for gpu in requested:
        if gpu not in inventory:raise ValueError('Configured UUID is unavailable')
        if not any(model in inventory[gpu]['name'] for model in ('A800','L20')):raise ValueError('This profile supports A800 and L20; qualify other hardware explicitly')
        if inventory[gpu]['total_mib']<required or inventory[gpu]['free_mib']<required:
            raise ValueError(f'Profile cannot fit GPU {gpu}: requires at least {required:.0f} MiB; no precision/input shortening is applied')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if set(requested)&{r.strip() for r in busy}:raise ValueError('Configured GPUs are occupied; existing processes left untouched')
    return dict(groups=[[dict(uuid=g,name=inventory[g]['name'],total_mib=inventory[g]['total_mib']) for g in group] for group in settings['groups']],
                required_mib_per_rank=required)


def configure(a):
    if getattr(a,'inference',False) and not a.tree:raise ValueError('Inference configure requires a supplied --tree')
    if not 1<=a.limit_per_task<=200:raise ValueError('limit-per-task must be 1..200')
    settings=dict(schema='ruler-corpus-execution-v1',kind='inference' if a.tree else 'collection',dataset=a.dataset,
        model=str(a.model),prepared=str(a.prepared),cache_root=str(a.cache_root),groups=groups(a.gpu_a,a.gpu_b),
        ruler=str(a.ruler) if a.ruler else None,data=str(a.data) if a.data else None,code=code_hashes(),
        watchdog_seconds=900,operational_retries=1)
    if a.tree:
        tree=load(a.tree);settings.update(actions=tree['actions'],tree_sha256=file_hash(a.tree),tree=str(a.tree))
        if a.dataset=='ruler':
            if not a.corpus:raise ValueError('RULER tree inference requires --corpus for frozen held-out inputs')
            from runner.corpus_training_data import open_corpus
            from runner.corpus_eval import evaluation_ids
            corpus=open_corpus(a.corpus,a.prepared,tree=tree)
            settings['selected_ids']=evaluation_ids(corpus,tree)
            settings['heldout_inputs']={pid:corpus.rows[pid]['sha256'] for pid in settings['selected_ids']}
        else:settings['selected_ids']=None
    else:
        if a.dataset!='ruler':raise ValueError('LongBench is evaluation-only; collection/training forbidden')
        settings['actions']=json.loads(a.actions.read_text()) if a.actions else DEFAULT_ACTIONS
    actions(settings['actions']);a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle(a.root)
        target=a.root/'settings.json'
        if target.exists():
            if json.loads(target.read_text())!=settings:raise ValueError('Settings changed; use a new collection version/output path')
        else:
            if any(p.name!='launch.lock' for p in a.root.iterdir()):raise ValueError('Use a new empty result directory')
            atomic_json(target,settings)
            if a.tree:
                (a.root/'tree.json').write_bytes(a.tree.read_bytes())
                (a.root/'tree.json.sha256').write_text(file_hash(a.tree)+'  tree.json\n')
        tranche=a.root/'tranche.json'
        if tranche.exists() and a.limit_per_task<json.loads(tranche.read_text())['limit_per_task'] and (a.root/'records').exists():
            raise ValueError('Cannot shrink a collected tranche')
        atomic_json(tranche,dict(limit_per_task=a.limit_per_task))
    return settings


def verify(root,hardware=True,publish=True):
    root=Path(root);s=json.loads((root/'settings.json').read_text())
    if s['code']!=code_hashes():raise ValueError('Configured implementation changed; use a new version')
    rows=read_rows(s)
    limit=json.loads((root/'tranche.json').read_text())['limit_per_task']
    if s['dataset']=='ruler':
        wanted=s.get('selected_ids')
        rows=[r for r in rows if r['id'] in wanted] if wanted else [r for r in rows if r['ordinal']<limit]
        if wanted:
            if {r['id'] for r in rows}!=set(wanted) or any(r['sha256']!=s['heldout_inputs'][r['id']] for r in rows):raise ValueError('Frozen held-out inputs changed/missing')
        elif len(rows)!=13*limit:raise ValueError('Prepare the requested tranche before verification')
        plan_sha=file_hash(Path(s['prepared'])/'plan.json')
    else:plan_sha=file_hash(Path(s['prepared'])/'preparation.json')
    from runner.corpus import relative
    from runner.tree_profiles import validate
    from run import check_model
    check_model(Path(s['model']))
    model_hash=file_hash(Path(s['model'])/'config.json')
    model_index=json.loads((Path(s['model'])/'model.safetensors.index.json').read_text())
    if not all(relative(s['model'],name).is_file() for name in set(model_index['weight_map'].values())):
        raise ValueError('Missing local model weight shards')
    if s['dataset']=='longbench-v2':
        preparation=json.loads((Path(s['prepared'])/'preparation.json').read_text())
        for name,digest in preparation['spec']['tokenizer_files'].items():
            if file_hash(relative(s['model'],name))!=digest:raise ValueError('LongBench tokenizer/model metadata changed')
    from importlib.metadata import version
    from runner.config import VERSIONS
    if any(version(name).split('+')[0]!=expected for name,expected in VERSIONS.items()):raise ValueError('Install the clean README pinned runtime versions')
    if s['dataset']=='ruler':
        from scripts.corpus_inputs import spec
        for name,digest in spec()['model_fingerprints'].items():
            if file_hash(Path(s['model'])/name)!=digest:raise ValueError('Pinned model/tokenizer mismatch')
    for row in rows:
        path=relative(s['prepared'],row['prepared'])
        if file_hash(path)!=row['sha256']:raise ValueError('Prepared input changed')
        sample=json.loads(path.read_text());validate(sample,s['dataset'])
        if sample['model_config_sha256']!=model_hash:raise ValueError('Prepared model mismatch')
    protocol={k:v for k,v in s.items() if k not in ('ruler','data','tree')}
    protocol.update(plan_sha256=plan_sha)
    old=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else None
    hw=check_hardware(s) if hardware else old.get('hardware') if old else None
    if hw is None:raise ValueError('Verify hardware before starting a worker')
    protocol['hardware']=hw
    if old is not None and old!=protocol:raise ValueError('Frozen protocol changed')
    if s['kind']=='inference':
        load(root/'tree.json')
        if file_hash(root/'tree.json')!=s['tree_sha256']:raise ValueError('Supplied tree changed')
    if old is None and publish:
        with (root/'protocol.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            target=root/'protocol.json'
            if target.exists():
                if json.loads(target.read_text())!=protocol:raise ValueError('Concurrent protocol mismatch')
            else:atomic_json(target,protocol)
    return protocol,rows


def snapshot_report(root,final=False,same_count=False):
    root=Path(root)
    if same_count:
        if final:raise ValueError('Matched status cannot publish completion')
        # Status must remain usable after a code update and without model files.
        protocol=json.loads((root/'protocol.json').read_text())
        rows=[json.loads(line) for line in (Path(protocol['prepared'])/'manifest.jsonl').read_text().splitlines() if line.strip()]
        if protocol['dataset']=='ruler':
            wanted=protocol.get('selected_ids')
            limit=json.loads((root/'tranche.json').read_text())['limit_per_task']
            rows=[r for r in rows if r['id'] in wanted] if wanted else [r for r in rows if r['ordinal']<limit]
    else:
        protocol,rows=verify(root,hardware=False)
    cases=['nocache','router'] if protocol['kind']=='inference' else list(actions(protocol['actions']))
    matching=None
    if same_count:
        from runner.matched_status import committed_records,match_records
        committed=committed_records(root,cases,protocol_identity(protocol))
        committed,matching=match_records(committed,[dict(prompt_id=r['id'],subtask=r['subtask']) for r in rows])
        selected={c:{r['prompt_id']:r for r in rs} for c,rs in committed.items()}
        probe_ids={p.parent.name for p in (root/'records'/'probe').glob('*/validated.json')}
    expected=503 if protocol['dataset']=='longbench-v2' else len(protocol.get('selected_ids') or []) if protocol['kind']=='inference' else 2600
    counts={c:0 for c in cases};probes=0;paired_rows=[];by_case={c:[] for c in cases}
    for row in rows:
        records={c:selected[c].get(row['id']) if same_count else accepted(root,c,row,protocol) for c in cases}
        for c,r in records.items():
            if r:counts[c]+=1;by_case[c].append(r)
        if protocol['kind']=='collection':
            probes+=(row['id'] in probe_ids and row['id'] in selected['nocache']) if same_count else accepted(root,'probe',row,protocol) is not None
        elif records['router']:probes+=1
        if records['nocache']:
            base=records['nocache']
            for c,r in records.items():
                if r is None:continue
                overhead=r['timings']['routing_overhead_seconds'] if protocol['kind']=='inference' and c=='router' else 0.
                paired_rows.append(dict(sample_id=row['id'],task=row['subtask'],action=c,accuracy=r['accuracy'],baseline_accuracy=base['accuracy'],
                    answer_ttft_seconds=r['timings']['answer_engine_ttft_seconds'],
                    estimated_router_ttft_seconds=r['timings']['answer_engine_ttft_seconds']+overhead,
                    baseline_ttft_seconds=base['timings']['answer_engine_ttft_seconds']))
    tranche_complete=all(v==len(rows) for v in counts.values()) and probes==len(rows)
    if final and not tranche_complete:raise ValueError('Missing records; cannot publish completion')
    if final:
        cleanup=json.loads((root/'cleanup.json').read_text())
        if not cleanup['owned_engines_exited']:raise ValueError('Owned engines must exit before final publication')
        for p in (root/'processes').glob('*.json'):
            state=json.loads(p.read_text())
            if alive(state) or group_alive(state['pid']):raise ValueError('Owned engine remains live')
    report=dict(planned_samples=expected,scheduled_samples=len(rows),accepted_answers=counts,accepted_probes=probes,
        answers=sum(counts.values()),tranche_complete=tranche_complete,complete=final and tranche_complete and len(rows)==expected,
        status='complete' if final and tranche_complete and len(rows)==expected else 'partial-complete' if final else 'partial',
        hardware=protocol['hardware'],kind=protocol['kind'],dataset=protocol['dataset'])
    if matching is not None:report['matching']=matching
    from runner.corpus_eval import summarize
    report['methods']={c:summarize([r for r in paired_rows if r['action']==c]) for c in cases if any(r['action']==c for r in paired_rows)}
    if protocol['kind']=='inference':
        from collections import Counter
        report['router_action_frequencies']=dict(Counter(r['decision']['action'] for r in by_case['router']))
        report['fresh_sparse_controls']=sum(r['decision']['action']!='nocache' for r in by_case['router'])
        report['fresh_answer_generations']=report['answers']+report['fresh_sparse_controls']
        report.update(timing='Measured live router TTFT includes probe/export/features/retirement/decision/sync/first answer token',
                      cross_dataset=protocol['dataset']=='longbench-v2',training_hardware=load(root/'tree.json')['provenance']['hardware'],refit=False)
        for metrics in report['methods'].values():
            metrics['timing']=report['timing']
            for aggregate in [metrics['overall'],*metrics['tasks'].values()]:
                aggregate['live_total_ttft_seconds']=aggregate.pop('estimated_router_ttft_seconds')
                aggregate['live_speedup']=aggregate.pop('estimated_speedup')
    else:
        for metrics in report['methods'].values():
            metrics['timing']='Measured fixed-action answer TTFT; independent probes reported separately'
            for aggregate in [metrics['overall'],*metrics['tasks'].values()]:
                aggregate['fixed_action_ttft_seconds']=aggregate.pop('estimated_router_ttft_seconds')
                aggregate['fixed_action_speedup']=aggregate.pop('estimated_speedup')
    for r in paired_rows:
        key='live_total_ttft_seconds' if protocol['kind']=='inference' else 'fixed_action_ttft_seconds'
        r[key]=r.pop('estimated_router_ttft_seconds')
    report['output_statistics']={c:dict(output_caps=sum(r.get('output_cap_reached',False) for r in rs),
        mean_thinking_tokens=sum(r.get('thinking_tokens',0) for r in rs)/len(rs),
        mean_answer_tokens=sum(r.get('answer_tokens',0) for r in rs)/len(rs),
        mean_control_tokens=sum(r.get('control_tokens',0) for r in rs)/len(rs)) for c,rs in by_case.items() if rs}
    stem='report_same_count' if same_count else 'report'
    atomic_json(root/(stem+'.json'),report)
    (root/(stem+'.txt')).write_text(json.dumps(report,indent=2)+'\n')
    import html
    (root/(stem+'.html')).write_text('<meta charset="utf-8"><pre>'+html.escape(json.dumps(report,indent=2))+'</pre>')
    if paired_rows or same_count:
        with (root/(stem+'.csv')).open('w') as handle:
            fields=list(paired_rows[0]) if paired_rows else ['sample_id','task','action','accuracy']
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(paired_rows)
    if final:
        atomic_json(root/('final-validation.json' if report['complete'] else f'partial-{len(rows)}-validation.json'),report)
    return report


def detach(root,resume=False):
    root=Path(root)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle(root)
        if (root/'final-validation.json').exists():raise ValueError('Completed inventory cannot be restarted')
        if not resume and (root/'supervisor.json').exists():raise ValueError('Use resume to collect missing records')
        verify(root)
        command=['nohup',sys.executable,'-u',str(Path(__file__).resolve()),'supervise','--root',str(root)]
        helper="""import json,subprocess,sys
c=json.load(sys.stdin)
with open(c['log'],'a') as log:
 p=subprocess.Popen(c['command'],cwd=c['cwd'],env=c['env'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid,flush=True)
"""
        result=subprocess.run([sys.executable,'-c',helper],input=json.dumps(dict(command=command,cwd=str(ROOT),env=environment(),log=str(root/'supervisor.log'))),text=True,capture_output=True,check=True)
        pid=int(result.stdout.strip())
        for _ in range(100):
            state_path=root/'supervisor.json'
            if state_path.exists() and json.loads(state_path.read_text()).get('pid')==pid:break
            if identity(pid) is None:raise RuntimeError('Supervisor exited; inspect log')
            time.sleep(.1)
        else:raise RuntimeError('Supervisor startup timed out')
        proc=Path('/proc')/str(pid);stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        status=(proc/'status').read_text();ignored=int(next(s.split()[1] for s in status.splitlines() if s.startswith('SigIgn:')),16)
        if (int(stat[1])!=1 or int(stat[3])!=pid or not ignored&1 or os.readlink(proc/'fd/0')!='/dev/null' or
                b'CUDA_VISIBLE_DEVICES=' not in (proc/'environ').read_bytes().split(b'\0')):raise RuntimeError('Detachment verification failed')
        atomic_json(root/'detachment.json',dict(pid=pid,identity=identity(pid),parent_pid=1,session_id=pid,sighup_ignored=True,stdin='/dev/null',cpu_only=True))
        print(f'Detached CPU supervisor {pid}',flush=True)


def supervise(root):
    root=Path(root);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',started_at=time.time())
        atomic_json(root/'supervisor.json',state);running={};session=uuid.uuid4().hex
        def launch(group,phase,retry):
            attempt=f'{session}-{retry}'
            cmd=[sys.executable,'-u','-m','runner.corpus_collect','--root',str(root),'--group',str(group),'--phase',phase,'--attempt',attempt]
            with (root/f'{phase}-group{group}-{attempt}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,env=environment(','.join(protocol['groups'][group])),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            receipt=dict(pid=child.pid,identity=identity(child.pid),phase=phase,group=group,retry=retry,started_at=time.time(),command=cmd)
            atomic_json(root/'processes'/f'{phase}-group{group}-{attempt}.json',receipt)
            running[group]=(child,receipt,time.monotonic())
        try:
            protocol,rows=verify(root);cleanup_caches(root,protocol)
            for phase in ('baseline','cached'):
                state.update(phase=phase);atomic_json(root/'supervisor.json',state)
                cases=['nocache'] if phase=='baseline' else ['router'] if protocol['kind']=='inference' else ['probe']+[a for a in actions(protocol['actions']) if a!='nocache']
                for group in range(len(protocol['groups'])):
                    if any(accepted(root,c,r,protocol) is None for r in rows if r['ordinal']%len(protocol['groups'])==group for c in cases):launch(group,phase,0)
                while running:
                    for group,(child,receipt,last) in list(running.items()):
                        progress=root/f'progress-group{group}.json'
                        if progress.exists() and progress.stat().st_mtime>receipt.get('last_progress',receipt['started_at']):
                            receipt['last_progress']=progress.stat().st_mtime;last=time.monotonic();running[group]=(child,receipt,last)
                        code=child.poll()
                        if code is None and time.monotonic()-last>900:terminate_owned(receipt);child.wait(timeout=10);code=75
                        if code is None:continue
                        if group_alive(child.pid):terminate_owned(receipt)
                        running.pop(group);cleanup_caches(root,protocol)
                        if code:
                            if code==75 and receipt['retry']==0:launch(group,phase,1)
                            else:raise RuntimeError(f'{phase} group{group} failed ({code}); validation failures are not retried')
                    if running:time.sleep(2)
            atomic_json(root/'cleanup.json',dict(owned_engines_exited=True,complete=True,at=time.time()))
            report=snapshot_report(root,final=True)
            state.update(state=report['status'],finished_at=time.time());atomic_json(root/'supervisor.json',state)
        except BaseException as error:
            for child,receipt,_ in running.values():terminate_owned(receipt);child.wait(timeout=10)
            state.update(state='failed',error=str(error));atomic_json(root/'supervisor.json',state);raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('configure','prepare','verify','detach','resume','status','status_same_count','report','supervise','train','replay','test','snapshot','relocate'))
    p.add_argument('--root',type=Path,required=True)
    for name in ('model','prepared','cache-root','ruler','data','tree','corpus','actions','output','decisions','evaluation-snapshot'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--dataset',choices=('ruler','longbench-v2'),default='ruler')
    p.add_argument('--gpu-a');p.add_argument('--gpu-b');p.add_argument('--limit-per-task',type=int,default=None,help='1..200; configure defaults to 200, later commands retain the frozen tranche')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--train-samples',type=int)
    p.add_argument('--trainer',default='ruler13-v1');p.add_argument('--seed',type=int,default=42);p.add_argument('--policy-count',type=int,default=3)
    p.add_argument('--depths',type=int,nargs='+',default=None,help='Train: maximum depths to search (1..32); default 1 2 3')
    p.add_argument('--accuracy-weight',type=float,default=None,help='Train: nonnegative loss weight; 0 favors TTFT, larger favors accuracy; omitted retains legacy selection')
    p.add_argument('--action-scope',choices=('original','all'),default='all',help='Offline readers: original inventory or include the completed 5/10 extension')
    p.add_argument('--evaluation',choices=('heldout','training'),default='heldout',help='Train/test: held-out split, or fit and evaluate on exactly the same complete samples')
    p.add_argument('--skip-validation',action='store_true',help='Offline only: trust saved features/results; skip artifact hashing and attention replay')
    p.add_argument('--mode',choices=('offline',),default='offline')
    p.add_argument('--inference',action='store_true',help=argparse.SUPPRESS)
    return p


def main():
    a=parser().parse_args()
    if a.command in ('train','test','replay','snapshot'):
        from runner.corpus_progress import session
        mode='dataset validation SKIPPED; reading saved features/results' if a.skip_validation else 'opening and validating corpus'
        with session('Offline '+a.command, detail=mode+' '+str(a.root), output=a.output):
            return execute(a)
    return execute(a)


def execute(a):
    if a.skip_validation and a.command not in ('train','test','replay','snapshot'):
        raise ValueError('--skip-validation is for offline commands only')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.resolve())
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):raise ValueError('Coordinator must be CPU-only')
    if a.command!='configure' and a.limit_per_task is not None:
        if a.command not in ('prepare','detach','resume'):raise ValueError('--limit-per-task applies to configure/prepare/detach/resume')
        if not 1<=a.limit_per_task<=200:raise ValueError('limit-per-task must be 1..200')
        with (a.root/'launch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle(a.root)
            target=a.root/'tranche.json';prior=json.loads(target.read_text())
            if a.limit_per_task<prior['limit_per_task'] and (a.root/'records').exists():raise ValueError('Cannot shrink a collected tranche')
            atomic_json(target,dict(limit_per_task=a.limit_per_task))
    if a.command=='configure':
        if a.limit_per_task is None:a.limit_per_task=200
        result=configure(a)
    elif a.command in ('train','replay','test','snapshot'):
        from runner.corpus_training_data import open_corpus
        requested_tree=load(a.tree) if a.command in ('test','snapshot') and a.tree else None
        corpus=open_corpus(a.root,a.prepared,tree=requested_tree,action_scope=a.action_scope,skip_validation=a.skip_validation)
        if corpus.protocol['kind']!='collection' or corpus.protocol['dataset']!='ruler':raise ValueError('Training/offline replay only accepts RULER corpus collections')
        if a.command=='train':
            from runner.corpus_train import train
            result=train(corpus,a.train_samples,a.output,a.seed,a.trainer,a.policy_count,
                         action_scope=a.action_scope,evaluation=a.evaluation,
                         depths=a.depths,accuracy_weight=a.accuracy_weight)
        elif a.command=='replay':
            from runner.corpus_eval import replay
            result=replay(corpus,[json.loads(line) for line in a.decisions.read_text().splitlines()],a.output)
        elif a.command=='test':
            from runner.corpus_eval import test_tree
            snapshot=json.loads(a.evaluation_snapshot.read_text()) if a.evaluation_snapshot else None
            result=test_tree(corpus,load(a.tree),a.output,snapshot,evaluation=a.evaluation)
        else:
            result=corpus.snapshot()
            if a.tree:
                tree=load(a.tree)
                if tree['provenance']['corpus_protocol_sha256']!=protocol_identity(corpus.protocol):raise ValueError('Tree belongs to another corpus')
                result['samples']=[r for r in result['samples'] if r['id'] not in tree['training_ids']]
                kept={r['id'] for r in result['samples']}
                result['acceptance_hashes']={k:v for k,v in result['acceptance_hashes'].items() if k.split('/')[2] in kept}
                result['excluded_training_tree']=tree['payload_sha256']
                result['counts']['complete']=len(kept)
            if a.output.exists():raise FileExistsError('Snapshot output exists')
            atomic_json(a.output,result)
    elif a.command=='prepare':
        s=json.loads((a.root/'settings.json').read_text())
        if s['dataset']=='ruler':
            limit=json.loads((a.root/'tranche.json').read_text())['limit_per_task']
            prepare(Path(s['ruler']),Path(s['model']),Path(s['prepared']),a.workers,limit)
        else:
            from scripts.longbench_v2 import prepare as lb_prepare
            from types import SimpleNamespace
            lb_prepare(SimpleNamespace(model=Path(s['model']),data=Path(s['data']),output=Path(s['prepared'])))
        result=dict(prepared=True)
    elif a.command=='verify':result=dict(protocol=verify(a.root)[0],gpu_inference_launched=False)
    elif a.command in ('detach','resume'):detach(a.root,a.command=='resume');return
    elif a.command=='supervise':supervise(a.root);return
    elif a.command=='report':idle(a.root);result=snapshot_report(a.root,final=True)
    elif a.command=='relocate':
        idle(a.root)
        for filename in ('settings.json','protocol.json'):
            path=a.root/filename;s=json.loads(path.read_text());before=protocol_identity(s)
            for key in ('model','prepared','cache_root'):
                if getattr(a,key) is not None:s[key]=str(getattr(a,key))
            if protocol_identity(s)!=before:raise ValueError('Relocation changed identity')
            atomic_json(path,s)
        result=dict(relocated=True,verify_required=True)
    else:
        from scripts.corpus_status import write_summary
        if a.command=='status_same_count':snapshot_report(a.root,same_count=True)
        write_summary(a.root,same_count=a.command=='status_same_count')
        return
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
