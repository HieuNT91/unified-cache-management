#!/usr/bin/env python3
"""Remote A800 configure/verify/detach/resume/status; no benchmark runtime dependency."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.setups import atomic_json,file_hash,fingerprint
from runner.router_policy import load_policy
from runner.router_process import identity,alive,group_alive
from runner.router_sweep import CASES,accepted
from scripts.router_inputs import COHORT,load_spec,verify_prepared,prepare


def groups(a,b):
    values=[a.split(','),b.split(',')]
    if any(len(g)!=4 for g in values) or len(set(sum(values,[])))!=8 or any(
            not x.startswith('GPU-') or len(x)!=40 for x in sum(values,[])):
        raise ValueError('Provide two disjoint groups of four full GPU UUIDs')
    return values


def code_hashes():
    paths=[ROOT/'run.py',ROOT/'run.sh']
    for base in ('runner','ucm','scripts'):
        paths.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json')
                     and '__pycache__' not in p.parts and 'vendor' not in p.parts)
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}


def configure(args):
    root=args.root
    settings=dict(model=str(args.model),prepared=str(args.prepared),cache_root=str(args.cache_root),
        policy=str(args.policy),ruler=str(args.ruler),groups=groups(args.gpu_a,args.gpu_b),
        cohort_sha256=file_hash(COHORT),code=code_hashes(),primary='router1',cases=list(CASES),
        answers=9100,router_probes=3900,watchdog_seconds=900,operational_retries=1)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'configure.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target=root/'settings.json'
        if target.exists():
            if json.loads(target.read_text())!=settings:raise ValueError('Existing router settings differ; use new result paths')
        else:
            if any(p.name!='configure.lock' for p in root.iterdir()):raise ValueError('Use a new empty router result directory')
            atomic_json(target,settings)
    print('Configured native router workflow; no inference launched.',flush=True)


def verify(root, publish=True):
    root=Path(root);settings=json.loads((root/'settings.json').read_text())
    if settings['code']!=code_hashes() or settings['cohort_sha256']!=file_hash(COHORT):
        raise ValueError('Configured code/cohort changed; review before using a new experiment directory')
    rows=verify_prepared(settings['prepared'])
    spec=load_spec()
    for name,expected in spec['model_fingerprints'].items():
        if file_hash(Path(settings['model'])/name)!=expected:raise ValueError('Model/tokenizer fingerprint mismatch')
    index=json.loads((Path(settings['model'])/'model.safetensors.index.json').read_text())
    if not all((Path(settings['model'])/name).is_file() for name in set(index['weight_map'].values())):
        raise ValueError('Missing local model shards')
    from importlib.metadata import version
    from runner.config import VERSIONS
    if any(version(name).split('+')[0]!=expected for name,expected in VERSIONS.items()):
        raise ValueError('Install the clean README pinned runtime versions')
    policy=load_policy(settings['policy'])
    if policy['provenance']['cohort_sha256']!=file_hash(COHORT) or policy['provenance']['training_protocol_sha256']!=spec['training_protocol_sha256']:
        raise ValueError('Policy was exported for a different frozen cohort')
    protocol={k:v for k,v in settings.items() if k not in ('policy','ruler')}
    protocol.update(policy_sha256=file_hash(settings['policy']),prepared_manifest_sha256=file_hash(Path(settings['prepared'])/'manifest.jsonl'),
        prepared_receipt_sha256=file_hash(Path(settings['prepared'])/'prepared.json'))
    if (root/'protocol.json').exists():
        if json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('Frozen execution protocol changed')
        if file_hash(root/'trees.json')!=protocol['policy_sha256']:raise ValueError('Installed policy changed')
        load_policy(root/'trees.json')
    elif publish:
        (root/'trees.json').write_bytes(Path(settings['policy']).read_bytes())
        (root/'trees.json.sha256').write_text(protocol['policy_sha256']+'  trees.json\n')
        atomic_json(root/'protocol.json',protocol)
    for case in CASES:
        for row in rows:accepted(root,case,row,protocol)
    return protocol,rows


def check_hardware(protocol):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    inventory={r[0].strip():r[1].strip() for r in csv.reader(raw.splitlines())}
    requested=sum(protocol['groups'],[])
    if any(u not in inventory or 'A800' not in inventory[u] for u in requested):
        raise ValueError('Every configured UUID must resolve to an A800; no local GPU qualification is performed')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if set(requested)&{r.strip() for r in busy}:raise ValueError('Selected A800 GPUs are occupied; existing processes were left untouched')


def idle(root):
    for path in [root/'supervisor.json',*(root/'processes').glob('*.json')]:
        if not path.exists():continue
        state=json.loads(path.read_text())
        if alive(state) or (path.parent.name=='processes' and group_alive(state['pid'])):
            raise ValueError('Owned supervisor/engine is still alive; refusing a duplicate launch')


def environment(devices=''):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=devices,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        ENABLE_SPARSE='TRUE',VLLM_USE_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ALLOW_INSECURE_SERIALIZATION='1',VLLM_ATTENTION_BACKEND='FLASH_ATTN',PLATFORM='cuda',
        VLLM_USE_REROPE='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')


def detach(root, resume=False):
    root=Path(root)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle(root)
        if (root/'final-validation.json').exists():raise ValueError('Completed scope cannot be restarted')
        if not resume and (root/'supervisor.json').exists():raise ValueError('Existing attempt; use resume for missing records')
        protocol,_=verify(root);check_hardware(protocol)
        command=['nohup',sys.executable,'-u',str(Path(__file__).resolve()),'supervise','--root',str(root)]
        # A short-lived CPU launcher exits before verification, so the supervisor
        # is genuinely reparented rather than depending on this CLI process.
        helper="""import json,subprocess,sys
config=json.load(sys.stdin)
with open(config['log'],'a') as log:
    child=subprocess.Popen(config['command'],cwd=config['cwd'],env=config['env'],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(child.pid,flush=True)
"""
        launched=subprocess.run([sys.executable,'-c',helper],input=json.dumps(dict(command=command,cwd=str(ROOT),
            env=environment(),log=str(root/'supervisor.log'))),capture_output=True,text=True,check=True)
        pid=int(launched.stdout.strip())
        for _ in range(100):
            if identity(pid) is None:raise RuntimeError('Supervisor exited; inspect supervisor.log')
            path=root/'supervisor.json'
            if path.exists() and json.loads(path.read_text()).get('pid')==pid:break
            time.sleep(.1)
        else:raise RuntimeError('Supervisor startup receipt timed out')
        state=json.loads(path.read_text())
        proc=Path('/proc')/str(pid)
        status=(proc/'status').read_text()
        env=(proc/'environ').read_bytes().split(b'\0')
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        ignored=int(next(s.split()[1] for s in status.splitlines() if s.startswith('SigIgn:')),16)
        if (int(stat[1])!=1 or int(stat[3])!=pid or not ignored&1 or os.readlink(proc/'fd/0')!='/dev/null'
                or b'CUDA_VISIBLE_DEVICES=' not in env):raise RuntimeError('Detachment verification failed')
        atomic_json(root/'detachment.json',dict(pid=pid,identity=identity(pid),session_id=int(stat[3]),
            parent_pid=int(stat[1]),sighup_ignored=True,stdin='/dev/null',cpu_only=True,resume=resume,at=time.time()))
        print(f'Detached CPU supervisor {pid}; both TP4 groups run only on the configured A800 UUIDs.',flush=True)


def terminate_owned(state):
    # A dead leader can leave ranks behind. Require its original session and
    # start time for every surviving member before terminating that owned group.
    pid=state['pid']
    def owned():
        members=[]
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields=path.read_text().rsplit(')',1)[1].split()
                if fields[0]!='Z' and int(fields[2])==pid:
                    if int(fields[3])!=pid or int(fields[19])<int(state['identity']['start']):
                        raise RuntimeError('Process-group identity changed; refusing intervention')
                    members.append(path.parent.name)
            except (FileNotFoundError,ProcessLookupError):
                pass
        return members
    if owned():
        os.killpg(pid,signal.SIGTERM)
        end=time.monotonic()+30
        while group_alive(pid) and time.monotonic()<end:time.sleep(.2)
        if owned():os.killpg(pid,signal.SIGKILL)
        end=time.monotonic()+10
        while group_alive(pid) and time.monotonic()<end:time.sleep(.1)
    if group_alive(pid):
        raise RuntimeError('Owned process group remains; automatic retry is blocked')


def cleanup_caches(root, protocol):
    """Remove only dead attempts' explicitly owned cache namespaces, never results."""
    import re
    import shutil
    anchor=(Path(protocol['cache_root'])/fingerprint(str(root))[:16]).resolve()
    removed=[]
    for path in (root/'sessions').glob('*/ownership.json'):
        owner=json.loads(path.read_text())
        if alive(owner) or group_alive(owner['pid']):
            continue
        cache=Path(owner['cache'])
        if (cache.is_symlink() or cache.parent.resolve()!=anchor/f"group{owner['group']}"
                or not re.fullmatch('[a-f0-9]{32}',cache.name)):
            raise ValueError('Cache ownership/path mismatch; no cleanup performed')
        if cache.exists():
            shutil.rmtree(cache)
            removed.append(str(cache))
            atomic_json(path.parent/'cache-cleanup.json',dict(owned_engines_exited=True,cache=str(cache),at=time.time()))
    return removed


def supervise(root):
    root=Path(root);signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),state='running',started_at=time.time())
        atomic_json(root/'supervisor.json',state)
        running={};session=uuid.uuid4().hex
        def launch(group,case,retry):
            attempt=f'{session}-{retry}'
            cmd=[sys.executable,'-u','-m','runner.router_sweep','--root',str(root),'--group',str(group),
                 '--case',case,'--attempt',attempt]
            log=(root/f'{case}-group{group}-{attempt}.log').open('a')
            child=subprocess.Popen(cmd,cwd=ROOT,env=environment(','.join(protocol['groups'][group])),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close()
            receipt=dict(pid=child.pid,identity=identity(child.pid),case=case,group=group,attempt=attempt,
                         started_at=time.time(),retry=retry,command=cmd)
            atomic_json(root/'processes'/f'{case}-group{group}-{attempt}.json',receipt)
            running[group]=(child,receipt,time.monotonic())
        try:
            protocol,rows=verify(root)
            cleanup_caches(root,protocol)
            for case in CASES:
                state.update(case=case);atomic_json(root/'supervisor.json',state)
                for group in (0,1):
                    if any(accepted(root,case,r,protocol) is None for r in rows if r['ordinal']%2==group):launch(group,case,0)
                while running:
                    for group,(child,receipt,last) in list(running.items()):
                        progress=root/f'progress-group{group}.json'
                        if progress.exists():
                            changed=progress.stat().st_mtime
                            if changed>receipt.get('last_progress',receipt['started_at']):
                                receipt['last_progress']=changed;last=time.monotonic();running[group]=(child,receipt,last)
                        code=child.poll()
                        timeout=code is None and time.monotonic()-last>900
                        if timeout:
                            terminate_owned(receipt);child.wait(timeout=10);code=75
                        if code is None:continue
                        if group_alive(child.pid):
                            terminate_owned(receipt)
                        if group_alive(child.pid):raise RuntimeError('Owned engines have not exited')
                        running.pop(group)
                        cleanup_caches(root,protocol)
                        if code:
                            if code==75 and receipt['retry']==0:
                                launch(group,case,1)
                            else:raise RuntimeError(f'{case} group{group} failed with exit {code}; validation failures are never retried')
                    if running:time.sleep(2)
            atomic_json(root/'cleanup.json',dict(complete=True,owned_engines_exited=True,at=time.time()))
            from runner.router_report import finalize
            finalize(root)
            state.update(state='complete',finished_at=time.time());atomic_json(root/'supervisor.json',state)
        except BaseException as error:
            for child,receipt,_ in running.values():
                terminate_owned(receipt);child.wait(timeout=10)
            state.update(state='failed',error=str(error),finished_at=time.time());atomic_json(root/'supervisor.json',state)
            raise


def status(root,same_count=False):
    result=dict(answers={},router_probes=0)
    for case in CASES:
        result['answers'][case]=len(list((root/'records'/case).glob('*/validated.json')))
    result['router_probes']=sum(result['answers'][c] for c in CASES if c.startswith('router'))
    for name in ('supervisor','progress-group0','progress-group1','final-validation'):
        path=root/(name+'.json')
        if path.exists():result[name]=json.loads(path.read_text())
    if 'supervisor' in result:result['supervisor']['alive']=alive(result['supervisor'])
    if same_count:
        from runner.matched_status import committed_records,match_records
        from runner.reporting import aggregate
        protocol=json.loads((root/'protocol.json').read_text())
        rows=[json.loads(line) for line in (Path(protocol['prepared'])/'manifest.jsonl').read_text().splitlines() if line.strip()]
        expected=[dict(prompt_id=r['id'],subtask=r['subtask']) for r in rows]
        records,matching=match_records(committed_records(root,CASES,fingerprint(protocol)),expected)
        result['matching']=matching
        result['available_answers']=matching['available_counts']
        result['answers']={c:len(rs) for c,rs in records.items()}
        result['router_probes']=sum(result['answers'][c] for c in CASES if c.startswith('router'))
        result['methods']={c:aggregate(rs,expected,dict(method=c),'running') for c,rs in records.items()}
        atomic_json(root/'status_same_count.json',result)
    print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('configure','prepare','verify','detach','resume','status','status_same_count','supervise','report'))
    p.add_argument('--root',type=Path,required=True)
    for name in ('model','prepared','policy','cache-root','ruler'):p.add_argument('--'+name,type=Path)
    p.add_argument('--gpu-a');p.add_argument('--gpu-b');p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.resolve())
    if os.environ.get('CUDA_VISIBLE_DEVICES',''):raise ValueError('Coordinator commands must be CPU-only')
    if a.command=='configure':configure(a)
    elif a.command=='prepare':
        s=json.loads((a.root/'settings.json').read_text());prepare(Path(s['ruler']),Path(s['model']),Path(s['prepared']),a.workers)
    elif a.command=='verify':verify(a.root);print('Verified 1300 frozen inputs, three policies and native code; no inference launched.')
    elif a.command in ('detach','resume'):detach(a.root,a.command=='resume')
    elif a.command=='supervise':supervise(a.root)
    elif a.command in ('status','status_same_count'):status(a.root,a.command=='status_same_count')
    else:
        idle(a.root)
        from runner.router_report import finalize
        finalize(a.root)


if __name__=='__main__':main()
