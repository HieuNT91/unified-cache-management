#!/usr/bin/env python3
"""Prepare, detach, resume and inspect a linked action extension."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
# Match the original collector's BLAS reduction order before importing NumPy.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runner.corpus_extension import LinkedCorpus,prepare
from runner.corpus_records import record_dir,publish
from runner.setups import atomic_json,file_hash,check_environment
from runner.router_process import identity,alive,group_alive
from scripts.corpus_control import idle,check_hardware
from scripts.router_control import code_hashes,environment,terminate_owned,cleanup_caches


def verify(root,hardware=False):
    corpus=LinkedCorpus(root)
    if corpus.protocol['code']!=code_hashes():raise ValueError('Frozen extension code changed')
    corpus.verify_sources()
    from importlib.metadata import version
    from runner.config import VERSIONS
    from scripts.corpus_inputs import spec
    from run import check_model
    check_model(Path(corpus.protocol['model']))
    if any(version(name).split('+')[0]!=expected for name,expected in VERSIONS.items()):
        raise ValueError('Pinned runtime environment changed')
    for name,digest in spec()['model_fingerprints'].items():
        if file_hash(Path(corpus.protocol['model'])/name)!=digest:raise ValueError('Pinned model/tokenizer changed')
    if hardware:check_hardware(corpus.protocol)
    return corpus


def collect(root,attempt):
    from runner.corpus_runtime import Engine
    corpus=verify(root);protocol=corpus.protocol;pending=corpus.pending()
    if not pending:return
    if check_environment(4)!=protocol['groups'][0]:raise ValueError('GPU UUIDs changed')
    with (root/'group0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine=Engine(root,protocol,0,'cached',attempt,pending)
        try:
            for row in pending:
                pid=row['id'];attention=corpus.attention(pid,replay=True);provenance=corpus.provenance(pid)
                engine.begin(row);staged=[]
                for action in protocol['added_actions']:
                    if corpus.outcome(pid,action) is not None:continue
                    folder=record_dir(root,action,pid)
                    if folder.exists():
                        history=root/'incomplete'/f'{action}-{pid}-{uuid.uuid4().hex}'
                        history.parent.mkdir(exist_ok=True);folder.rename(history)
                    folder.mkdir(parents=True)
                    record,ds=engine.answer(corpus.actions[action],action,attention)
                    record['derived_mask_provenance']=provenance;staged.append((action,record,ds))
                deletion=engine.end()
                for action,record,ds in staged:
                    record['cache_deletion']=deletion;publish(root,action,row,protocol,record,ds)
                    corpus.outcome(pid,action)
                    atomic_json(root/'progress-group0.json',dict(prompt_id=pid,case=action,accepted_at=time.time()))
                    print(f'{action} {pid}: accepted',flush=True)
                if pid==protocol['prompt_ids'][0] and not (root/'startup-verification.json').exists():
                    initial=json.loads((engine.session/'initialization.json').read_text())
                    atomic_json(root/'startup-verification.json',dict(prompt_id=pid,actions=protocol['added_actions'],
                        complete=True,answers=[corpus.outcome(pid,a)['accuracy'] for a in protocol['added_actions']],
                        all_rank_scores_masks=True,all64_selection=True,original_attention_replayed=True,cache_immutable=True,
                        retirement_validated=True,cache_deletion=deletion,warmup_shards=initial['warmup_readiness']['verified_shards'],
                        yarn_normalized=True,no_new_independent_probe=True,at=time.time()))
        finally:engine.close()


def finalize(root):
    from runner.extension_train import train
    corpus=verify(root)
    for p in (root/'processes').glob('*.json'):
        state=json.loads(p.read_text())
        if alive(state) or group_alive(state['pid']):raise ValueError('Owned engines remain alive')
    cleanup=json.loads((root/'cleanup.json').read_text())
    if not cleanup['owned_engines_exited']:raise ValueError('Missing engine exit gate')
    matrix=corpus.matrix()
    target=root/'matrix.json'
    if target.exists():
        if json.loads(target.read_text())!=matrix:raise ValueError('Pinned matrix changed')
    else:atomic_json(target,matrix)
    train(root/'training',matrix,corpus.protocol['previous_tree'],file_hash(target))
    atomic_json(root/'complete.json',dict(complete=True,answers=len(corpus.rows)*len(corpus.protocol['added_actions']),prompts=len(corpus.rows),reused_probes=len(corpus.rows),
        owned_engines_exited=True,matrix_sha256=file_hash(target),training_complete_sha256=file_hash(root/'training/complete.json')))


def supervise(root):
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),state='verifying',started_at=time.time())
        atomic_json(root/'supervisor.json',state);child=None;receipt=None
        try:
            corpus=verify(root,hardware=True);cleanup_caches(root,corpus.protocol)
            for retry in range(2):
                if not corpus.pending():break
                attempt=uuid.uuid4().hex
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(root),'--attempt',attempt]
                with (root/f'worker-{attempt}.log').open('a') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,env=environment(','.join(corpus.protocol['groups'][0])),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                receipt=dict(pid=child.pid,identity=identity(child.pid),retry=retry,command=cmd,started_at=time.time())
                atomic_json(root/'processes'/f'{attempt}.json',receipt)
                state.update(state='collecting',worker=receipt);atomic_json(root/'supervisor.json',state)
                last=time.monotonic();mtime=time.time();code=None
                while code is None:
                    progress=root/'progress-group0.json'
                    if progress.exists() and progress.stat().st_mtime>mtime:
                        mtime=progress.stat().st_mtime;last=time.monotonic()
                    code=child.poll()
                    if code is None and time.monotonic()-last>900:
                        terminate_owned(receipt);child.wait(timeout=10);code=75
                    if code is None:time.sleep(2)
                if group_alive(child.pid):terminate_owned(receipt)
                if group_alive(child.pid):raise RuntimeError('Owned engines remain alive')
                cleanup_caches(root,corpus.protocol)
                atomic_json(root/'processes'/f'{attempt}-exit.json',dict(**receipt,exit_code=code,owned_engines_exited=True))
                child=None
                if code==0:break
                if code!=75 or retry:raise RuntimeError(f'Collection failed ({code}); validation mismatches are not retried')
            atomic_json(root/'cleanup.json',dict(owned_engines_exited=True,at=time.time()))
            state.update(state='training');atomic_json(root/'supervisor.json',state)
            finalize(root)
            state.update(state='complete',finished_at=time.time());atomic_json(root/'supervisor.json',state)
        except BaseException as error:
            if child is not None:terminate_owned(receipt);child.wait(timeout=10)
            state.update(state='failed',error=str(error),finished_at=time.time());atomic_json(root/'supervisor.json',state);raise


def detach(root,resume=False):
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle(root)
        if (root/'complete.json').exists():raise ValueError('Completed extension cannot restart')
        if (root/'supervisor.json').exists() and not resume:raise ValueError('Use resume for missing extension records')
        # Full source verification runs inside the detached supervisor before GPU launch.
        corpus=LinkedCorpus(root);check_hardware(corpus.protocol)
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
            if (root/'supervisor.json').exists() and json.loads((root/'supervisor.json').read_text())['pid']==pid:break
            if identity(pid) is None:raise RuntimeError('Supervisor exited; inspect log')
            time.sleep(.1)
        else:raise RuntimeError('Supervisor startup timed out')
        proc=Path('/proc')/str(pid);stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        ignored=int(next(s.split()[1] for s in (proc/'status').read_text().splitlines() if s.startswith('SigIgn:')),16)
        if int(stat[1])!=1 or int(stat[3])!=pid or not ignored&1 or os.readlink(proc/'fd/0')!='/dev/null' or b'CUDA_VISIBLE_DEVICES=' not in (proc/'environ').read_bytes().split(b'\0'):
            raise RuntimeError('Detachment verification failed')
        receipt=dict(pid=pid,identity=identity(pid),parent_pid=1,session_id=pid,sighup_ignored=True,stdin='/dev/null',cpu_only=True,resume=resume)
        atomic_json(root/'detachment.json',receipt);print(json.dumps(receipt))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','detach','resume','supervise','worker','status','report'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--base-corpus',type=Path)
    p.add_argument('--added-actions',type=int,nargs='+',default=[5,10]);p.add_argument('--previous-tree',type=Path);p.add_argument('--attempt')
    a=p.parse_args();root=a.root.resolve()
    if a.command!='worker' and os.environ.get('CUDA_VISIBLE_DEVICES',''):raise ValueError('Control commands must be CPU-only')
    if a.command=='prepare':print(prepare(root,a.base_corpus,a.added_actions,a.previous_tree))
    elif a.command=='verify':verify(root);print('Sources and code verified')
    elif a.command in ('detach','resume'):detach(root,a.command=='resume')
    elif a.command=='supervise':supervise(root)
    elif a.command=='worker':
        try:collect(root,a.attempt)
        except (TimeoutError,ConnectionError):
            import traceback;traceback.print_exc();raise SystemExit(75)
        except Exception:
            import traceback;traceback.print_exc();raise SystemExit(76)
    elif a.command=='report':idle(root);finalize(root)
    else:
        c=LinkedCorpus(root);result=dict(answers={a:len(list((root/'records'/a).glob('*/validated.json'))) for a in c.protocol['added_actions']})
        for name in ('supervisor','progress-group0','startup-verification','complete'):
            if (root/(name+'.json')).exists():result[name]=json.loads((root/(name+'.json')).read_text())
        if 'supervisor' in result:result['supervisor']['alive']=alive(result['supervisor'])
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
