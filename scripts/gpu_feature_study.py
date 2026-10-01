#!/usr/bin/env python3
"""Detached GPU collection -> CPU feature search -> locked live comparison."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runner.gpu_study import prepare,verify,extract,collect,accepted,report,sealed,progress,read
from runner.router_process import identity,alive,group_alive
from runner.setups import atomic_json,file_hash
from scripts.corpus_control import idle,check_hardware
from scripts.router_control import environment,terminate_owned,cleanup_caches


def run_phase(root,phase,protocol,state):
    for retry in range(2):
        failures=sum(read(p).get('exit_code')==75 for p in (root/'processes').glob('*-exit.json'))
        if retry and failures>1:raise RuntimeError('Study operational retry allowance exhausted')
        check_hardware(protocol)  # Includes busy-device gate immediately before every GPU phase/retry.
        attempt=uuid.uuid4().hex
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(root),'--phase',phase,'--attempt',attempt]
        with (root/f'{phase}-{attempt}.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=environment(','.join(protocol['groups'][0])),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(pid=child.pid,identity=identity(child.pid),phase=phase,attempt=attempt,retry=retry,started_at=time.time(),command=command)
        atomic_json(root/'processes'/f'{attempt}.json',receipt);state.update(state=phase,worker=receipt);atomic_json(root/'supervisor.json',state)
        last=time.monotonic();mtime=(root/'progress.json').stat().st_mtime if (root/'progress.json').exists() else 0
        try:
            while child.poll() is None:
                p=root/'progress.json'
                if p.exists() and p.stat().st_mtime>mtime:mtime=p.stat().st_mtime;last=time.monotonic()
                if time.monotonic()-last>900:
                    terminate_owned(receipt);child.wait(timeout=10);code=75;break
                time.sleep(2)
            else:code=child.returncode
        except BaseException:
            terminate_owned(receipt);child.wait(timeout=10);raise
        if group_alive(child.pid):terminate_owned(receipt)
        if group_alive(child.pid):raise RuntimeError('Owned engines failed to exit')
        cleanup_caches(root,protocol)
        atomic_json(root/'processes'/f'{attempt}-exit.json',dict(**receipt,exit_code=code,owned_engines_exited=True))
        if code==0:
            sealed(root/f'{phase}-engine-exit.json',dict(owned_engines_exited=True,phase=phase))
            return
        if code!=75 or retry:raise RuntimeError(f'{phase} failed ({code}); validation failures are not retried')


def cpu_search(root):
    from runner.gpu_study_search import search
    protocol,corpus,matrix=verify(root)
    if not sealed(root/'collect-engine-exit.json')['owned_engines_exited']:raise ValueError('GPU collection has not exited')
    for pid in matrix['ids']:
        record=accepted(root,'diagnostic',pid)
        if record is None:raise ValueError('Diagnostic collection incomplete')
        target=root/'features'/f'{pid}.json'
        sealed(target,dict(features=record['features'],input_sha256=corpus.rows[pid]['sha256']))
    search(root,corpus,matrix,sealed(root/'baseline.json'),sealed,lambda msg:progress(root,'search',msg))


def supervise(root):
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),state='verifying',started_at=time.time());atomic_json(root/'supervisor.json',state)
        try:
            protocol,corpus,matrix=verify(root,full=True);cleanup_caches(root,protocol)
            if not (root/'extraction-complete.json').exists():extract(root)
            if not all(accepted(root,'diagnostic',p) is not None for p in matrix['ids']):run_phase(root,'collect',protocol,state)
            elif not (root/'collect-engine-exit.json').exists():
                if any(group_alive(read(p)['pid']) for p in (root/'processes').glob('*.json')):raise ValueError('Owned engine group remains')
                sealed(root/'collect-engine-exit.json',dict(owned_engines_exited=True,phase='collect'))
            state.update(state='search');atomic_json(root/'supervisor.json',state)
            cpu_search(root)
            if not all(accepted(root,c,p) is not None for p in matrix['ids'] for c in ('current','winner')):run_phase(root,'live',protocol,state)
            if any(group_alive(read(p)['pid']) for p in (root/'processes').glob('*.json')):raise ValueError('Owned engines remain before final reporting')
            report(root);state.update(state='complete',finished_at=time.time());atomic_json(root/'supervisor.json',state)
        except BaseException as e:
            state.update(state='failed',error=str(e),finished_at=time.time());atomic_json(root/'supervisor.json',state);raise


def detach(root,resume=False):
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle(root)
        if (root/'complete.json').exists():raise ValueError('Study already complete')
        if (root/'supervisor.json').exists() and not resume:raise ValueError('Use resume for missing records')
        protocol=read(root/'protocol.json');check_hardware(protocol)
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
            if (root/'supervisor.json').exists() and read(root/'supervisor.json')['pid']==pid:break
            if identity(pid) is None:raise RuntimeError('Supervisor exited; inspect log')
            time.sleep(.1)
        else:raise RuntimeError('Supervisor startup timeout')
        proc=Path('/proc')/str(pid);stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        ignored=int(next(s.split()[1] for s in (proc/'status').read_text().splitlines() if s.startswith('SigIgn:')),16)
        if int(stat[1])!=1 or int(stat[3])!=pid or not ignored&1 or os.readlink(proc/'fd/0')!='/dev/null' or b'CUDA_VISIBLE_DEVICES=' not in (proc/'environ').read_bytes().split(b'\0'):
            raise RuntimeError('Detached ownership verification failed')
        receipt=dict(pid=pid,identity=identity(pid),parent_pid=1,session_id=pid,sighup_ignored=True,stdin='/dev/null',cpu_only=True,resume=resume)
        atomic_json(root/'detachment.json',receipt);print(json.dumps(receipt))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','verify','extract','search','live-validation','report','detach','resume','status','supervise','worker'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--extension',type=Path);p.add_argument('--source-worktree',type=Path,default=ROOT)
    p.add_argument('--phase',choices=['collect','live']);p.add_argument('--attempt');a=p.parse_args();root=a.root.resolve()
    if a.stage!='worker' and os.environ.get('CUDA_VISIBLE_DEVICES',''):raise ValueError('Control process must be CPU-only')
    if a.stage=='prepare':prepare(root,a.extension.resolve(),a.source_worktree.resolve())
    elif a.stage=='verify':verify(root,full=True);print('Frozen sources, runtime and environment verified')
    elif a.stage=='extract':idle(root);extract(root)
    elif a.stage=='search':idle(root);cpu_search(root)
    elif a.stage in ('live-validation','report'):idle(root);report(root)
    elif a.stage in ('detach','resume'):detach(root,a.stage=='resume')
    elif a.stage=='supervise':supervise(root)
    elif a.stage=='worker':
        try:collect(root,a.phase,a.attempt)
        except (TimeoutError,ConnectionError):
            import traceback;traceback.print_exc();raise SystemExit(75)
        except Exception:
            import traceback;traceback.print_exc();raise SystemExit(76)
    else:
        result=dict(counts={c:len(list((root/'records'/c).glob('*/validated.json'))) for c in ('diagnostic','current','winner')})
        for n in ('supervisor','progress','detachment','startup-verification','live-startup-verification','complete'):
            path=root/(n+'.json')
            if path.exists():result[n]=sealed(path) if n in ('startup-verification','live-startup-verification','complete') else read(path)
        if 'supervisor' in result:result['supervisor']['alive']=alive(result['supervisor'])
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
