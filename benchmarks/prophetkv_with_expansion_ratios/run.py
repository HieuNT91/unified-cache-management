#!/usr/bin/env python3
"""Locked persistent-engine expansion comparison; no separate model qualification."""
import argparse
from contextlib import contextmanager
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from prophetkv_gpu import REPO, ROOT, PYTHON, GPU_UUIDS, check, processes, identity
from prophetkv_common import sha, dump
from cacheblend_prophetkv import CASES
import validation as val

from settings import PARENT, PARAMETERS
OLD=PARENT
SOURCE=Path(__file__).resolve().parent
CACHE=Path('/tmp/ucm-prophetkv-with-expansion-ratios-20260923')
STOP=threading.Event(); ACTIVE={}; MUTEX=threading.RLock()


def hashes(root):
    return {str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*'))
            if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}

@contextmanager
def lock():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def cpu_env():
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(CUDA_VISIBLE_DEVICES='',PROPHETKV_REPO=str(REPO),OMP_NUM_THREADS='4',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0')
    return env


def environment(gpu,receipt):
    check(gpu)
    env=cpu_env();env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu],SELECTOR_PHYSICAL_GPU=str(gpu),
        SELECTOR_UCM_ROOT=str(ROOT/'private_ucm'),PROPHETKV_SCHEDULER_RECEIPT=str(receipt),
        CUDA_HOME=str(Path('/home/thnguyen/unified-cache-management/.tools/cuda-12.6')),LD_LIBRARY_PATH=str(Path('/home/thnguyen/unified-cache-management/.tools/cuda-12.6/lib64')),
        ENABLE_SPARSE='TRUE',CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_ALLOW_INSECURE_SERIALIZATION='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'])
    return env



def group_alive(pid):
    for f in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=f.read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pid and fields[0]!='Z':return True
        except (OSError,ValueError,IndexError):pass
    return False


def terminate(child):
    if group_alive(child.pid):
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        for _ in range(50):
            if not group_alive(child.pid):break
            time.sleep(.1)
        if group_alive(child.pid):
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    child.wait(timeout=10)
    for _ in range(50):
        if not group_alive(child.pid):return
        time.sleep(.1)
    raise RuntimeError('Owned engine group remains live')


def stop(*_):STOP.set()


def idle(gpu):
    while processes(gpu):
        if STOP.wait(2):raise InterruptedError('Stopped')
    check(gpu)


def accept(entry,case,log_text,smoke):
    path=Path(entry['output'])
    if val.valid(path,entry,case,smoke):return True
    if not path.exists():return False
    record=json.loads(path.read_text());rid=record['request_id']
    begin='REQUEST_BEGIN '+rid;end='REQUEST_COMPLETE '+rid+' '+str(path)
    if begin not in log_text or end not in log_text:return False
    excerpt=begin+log_text.split(begin,1)[1].split(end,1)[0]+end+'\n'
    lp=path.with_suffix('.log')
    if lp.exists() and lp.read_text()!=excerpt:raise ValueError('Immutable excerpt changed')
    if not lp.exists():lp.write_text(excerpt)
    val.validate(path,entry,case,smoke)
    dump(path.with_suffix('.validated.json'),dict(record_sha256=sha(path),log_sha256=sha(lp),
        diagnostics_sha256=sha(path.with_suffix('.diagnostics.json')),accepted_at=time.time()))
    return True


def session(p,gpu,case,entries,phase=0,smoke=False,fail_after=None):
    if smoke:raise ValueError('Smoke is disabled')
    for attempt in range(2):
        remaining=[e for e in entries if not val.valid(Path(e['output']),e,case,smoke)]
        if not remaining:return
        if STOP.is_set():raise InterruptedError('Stopped')
        idle(gpu)
        for entry in remaining:
            path=Path(entry['output'])
            if path.exists():
                archive=path.parent/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
                for f in path.parent.glob(path.stem+'.*'):f.rename(archive/f.name)
        session_id=uuid.uuid4().hex
        directory=ROOT/'sessions';directory.mkdir(exist_ok=True)
        manifest=directory/(session_id+'.manifest.json');dump(manifest,remaining)
        metadata=directory/(session_id+'.json');log_path=directory/(session_id+'.log')
        cache=CACHE/f'gpu-{gpu}'
        if cache.exists():shutil.rmtree(cache) # previous owned process is confirmed exited
        cmd=[str(PYTHON),'-u',str(ROOT/'source/worker.py'),'--protocol',str(ROOT/'protocol.json'),
            '--manifest',str(manifest),'--case',case,'--cache-dir',str(cache),'--session',str(metadata),
            '--phase',str(phase),'--engine-start-count',str(attempt+1)]
        if smoke:cmd.append('--smoke')
        env=environment(gpu,directory/(session_id+'.scheduler.json'))
        if fail_after and attempt==0:env['PROPHETKV_FAIL_AFTER']=str(fail_after)
        child=None
        accepted=set()
        dump(directory/(session_id+'.launch.json'),dict(session_id=session_id,case=case,gpu=gpu,phase=phase,attempt=attempt+1,started_at=time.time(),manifest=str(manifest)))
        with log_path.open('w') as log:
            try:
                child=subprocess.Popen(cmd,cwd='/tmp',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with MUTEX:ACTIVE[gpu]=child
                dump(ROOT/f'gpu-{gpu}.json',dict(state='running',case=case,session_id=session_id,pid=child.pid,
                    identity=identity(child.pid),manifest=str(manifest),phase=phase,smoke=smoke))
                last=time.monotonic();size=0;last_accepted=0
                while True:
                    content=log_path.read_text(errors='replace')
                    if len(content)!=size:size=len(content)
                    for entry in remaining:
                        if entry['output'] not in accepted and accept(entry,case,content,smoke):accepted.add(entry['output'])
                    if len(accepted)!=last_accepted:
                        last=time.monotonic();last_accepted=len(accepted)
                    val.progress(p,'running')
                    if child.poll() is not None:break
                    if STOP.is_set() or val.FATAL.search(content) or time.monotonic()-last>900:
                        terminate(child);break
                    STOP.wait(1)
                code=child.wait()
                content=log_path.read_text(errors='replace')
                for entry in remaining:
                    if entry['output'] not in accepted:accept(entry,case,content,smoke)
            finally:
                if child is not None:terminate(child)
                with MUTEX:ACTIVE.pop(gpu,None)
                if cache.exists():shutil.rmtree(cache)
                dump(ROOT/f'gpu-{gpu}.json',dict(state='exited',pid=child.pid if child else None,session_id=session_id,engine_group_exited=True,cache_removed=True))
        if STOP.is_set():raise InterruptedError('Stopped')
        if code==0 and all(val.valid(Path(e['output']),e,case,smoke) for e in entries):return
        if attempt==1:raise RuntimeError('Persistent worker failed twice: '+str(log_path))


def prepare():
    from extension import prepare_extension
    with lock():
        if (ROOT/'protocol.json').exists():return verify()
        p=prepare_extension(SOURCE,hashes)
        val.progress(p,'prepared')
        return p


def verify():
    from extension import verify_extension
    return verify_extension(SOURCE,hashes)


def verify_selector_checks():
    checks=json.loads((ROOT/'selector-validation.json').read_text())
    if not checks['complete'] or checks['source_hashes']!=hashes(ROOT/'private_ucm/ucm/sparse/prophetkv'):
        raise ValueError('Missing current CPU/CUDA selector validation')
    if checks['case_parameters']!=PARAMETERS:
        raise ValueError('Selector checks used different parameters')


def ensure_no_workers():
    for path in ROOT.glob('gpu-*.json'):
        state=json.loads(path.read_text())
        if state.get('pid') and group_alive(state['pid']):
            raise RuntimeError('Owned engine group is still live: '+str(path))


def run():
    with lock():
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only supervisor required')
        p=verify();verify_selector_checks();ensure_no_workers()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        status=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),identity=identity(os.getpid()),
                    started_at=time.time(),state='running')
        dump(ROOT/'supervisor.json',status)
        state='failed'
        try:
            def sweep(gpu):
                samples=[m for m in p['samples'] if m['physical_gpu']==gpu]
                for phase,case in enumerate(p['phases'][str(gpu)]):
                    entries=[dict(m,output=str(ROOT/'records'/m['id']/(case+'.json'))) for m in samples]
                    session(p,gpu,case,entries,phase)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending={pool.submit(sweep,gpu) for gpu in GPU_UUIDS}
                while pending:
                    done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        if future.exception():STOP.set()
                        future.result()
            rows=val.all_records(p)
            if len(rows)!=600:raise ValueError('Incomplete measurement matrix')
            state='complete'
        except BaseException as error:
            STOP.set();state='stopped' if isinstance(error,InterruptedError) else 'failed'
            dump(ROOT/'failure.json',dict(error=repr(error),at=time.time()))
            raise
        finally:
            STOP.set()
            with MUTEX:children=list(ACTIVE.values())
            for child in children:terminate(child)
            ensure_no_workers()
            dump(ROOT/'cleanup.json',dict(complete=True,state=state,at=time.time(),owned_workers_exited=True))
            dump(ROOT/'supervisor.json',{**status,'state':state,'ended_at':time.time()})
            val.progress(p,state)


def detach():
    with (ROOT/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with lock():
            verify();verify_selector_checks();ensure_no_workers()
            for name in ('supervisor.json','reporter.json'):
                if (ROOT/name).exists():
                    saved=json.loads((ROOT/name).read_text())
                    if identity(saved['pid'])==saved['identity']:
                        raise RuntimeError('Refusing duplicate '+name)
        processes_to_launch=(('supervisor', 'run.py', ['run']), ('reporter', 'report.py', ['watch']))
        for name,script,args in processes_to_launch:
            cmd=['nohup',str(PYTHON),'-u',str(SOURCE/script),*args]
            with (ROOT/(name+'.log')).open('a') as log:
                child=subprocess.Popen(cmd,cwd='/tmp',env=cpu_env(),stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if child.poll() is not None:
                    raise RuntimeError(name+' exited; inspect its log')
                path=ROOT/(name+'.json')
                if path.exists() and json.loads(path.read_text()).get('pid')==child.pid:
                    break
                time.sleep(.2)
            else:raise RuntimeError(name+' failed to publish state')
            dump(ROOT/(name+'-launch.json'),dict(pid=child.pid,identity=identity(child.pid),command=cmd,
                  pgid=os.getpgid(child.pid),sid=os.getsid(child.pid),nohup=True,stdin='/dev/null',at=time.time()))
            print(name,child.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','run','detach','status','verify','smoke'))
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='verify':verify()
    elif args.command=='run':run()
    elif args.command=='detach':detach()
    elif args.command=='smoke':parser.error('No separate qualification/smoke phase is authorized')
    else:print((ROOT/'progress.json').read_text())
