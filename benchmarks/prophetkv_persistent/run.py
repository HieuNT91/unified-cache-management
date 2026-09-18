#!/usr/bin/env python3
"""Pinned persistent-engine sweep; validation gates precede measured phases."""
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

OLD=REPO/'.results/prophetkv-qwen3-ruler100-longbenchv2-64k-c4096-20260918'
SOURCE=Path(__file__).resolve().parent
CACHE=Path('/tmp/ucm-prophetkv-persistent-20260918')
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
        CUDA_HOME=str(REPO/'.tools/cuda-12.6'),LD_LIBRARY_PATH=str(REPO/'.tools/cuda-12.6/lib64'),
        ENABLE_SPARSE='TRUE',CUDA_DEVICE_ORDER='PCI_BUS_ID',VLLM_ALLOW_INSECURE_SERIALIZATION='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'])
    return env


def prepare():
    with lock():
        if (ROOT/'protocol.json').exists():return verify()
        old=json.loads((OLD/'supervisor.json').read_text())
        if identity(old['pid'])==old['identity']:raise RuntimeError('Original supervisor is live')
        if not json.loads((OLD/'cleanup.json').read_text())['complete']:raise RuntimeError('Original cleanup incomplete')
        p=json.loads((OLD/'protocol.json').read_text())
        for meta in p['samples']:
            if sha(meta['input_path'])!=meta['input_sha256']:raise ValueError('Frozen input changed')
        private=ROOT/'private_ucm/ucm'
        shutil.copytree(OLD/'private_ucm/ucm',private,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
        shutil.copy2(SOURCE/'persistent_connector.py',private/'integration/vllm/persistent_connector.py')
        shutil.copy2(SOURCE/'lifecycle.py',private/'sparse/prophetkv/lifecycle.py')
        shutil.copytree(SOURCE,ROOT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
        p.update(study='ProphetKV persistent-engine UCM/vLLM port',parent_protocol_sha256=sha(OLD/'protocol.json'),
            prior_results=str(OLD),sources=hashes(ROOT/'source'),private_ucm_hashes=hashes(private),
            max_model_len=((max(m['tokens'] for m in p['samples'])+128+63)//64)*64,
            model_length_policy='fixed longest frozen prompt plus 128, rounded to 64; identical across configurations',
            engine_policy='one engine per GPU/configuration; 12 normal initializations; validation/recovery separate',
            dummy_holders=False,created_at=time.time())
        dump(ROOT/'protocol.json',p)
        dump(ROOT/'execution.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),sources=p['sources'],
            private_ucm_hashes=p['private_ucm_hashes'],gpu_uuids=GPU_UUIDS,
            phases={gpu:list(CASES[gpu%3:]+CASES[:gpu%3]) for gpu in GPU_UUIDS},
            parent_cleanup_sha256=sha(OLD/'cleanup.json'),restarted_measurements=2652))
        dump(ROOT/'prompt_manifest.json',p['samples']);val.progress(p,'prepared')
        return p


def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    if hashes(ROOT/'source')!=p['sources'] or hashes(SOURCE)!=p['sources']:raise ValueError('Pinned source changed')
    if hashes(ROOT/'private_ucm/ucm')!=p['private_ucm_hashes']:raise ValueError('Private UCM changed')
    for path,digest in p['runtime_file_hashes'].items():
        if sha(path)!=digest:raise ValueError('Runtime changed')
    for m in p['samples']:
        if sha(m['input_path'])!=m['input_sha256']:raise ValueError('Frozen prompt changed')
    if len(p['samples'])!=884 or p['measured_requests']!=2652:raise ValueError('Scope changed')
    return p


def group_alive(pid):
    for f in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=f.read_text().split()
            if int(fields[4])==pid and fields[2]!='Z':return True
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
                last=time.monotonic();size=0
                while True:
                    content=log_path.read_text(errors='replace')
                    if len(content)!=size:last=time.monotonic();size=len(content)
                    for entry in remaining:
                        if entry['output'] not in accepted and accept(entry,case,content,smoke):accepted.add(entry['output'])
                    val.progress(p,'validation' if phase<0 else 'sweep')
                    if child.poll() is not None:break
                    if STOP.is_set() or val.FATAL.search(content) or time.monotonic()-last>600:
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
        if STOP.is_set():raise InterruptedError('Stopped')
        if code==0 and all(val.valid(Path(e['output']),e,case,smoke) for e in entries):return
        if attempt==1:raise RuntimeError('Persistent worker failed twice: '+str(log_path))


def run():
    with lock():
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only supervisor required')
        p=verify();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        status=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),identity=identity(os.getpid()),started_at=time.time(),state='running')
        dump(ROOT/'supervisor.json',status);state='failed'
        try:
            from gates import gates
            gates(p)
            sweep_started=time.time()
            dump(ROOT/'sweep-start.json',dict(started_at=sweep_started,validation_seconds=sweep_started-status['started_at']))
            def sweep(gpu):
                methods=CASES[gpu%3:]+CASES[:gpu%3]
                samples=[m for m in p['samples'] if m['physical_gpu']==gpu]
                for phase,case in enumerate(methods):
                    entries=[dict(m,output=str(ROOT/'records'/m['id']/(case+'.json'))) for m in samples]
                    session(p,gpu,case,entries,phase)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending={pool.submit(sweep,gpu) for gpu in GPU_UUIDS}
                while pending:
                    done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in done:
                        if f.exception():STOP.set()
                        f.result()
            rows=val.all_records(p)
            if len(rows)!=2652:raise ValueError('Incomplete matrix')
            from prophetkv_report import report
            report(rows,'complete',True)
            measured_sessions={r['session_id'] for r in rows}
            dump(ROOT/'final/session-costs.json',dict(normal_engine_initializations=12,
                accepted_measurement_sessions=len(measured_sessions),
                total_sweep_seconds=time.time()-sweep_started,
                validation_seconds=sweep_started-status['started_at'],
                attempted_measurement_initializations=sum(json.loads(f.read_text())['phase']>=0 for f in (ROOT/'sessions').glob('*.launch.json')),
                sessions=[json.loads((ROOT/'sessions'/(s+'.json')).read_text()) for s in sorted(measured_sessions)]))
            state='complete'
        except BaseException as error:
            STOP.set();state='stopped' if isinstance(error,InterruptedError) else 'failed'
            dump(ROOT/'failure.json',dict(error=repr(error),at=time.time()))
            raise
        finally:
            STOP.set()
            with MUTEX:children=list(ACTIVE.values())
            for child in children:terminate(child)
            dump(ROOT/'cleanup.json',dict(complete=True,state=state,at=time.time(),owned_workers_exited=True))
            dump(ROOT/'supervisor.json',{**status,'state':state,'ended_at':time.time()});val.progress(p,state)
            if state=='complete':dump(ROOT/'final/validation.json',dict(complete=True,validated=2652,
                protocol_sha256=sha(ROOT/'protocol.json'),gates_sha256=sha(ROOT/'gates/validation.json'),
                cleanup_sha256=sha(ROOT/'cleanup.json'),artifacts=hashes(ROOT/'final')))


def detach():
    with lock():verify()
    cmd=['nohup',str(PYTHON),'-u',str(SOURCE/'run.py'),'run']
    with (ROOT/'supervisor.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=REPO,env=cpu_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/'nohup-launch.json',dict(pid=child.pid,command=cmd,identity=identity(child.pid),at=time.time()))
    print(child.pid,flush=True)

if __name__=='__main__':
    sys.modules['run']=sys.modules[__name__]
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run','detach','status'));args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='run':run()
    elif args.command=='detach':detach()
    else:print((ROOT/'progress.json').read_text())
