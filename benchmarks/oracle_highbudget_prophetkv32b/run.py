"""Locked detached sequential experiments; phase2 cannot start before phase1 reporting."""
import os,sys,json,time,signal,subprocess,shutil,fcntl,uuid
from pathlib import Path
from prepare import H,D,sha
from common import load,dump,identity,group_alive,selected_devices,busy_devices
from query_config import STAGES,TARGETS
from validate import validate
import suite
PYTHON=Path('/home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python')
CACHE=Path('/tmp/ucm-oracle-highbudget-prophetkv32b-20260925');STOP=False

def env():
    e=suite.cpu_env();cuda='/home/thnguyen/unified-cache-management/.tools/cuda-12.6'
    e.update(CUDA_HOME=cuda,PATH=str(PYTHON.parent)+':'+cuda+'/bin:'+e['PATH'],LD_LIBRARY_PATH=cuda+'/lib64:'+e.get('LD_LIBRARY_PATH',''),VLLM_ALLOW_LONG_MAX_MODEL_LEN='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',WATCHDOG_SECONDS='900',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return e

def verify():
    for name,h in load(D/'implementation.json').items():assert sha(name)==h,name
    assert load(D/'cpu-validation.json')['complete'] and load(D/'cpu-report-validation.json')['complete']
    for stage in STAGES:
        p=load(D/stage/'protocol.json');assert len(load(D/stage/'schedule.json'))==TARGETS[stage]
        for m in p['samples']:assert sha(m['input_path'])==m['input_sha256']
        for path in (D/stage/'records').rglob('*.validated.json'):
            receipt=load(path);assert receipt['complete']
            for name,h in receipt['sha256'].items():assert sha(name)==h,name

def accepted(m):return Path(m['output']).with_suffix('.validated.json').exists()
def accept(root,m,p,text):
    path=Path(m['output'])
    if accepted(m):return True
    if not path.exists():return False
    r=load(path);begin='REQUEST_BEGIN '+r['request_id']+'\n';end='REQUEST_COMPLETE '+r['request_id']+' '+str(path)+'\n'
    if end not in text:return False
    assert begin in text
    path.with_suffix('.log').write_text(begin+text.split(begin,1)[1].split(end,1)[0]+end+'WORKER_COMPLETE\n')
    validate(root,m,p,path)
    receipt=root/('first-'+m['case']+'-validation.json')
    if not receipt.exists():dump(receipt,dict(complete=True,record=str(path),validation=str(path.with_suffix('.validated.json')),at=time.time()))
    return True

def stop(*_):
    global STOP
    STOP=True

def measure(stage):
    root=D/stage;p=load(root/'protocol.json');entries=load(root/'schedule.json');last_active=None
    if (root/'stage.json').exists() and load(root/'stage.json')['state']=='complete':return
    dump(root/'stage.json',dict(state='running',started_at=time.time()))
    for case in p['cases']:
        for attempt in range(2):
            if STOP:raise InterruptedError('Stopped')
            pending=[m for m in entries if m['case']==case and not accepted(m)]
            if not pending:break
            listing,devices=selected_devices([1,2,3,4]);assert devices==p['gpu_devices'] and not busy_devices(devices)
            assert not CACHE.exists(),'Owned cache unexpectedly exists'
            for m in pending:
                path=Path(m['output']);artifacts=list(path.parent.glob(case+'.*'))
                if artifacts:
                    archive=path.parent/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
                    for f in artifacts:f.rename(archive/f.name)
            sid=uuid.uuid4().hex;session=root/'sessions'/(sid+'.json');manifest=session.with_suffix('.manifest.json');log=session.with_suffix('.log');dump(manifest,pending)
            e=env();e.update(CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in devices),REMOTE_GPU_DEVICES=json.dumps(devices),SELECTOR_UCM_ROOT=str(H/'private_ucm'),ENABLE_SPARSE='TRUE',VLLM_USE_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_ALLOW_INSECURE_SERIALIZATION='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PROPHETKV_SCHEDULER_RECEIPT=str(session.with_suffix('.scheduler.json')),TP4_MEMORY_AUDIT=str(session.with_suffix('.memory-audit')))
            command=[str(PYTHON),'-u',str(H/'runtime/persistent_worker.py'),'--protocol',str(root/'protocol.json'),'--manifest',str(manifest),'--case',case,'--cache-dir',str(CACHE),'--session',str(session)]
            with log.open('w') as f:child=subprocess.Popen(command,cwd='/tmp',env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(pid=child.pid,identity=identity(child.pid),command=command,log=str(log),attempt=attempt+1,stage=stage,case=case,devices=devices,inventory=listing,state='running')
            last_active=active;dump(D/'active.json',active);dump(root/'active.json',active)
            last=time.monotonic();count=sum(accepted(m) for m in entries);failure=None
            try:
                while True:
                    text=log.read_text(errors='replace')
                    for m in pending:
                        if not accepted(m):accept(root,m,p,text)
                    new=sum(accepted(m) for m in entries)
                    if new!=count:count=new;last=time.monotonic()
                    progress=dict(state='measuring',stage=stage,case=case,validated=count,target=TARGETS[stage],at=time.time())
                    dump(root/'progress.json',progress);dump(D/'progress.json',progress|dict(total_validated=count+(TARGETS['oracle'] if stage=='highbudget' else 0),total_target=1230))
                    if child.poll() is not None:
                        if child.returncode or not all(accepted(m) for m in pending):raise RuntimeError('Incomplete session: '+str(log))
                        break
                    if STOP:raise InterruptedError('Stopped')
                    if suite.FATAL.search(text):raise RuntimeError('Fatal log: '+str(log))
                    if time.monotonic()-last>900:raise RuntimeError('900-second progress watchdog')
                    time.sleep(1)
            except (RuntimeError,AssertionError,ValueError) as ex:failure=repr(ex)
            finally:
                suite.terminate(child);assert not group_alive(child.pid)
                if CACHE.exists():shutil.rmtree(CACHE)
                done=active|dict(state='exited',engine_group_exited=True,cache_removed=True,error=failure)
                dump(D/'active.json',done);dump(root/'active.json',done)
            if failure:
                dump(session.with_suffix('.failure.json'),dict(error=failure,at=time.time(),attempt=attempt+1))
                if attempt==1:raise RuntimeError(failure)
    assert all(accepted(m) for m in entries)
    active=last_active or load(root/'active.json');assert not group_alive(active['pid']) and not CACHE.exists()
    dump(root/'cleanup.json',dict(complete=True,engine_exited=True,cache_removed=True,last_engine_pid=active['pid'],at=time.time()))
    dump(root/'stage.json',dict(state='complete',validated=len(entries),last_engine_pid=active['pid'],ended_at=time.time()))
    dump(root/'progress.json',dict(state='measurements_complete',validated=len(entries),target=TARGETS[stage],at=time.time()))

def phase_ready(stage):
    root=D/stage;cert=root/'final-validation.json'
    if not cert.exists():return False
    c=load(cert);assert c['complete'] and c['validated']==TARGETS[stage]
    for name,h in c['artifact_sha256'].items():assert sha(name)==h,name
    assert load(root/'cleanup.json')['complete'] and not group_alive(load(root/'stage.json')['last_engine_pid'])
    return True

def run():
    with (D/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify();assert os.environ['CUDA_VISIBLE_DEVICES']==''
        if (D/'active.json').exists():assert not group_alive(load(D/'active.json')['pid'])
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),pgid=os.getpgrp(),sid=os.getsid(0),state='running',started_at=time.time());dump(D/'supervisor.json',state)
        outcome='failed';error=None
        try:
            for stage in STAGES:
                measure(stage)
                deadline=time.monotonic()+1800
                while not phase_ready(stage):
                    if STOP:raise InterruptedError('Stopped')
                    if (D/'reporter.json').exists():
                        s=load(D/'reporter.json');assert s['state']!='failed' and identity(s['pid'])==s['identity'],'Reporter exited'
                    if time.monotonic()>deadline:raise RuntimeError('Report progress watchdog')
                    time.sleep(2)
                dump(D/(stage+'-completed.json'),dict(complete=True,validated=TARGETS[stage],engine_exited=True,report_verified=True,at=time.time()))
            outcome='complete'
        except BaseException as ex:
            error=repr(ex);raise
        finally:
            exited=not (D/'active.json').exists() or not group_alive(load(D/'active.json')['pid'])
            dump(D/'cleanup.json',dict(complete=exited and not CACHE.exists(),engine_exited=exited,cache_removed=not CACHE.exists(),at=time.time()))
            dump(D/'supervisor.json',state|dict(state=outcome,error=error,ended_at=time.time()))
            dump(D/'progress.json',dict(state=outcome,total_validated=sum(1 for stage in STAGES for f in (D/stage/'records').rglob('*.validated.json')),total_target=1230,at=time.time()))

def report():
    with (D/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),identity=identity(os.getpid()),pgid=os.getpgrp(),sid=os.getsid(0),state='waiting',started_at=time.time());dump(D/'reporter.json',state)
        try:
            for stage in STAGES:
                root=D/stage
                while not (root/'stage.json').exists() or load(root/'stage.json')['state']!='complete':
                    s=load(D/'supervisor.json')
                    if identity(s['pid'])!=s['identity']:raise RuntimeError('Supervisor exited before stage completion')
                    time.sleep(5)
                if not phase_ready(stage):
                    verify();dump(D/'reporter.json',state|dict(state='reporting',stage=stage))
                    from reporting import finalize
                    finalize(root)
                dump(D/'reporter.json',state|dict(state='waiting',completed_stage=stage))
            while True:
                s=load(D/'supervisor.json')
                if identity(s['pid'])!=s['identity']:
                    assert s['state']=='complete';break
                time.sleep(2)
            assert load(D/'cleanup.json')['complete']
            dump(D/'final-validation.json',dict(complete=True,validated=1230,stages=TARGETS,engine_exited=True,at=time.time()))
            dump(D/'reporter.json',state|dict(state='complete',ended_at=time.time()))
        except BaseException as ex:
            dump(D/'reporter.json',state|dict(state='failed',error=repr(ex),ended_at=time.time()));raise

def status():
    for root in [D,*[D/s for s in STAGES]]:
        for n in ['progress','supervisor','reporter','active']:
            if (root/(n+'.json')).exists():print(root.name,n,json.dumps(load(root/(n+'.json'))))

def detach():
    with (D/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        for n in ['supervisor','reporter']:
            f=D/(n+'.json')
            if f.exists():
                s=load(f);assert identity(s['pid'])!=s['identity'],'Duplicate '+n
        for n,command in [('supervisor','run'),('reporter','report')]:
            cmd=['nohup',str(PYTHON),'-u',str(H/'run.py'),command]
            with (D/(n+'.log')).open('a') as f:child=subprocess.Popen(cmd,cwd='/tmp',env=env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(300):
                assert child.poll() is None,n+' exited'
                if (D/(n+'.json')).exists() and load(D/(n+'.json'))['pid']==child.pid:break
                time.sleep(.2)
            else:raise RuntimeError(n+' did not publish state')
            dump(D/(n+'-launch.json'),dict(pid=child.pid,identity=identity(child.pid),command=cmd,at=time.time()));print(n,child.pid,flush=True)
if __name__=='__main__':{'run':run,'report':report,'detach':detach,'status':status}[sys.argv[1]]()
