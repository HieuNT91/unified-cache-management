import os,sys,json,time,signal,subprocess,shutil,fcntl,uuid,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parents[1]/'.results/vanilla-prophetkv32b-40samples-output256-20260924';sys.path.insert(0,str(HERE/'runtime'))
import suite
from common import load,dump,identity,group_alive,selected_devices,busy_devices
from validate import validate,sha
PYTHON=Path('/home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python');CACHE=Path('/tmp/ucm-vanilla-prophetkv32b-20260924');STOP=False

def env():
 e=suite.cpu_env();cuda='/home/thnguyen/unified-cache-management/.tools/cuda-12.6'
 e.update(CUDA_HOME=cuda,PATH=str(PYTHON.parent)+':'+cuda+'/bin:'+e['PATH'],LD_LIBRARY_PATH=cuda+'/lib64:'+e.get('LD_LIBRARY_PATH',''),VLLM_ALLOW_LONG_MAX_MODEL_LEN='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',WATCHDOG_SECONDS='900',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 return e

def verify():
 for name,h in load(D/'implementation.json').items():assert sha(name)==h,name
 for m in load(D/'protocol.json')['samples']:assert sha(m['input_path'])==m['input_sha256']
 assert load(D/'cpu-validation.json')['complete']

def accepted(m):return Path(m['output']).with_suffix('.validated.json').exists()
def accept(m,p,text):
 path=Path(m['output'])
 if accepted(m):return True
 if not path.exists():return False
 r=load(path);begin='REQUEST_BEGIN '+r['request_id']+'\n';end='REQUEST_COMPLETE '+r['request_id']+' '+str(path)+'\n'
 if end not in text:return False
 assert begin in text
 excerpt=begin+text.split(begin,1)[1].split(end,1)[0]+end+'WORKER_COMPLETE\n'
 path.with_suffix('.log').write_text(excerpt)
 validate(m,p,path)
 receipt=D/('first-'+m['case']+'-validation.json')
 if not receipt.exists():dump(receipt,dict(complete=True,record=str(path),validation=str(path.with_suffix('.validated.json')),at=time.time()))
 return True

def stop(*_):
 global STOP
 STOP=True

def run():
 global STOP
 with (D/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
  assert os.environ['CUDA_VISIBLE_DEVICES']==''
  if (D/'active.json').exists():assert not group_alive(load(D/'active.json')['pid'])
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  state=dict(pid=os.getpid(),identity=identity(os.getpid()),pgid=os.getpgrp(),sid=os.getsid(0),state='running',started_at=time.time());dump(D/'supervisor.json',state)
  p=load(D/'protocol.json');entries=[m|dict(case=case,output=str(D/'records'/m['id']/(case+'.json'))) for case in p['cases'] for m in p['samples']]
  outcome='failed'
  try:
   for case in p['cases']:
    for attempt in range(2):
     pending=[m for m in entries if m["case"]==case and not accepted(m)]
     if not pending:break
     listing,devices=selected_devices([1,2,3,4]);assert devices==p['gpu_devices'];assert not busy_devices(devices),'Allowed GPUs occupied'
     assert not CACHE.exists(),'Owned cache unexpectedly exists'
     for m in pending:
      path=Path(m['output']);artifacts=list(path.parent.glob(case+'.*'))
      if artifacts:
       archive=path.parent/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
       for f in artifacts:f.rename(archive/f.name)
     sid=uuid.uuid4().hex;session=D/'sessions'/(sid+'.json');manifest=session.with_suffix('.manifest.json');log=session.with_suffix('.log');dump(manifest,pending)
     e=env();e.update(CUDA_VISIBLE_DEVICES=','.join(d['uuid'] for d in devices),REMOTE_GPU_DEVICES=json.dumps(devices),SELECTOR_UCM_ROOT=str(HERE/'private_ucm'),ENABLE_SPARSE='TRUE',VLLM_USE_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_ALLOW_INSECURE_SERIALIZATION='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PROPHETKV_SCHEDULER_RECEIPT=str(session.with_suffix('.scheduler.json')),TP4_MEMORY_AUDIT=str(session.with_suffix('.memory-audit')))
     command=[str(PYTHON),'-u',str(HERE/'runtime/persistent_worker.py'),'--protocol',str(D/'protocol.json'),'--manifest',str(manifest),'--case',case,'--cache-dir',str(CACHE),'--session',str(session)]
     with log.open('w') as f:child=subprocess.Popen(command,cwd='/tmp',env=e,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     active=dict(pid=child.pid,identity=identity(child.pid),command=command,log=str(log),attempt=attempt+1,devices=devices,inventory=listing,state='running');dump(D/'active.json',active)
     last=time.monotonic();count=sum(accepted(m) for m in entries);failure=None
     try:
      while True:
       text=log.read_text(errors='replace')
       for m in pending:
        if not accepted(m):accept(m,p,text)
       new=sum(accepted(m) for m in entries)
       if new!=count:count=new;last=time.monotonic()
       dump(D/'progress.json',dict(state='measuring',case=case,validated=count,target=360,at=time.time()))
       if child.poll() is not None:
        if child.returncode or not all(accepted(m) for m in pending):raise RuntimeError('Incomplete measurement session: '+str(log))
        break
       if STOP:raise InterruptedError('Stopped')
       if suite.FATAL.search(text):raise RuntimeError('Fatal measurement log: '+str(log))
       if time.monotonic()-last>900:raise RuntimeError('900-second measurement progress watchdog')
       time.sleep(1)
     except RuntimeError as ex:failure=repr(ex)
     finally:
      suite.terminate(child);assert not group_alive(child.pid)
      if CACHE.exists():shutil.rmtree(CACHE)
      dump(D/'active.json',active|dict(state='exited',engine_group_exited=True,cache_removed=True,error=failure))
     if failure:
      dump(D/'sessions'/(sid+'.failure.json'),dict(error=failure,at=time.time(),attempt=attempt+1))
      if attempt==1:raise RuntimeError(failure)
   assert all(accepted(m) for m in entries)
   outcome='complete'
  finally:
   exited=not (D/'active.json').exists() or not group_alive(load(D/'active.json')['pid'])
   dump(D/'cleanup.json',dict(complete=exited and not CACHE.exists(),engine_exited=exited,cache_removed=not CACHE.exists(),at=time.time()))
   dump(D/'supervisor.json',state|dict(state=outcome,ended_at=time.time()))
   dump(D/'progress.json',dict(state='measurements_complete' if outcome=='complete' else outcome,validated=sum(accepted(m) for m in entries),target=360,at=time.time()))

def report():
 with (D/'report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=dict(pid=os.getpid(),identity=identity(os.getpid()),pgid=os.getpgrp(),sid=os.getsid(0),state='waiting',started_at=time.time());dump(D/'reporter.json',state)
  try:
   while True:
    s=load(D/'supervisor.json')
    if identity(s['pid'])!=s['identity']:
     if s['state']!='complete':raise RuntimeError('Measurement supervisor did not complete')
     break
    time.sleep(10)
   verify();assert load(D/'cleanup.json')['engine_exited'];assert not group_alive(load(D/'active.json')['pid'])
   from reporting import finalize
   finalize()
   dump(D/'progress.json',dict(state='complete',validated=360,target=360,at=time.time()))
   dump(D/'reporter.json',state|dict(state='complete',ended_at=time.time()))
  except BaseException as ex:
   dump(D/'reporter.json',state|dict(state='failed',error=repr(ex),ended_at=time.time()));raise

def status():
 for name in ['progress','supervisor','reporter','active']:
  if (D/(name+'.json')).exists():print(name,json.dumps(load(D/(name+'.json'))))

def detach():
 with (D/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
  for n in ['supervisor','reporter']:
   f=D/(n+'.json')
   if f.exists():
    s=load(f);assert identity(s['pid'])!=s['identity'],'Duplicate '+n
  for n,command in [('supervisor','run'),('reporter','report')]:
   cmd=['nohup',str(PYTHON),'-u',str(HERE/'run.py'),command]
   with (D/(n+'.log')).open('a') as f:child=subprocess.Popen(cmd,cwd='/tmp',env=env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   for _ in range(150):
    assert child.poll() is None,n+' exited'
    if (D/(n+'.json')).exists() and load(D/(n+'.json'))['pid']==child.pid:break
    time.sleep(.2)
   else:raise RuntimeError(n+' did not publish state')
   dump(D/(n+'-launch.json'),dict(pid=child.pid,identity=identity(child.pid),command=cmd,at=time.time()));print(n,child.pid,flush=True)
if __name__=='__main__':{'run':run,'report':report,'detach':detach,'status':status}[sys.argv[1]]()
