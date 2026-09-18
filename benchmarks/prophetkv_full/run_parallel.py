#!/usr/bin/env python3
"""Dataset-gated four-GPU scheduling without dummy holders."""
import argparse,concurrent.futures,json,os,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
import run_verified as v
r=v.r
from prophetkv_common import dump,sha
AMEND=r.ROOT/'execution-v3'
SOURCE='benchmarks/prophetkv_full/run_parallel.py'
LB_READY=threading.Event()

def verify():
    p=v.verify();a=json.loads((AMEND/'amendment.json').read_text())
    if a['parent_amendment_sha256']!=sha(v.AMEND/'amendment.json'):raise ValueError('Parent amendment changed')
    if a['source_sha256']!=sha(r.REPO/SOURCE) or a['source_sha256']!=sha(AMEND/'source'/SOURCE):raise ValueError('Scheduler changed')
    return p

def freeze():
    with r.lock():
        v.verify()
        if (AMEND/'amendment.json').exists():return verify()
        target=AMEND/'source'/SOURCE;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r.REPO/SOURCE,target)
        dump(AMEND/'amendment.json',dict(parent_amendment_sha256=sha(v.AMEND/'amendment.json'),source_sha256=sha(r.REPO/SOURCE),created_at=time.time(),
            reason='User requested use of all four GPUs and disabling dummy holders during benchmark execution.',
            scheduling='RULER work starts on GPUs1,3,4 after its own passed model gate. GPU2 completes LongBench gates then joins RULER work. Each GPU runs its LongBench work only after all LongBench gates pass. Prompt GPU affinity unchanged.',
            dummy_holders=False,scope_unchanged=True,inference_unchanged=True))

def run():
    v.install();r.verify=verify;r.reserve=lambda gpu:None
    from prophetkv_report import report
    with r.lock():
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only supervisor required')
        signal.signal(signal.SIGTERM,r.stop);signal.signal(signal.SIGINT,r.stop)
        p=verify();state='failed'
        status=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),identity=r.identity(os.getpid()),command=sys.argv,started_at=time.time(),state='running',execution_amendment_sha256=sha(AMEND/'amendment.json'))
        dump(r.ROOT/'supervisor.json',status)
        try:
            for gpu in r.GPU_UUIDS:r.release(gpu)
            v.self_test();r.numerical()
            ruler=[m for m in p['samples'] if m['dataset']=='ruler'];lb=[m for m in p['samples'] if m['dataset']=='longbench_v2']
            ruler_audit=max(ruler,key=lambda m:m['tokens'])
            v.native_reference(ruler_audit,r.ROOT/'smoke'/ruler_audit['id']);v.original_smoke(p,ruler_audit)
            lb_audits={m['id']:m for m in (max(lb,key=lambda m:m['tokens']),max(lb,key=lambda m:m['fresh_suffix_tokens']))}
            if len({m['physical_gpu'] for m in lb_audits.values()})!=1:raise ValueError('Expected frozen LongBench audit affinity')
            audit_gpu=next(iter(lb_audits.values()))['physical_gpu']
            def event(gpu,phase,kind):
                with r.MUTEX:
                    with (r.ROOT/'phases.jsonl').open('a') as f:f.write(json.dumps(dict(gpu=gpu,phase=phase,event=kind,at=time.time(),execution='v3'))+'\n')
            def worker(gpu):
                if gpu==audit_gpu:
                    event(gpu,'longbench_audits','start')
                    for m in lb_audits.values():
                        v.native_reference(m,r.ROOT/'smoke'/m['id']);v.original_smoke(p,m)
                    LB_READY.set();event(gpu,'longbench_audits','end')
                event(gpu,'ruler','start');r.per_gpu(p,gpu,ruler,False);event(gpu,'ruler','end')
                while not LB_READY.wait(1):
                    if r.STOP.is_set():raise InterruptedError('Stopped')
                event(gpu,'longbench_v2','start');r.per_gpu(p,gpu,lb,False);event(gpu,'longbench_v2','end')
            dump(r.ROOT/'phase.json',dict(phase='dataset_gated_four_gpu_sweep',started_at=time.time(),dummy_holders=False,longbench_audit_gpu=audit_gpu))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                pending={pool.submit(worker,gpu) for gpu in r.GPU_UUIDS}
                while pending:
                    done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                    for f in done:
                        if f.exception():r.STOP.set()
                        f.result()
            verify();rows=r.all_records(p)
            if len(rows)!=p['measured_requests']:raise ValueError('Incomplete matrix')
            report(rows,'complete',True);state='complete'
        except BaseException as error:
            r.STOP.set();state='stopped' if isinstance(error,InterruptedError) else 'failed'
            dump(r.ROOT/'failure.json',dict(error=repr(error),at=time.time()))
            report(r.all_records(p),state)
            raise
        finally:
            r.STOP.set()
            with r.MUTEX:children=list(r.ACTIVE.values())
            for child in children:r.terminate(child)
            for gpu in r.GPU_UUIDS:r.release(gpu);r.cleanup_cache(gpu)
            dump(r.ROOT/'cleanup.json',dict(complete=True,state=state,at=time.time(),owned_workers_exited=True,placeholders_released=True,caches_removed=not any(r.CACHE.glob('gpu-*'))))
            r.progress(p,state);dump(r.ROOT/'supervisor.json',{**status,'state':state,'ended_at':time.time()})
            if state=='complete':dump(r.ROOT/'final/validation.json',dict(complete=True,validated=len(rows),protocol_sha256=sha(r.ROOT/'protocol.json'),execution_amendment_sha256=sha(AMEND/'amendment.json'),cleanup_sha256=sha(r.ROOT/'cleanup.json'),gates=r.hashes(r.ROOT/'gates'),smoke_certificates={str(f.relative_to(r.ROOT)):sha(f) for f in (r.ROOT/'smoke').glob('*/validation.json')},artifacts=r.hashes(r.ROOT/'final')))

def detach():
    with r.lock():verify()
    cmd=['nohup',str(r.PYTHON),'-u',str(Path(__file__).resolve()),'run']
    with (r.ROOT/'supervisor.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=r.REPO,env=r.cpu_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(r.ROOT/'nohup-launch.json',dict(pid=child.pid,command=cmd,start_new_session=True,stdin='/dev/null',execution_amendment_sha256=sha(AMEND/'amendment.json'),at=time.time()))
    print(f'Detached four-GPU supervisor PID {child.pid}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','detach','run','status'));args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='detach':detach()
    elif args.command=='run':run()
    else:print((r.ROOT/'progress.json').read_text())
