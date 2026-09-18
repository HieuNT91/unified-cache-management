#!/usr/bin/env python3
"""Execution amendment: independently matched native-prefix numerical controls."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys,time
from pathlib import Path
import run_prophetkv as r
from prophetkv_common import sha,dump
AMEND=r.ROOT/'execution-v2'
FILES=('benchmarks/prophetkv_full/run_verified.py','benchmarks/prophetkv_full/dense_prefix_reference.py')
original_verify=r.verify
original_smoke=r.smoke

def freeze():
    with r.lock():
        original_verify()
        if (AMEND/'amendment.json').exists():return verify()
        AMEND.mkdir(exist_ok=True)
        prior=AMEND/'prior';prior.mkdir(exist_ok=True)
        for name in ('supervisor.json','supervisor.log','progress.json','cleanup.json','failure.json','nohup-launch.json','nohup-detachment.json','smoke/vt-65536-002-c4096/equivalence.json'):
            src=r.ROOT/name
            if src.exists():
                target=prior/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
        for name in FILES:
            target=AMEND/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r.REPO/name,target)
        dump(AMEND/'amendment.json',dict(created_at=time.time(),protocol_sha256=sha(r.ROOT/'protocol.json'),
            sources={n:sha(r.REPO/n) for n in FILES},prior_artifacts=r.hashes(prior),
            reason='Native vLLM exact-prefix reuse reproduces the no-cache discrepancy; ProphetKV100 and native dense reference are bitwise identical at all36 audited layers.',
            diagnostic_sha256=sha(r.ROOT/'diagnostic/native-prefix-comparison.json'),
            gate='All36 fresh-suffix hidden-state tensors must be bitwise equal to independent native dense computation after exact first-chunk reuse. Also record full-prefill comparison without conflating BF16 prefix-reuse drift with sparse implementation error.',
            unchanged=['measured inference','no-cache baseline','10%/20% methods','prompts','scoring','timing','sample GPU assignments','protocol hash','existing records'],
            scope=dict(ruler_prompts=700,longbench_prompts=184,measured_requests=2652)))

def verify():
    p=original_verify();a=json.loads((AMEND/'amendment.json').read_text())
    if a['protocol_sha256']!=sha(r.ROOT/'protocol.json'):raise ValueError('Original protocol changed')
    for n,d in a['sources'].items():
        if sha(r.REPO/n)!=d or sha(AMEND/'source'/n)!=d:raise ValueError('Amendment source changed')
    if r.hashes(AMEND/'prior')!=a['prior_artifacts']:raise ValueError('Preserved failed artifacts changed')
    return p

def reference_valid(path,meta):
    try:
        x=json.loads(path.read_text());log=path.with_suffix('.log').read_text()
        return (x['sample_id']==meta['id'] and x['input_sha256']==meta['input_sha256']
            and x['protocol_sha256']==sha(r.ROOT/'protocol.json')
            and x['source_sha256']==sha(r.REPO/FILES[1]) and x['num_cached_tokens']==meta['boundaries'][1]
            and x['computed_prompt_tokens']==meta['tokens']-meta['boundaries'][1]
            and x['physical_gpu']==meta['physical_gpu'] and x['gpu_uuid']==r.GPU_UUIDS[meta['physical_gpu']]
            and x['audit_sha256']==sha(path.with_suffix('.model-audit.pt'))
            and 'REFERENCE_COMPLETE' in log and not r.FATAL.search(log))
    except (OSError,KeyError,ValueError):return False

def native_reference(meta,directory):
    directory.mkdir(parents=True,exist_ok=True);out=directory/'native-prefix-reference.json'
    if reference_valid(out,meta):return
    gpu=meta['physical_gpu']
    for attempt in range(2):
        if r.STOP.is_set():raise InterruptedError('Stopped')
        r.idle(gpu)
        if out.exists() or out.with_suffix('.log').exists():
            archive=directory/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
            for f in directory.glob('native-prefix-reference.*'):f.rename(archive/f.name)
        cmd=[str(r.PYTHON),'-u',str(AMEND/'source'/FILES[1]),'--protocol',str(r.ROOT/'protocol.json'),'--sample',meta['input_path'],'--output',str(out)]
        child=None
        try:
            env=r.environment(gpu)
            # Reference dependencies are pinned original worker/helper sources.
            env['PROPHETKV_REFERENCE_HELPERS']=str(r.ROOT/'source/benchmarks/prophetkv_full')
            with out.with_suffix('.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd='/tmp',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with r.MUTEX:r.ACTIVE[gpu]=child
                dump(r.ROOT/f'gpu-{gpu}.json',dict(state='native_dense_reference',pid=child.pid,pgid=child.pid,gpu=gpu,sample=meta['id'],command=cmd,started_at=time.time()))
                print(f'native_reference gpu={gpu} sample={meta["id"]} pid={child.pid}',flush=True)
                last=time.monotonic();size=0
                while child.poll() is None:
                    content=out.with_suffix('.log').read_text(errors='replace')
                    if len(content)!=size:last=time.monotonic();size=len(content)
                    if r.STOP.is_set() or r.FATAL.search(content) or time.monotonic()-last>600:
                        r.terminate(child);break
                    r.STOP.wait(1)
                code=child.wait()
        finally:
            if child is not None:r.terminate(child)
            with r.MUTEX:r.ACTIVE.pop(gpu,None)
            if not r.STOP.is_set():r.reserve(gpu)
        if r.STOP.is_set():raise InterruptedError('Stopped')
        if code==0 and reference_valid(out,meta):return
        if attempt==1:raise RuntimeError('Native reference failed; see '+str(out.with_suffix('.log')))

def compare_tensors(full,method,native):
    import torch
    if set(full)!=set(range(36)) or set(method)!=set(full) or set(native)!=set(full):raise ValueError('Missing audit layers')
    rows=[]
    for i in range(36):
        if full[i].shape!=method[i].shape or full[i].shape!=native[i].shape:raise ValueError('Audit shape mismatch')
        if not all(torch.isfinite(x[i]).all() for x in (full,method,native)):raise ValueError('Nonfinite audit')
        rms=lambda x,y:float((x-y).square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-6))
        rows.append(dict(layer=i,passed=torch.equal(method[i],native[i]),bitwise_equal=torch.equal(method[i],native[i]),
            method_vs_native_relative_rms=rms(method[i],native[i]),method_vs_no_cache_relative_rms=rms(method[i],full[i]),
            native_vs_no_cache_relative_rms=rms(native[i],full[i])))
    return rows

def equivalence(directory):
    import torch
    load=lambda n:torch.load(directory/f'{n}.model-audit.pt',weights_only=True)[0]
    rows=compare_tensors(load('baseline'),load('prophetkv-100'),load('native-prefix-reference'))
    passed=all(x['passed'] for x in rows)
    dump(directory/'equivalence.json',dict(complete=True,passed=passed,execution_amendment_sha256=sha(AMEND/'amendment.json'),
        control='ProphetKV100 vs independent native dense exact-prefix reference',tolerance='bitwise equality at all36 layers',
        compared='all fresh-suffix hidden states; no-cache differences retained separately',layers=rows))
    if not passed:raise ValueError('Native dense-prefix equivalence failed')

AUDITED=False

def smoke(p,meta):
    global AUDITED
    if AUDITED:return original_smoke(p,meta)
    ruler=[m for m in p['samples'] if m['dataset']=='ruler'];lb=[m for m in p['samples'] if m['dataset']=='longbench_v2']
    audits={m['id']:m for m in (max(ruler,key=lambda m:m['tokens']),max(lb,key=lambda m:m['tokens']),max(lb,key=lambda m:m['fresh_suffix_tokens']))}
    def group(gpu):
        for m in audits.values():
            if m['physical_gpu']!=gpu:continue
            native_reference(m,r.ROOT/'smoke'/m['id'])
            original_smoke(p,m)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(group,gpu) for gpu in sorted({m['physical_gpu'] for m in audits.values()})}
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                if f.exception():r.STOP.set()
                f.result()
    AUDITED=True

def self_test():
    import torch
    a={i:torch.ones(2,2) for i in range(36)};b={i:torch.full((2,2),1.1) for i in range(36)}
    assert all(r['passed'] for r in compare_tensors(a,b,b))
    c={i:x.clone() for i,x in b.items()};c[17][0,0]+=.001
    assert not all(r['passed'] for r in compare_tensors(a,c,b))
    print('Matched-reference gate positive/negative tests passed',flush=True)

def install():
    r.verify=verify;r.equivalence=equivalence;r.smoke=smoke
    import prophetkv_report
    original_report=prophetkv_report.report
    def report(*args,**kwargs):
        directory=original_report(*args,**kwargs)
        note='\nExecution amendment: the 100% control is checked for bitwise equality against independent native dense computation after exact first-chunk reuse. The original no-cache comparison is retained; native prefix reuse reproduces its BF16 numerical differences. The measured baseline remains fresh full prefill. See execution-v2/amendment.json and smoke/*/equivalence.json.\n'
        for path in (directory/'REPORT.md',r.ROOT/'REPORT.md'):
            with path.open('a') as f:f.write(note)
        return directory
    prophetkv_report.report=report

def detach():
    with r.lock():verify()
    cmd=['nohup',str(r.PYTHON),'-u',str(Path(__file__).resolve()),'run']
    with (r.ROOT/'supervisor.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=r.REPO,env=r.cpu_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(r.ROOT/'nohup-launch.json',dict(pid=child.pid,command=cmd,start_new_session=True,stdin='/dev/null',execution_amendment_sha256=sha(AMEND/'amendment.json'),at=time.time()))
    print(f'Detached verified supervisor PID {child.pid}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','test','detach','run','status','report'));args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='test':self_test()
    elif args.command=='detach':detach()
    elif args.command=='status':print((r.ROOT/'progress.json').read_text())
    else:
        install()
        if args.command=='run':self_test();r.run()
        else:
            with r.lock():
                from prophetkv_report import report
                p=verify();report(r.all_records(p),json.loads((r.ROOT/'progress.json').read_text())['state'])
