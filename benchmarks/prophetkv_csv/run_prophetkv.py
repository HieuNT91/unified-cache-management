#!/usr/bin/env python3
"""ProphetKV preparation, audited pilots, gated detached sweep and reporting."""
import argparse
import concurrent.futures
from contextlib import contextmanager
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from prophetkv_gpu import REPO,ROOT,PYTHON,GPU_UUIDS,check,processes,reserve,release,identity
from prophetkv_data import MODEL,TASKS,LENGTHS,CHUNKS,generate_sources,prepare_samples
from prophetkv_common import sha,dump,load_sample,score
from cacheblend_prophetkv import CASES,CONTROLS,TIMING

CACHE=Path('/tmp/ucm-prophetkv-csv-backfill-20260918')
INSTALLED=REPO/'.envs/cacheblend/lib/python3.10/site-packages/ucm'
SOURCES=tuple('benchmarks/prophetkv_csv/'+n for n in ('run_prophetkv.py','cacheblend_prophetkv.py','prophetkv_common.py',
    'prophetkv_data.py','prophetkv_gpu.py','prophetkv_report.py','test_prophetkv.py','cacheblend_ruler.py','PROPHETKV.md'))+tuple(str(p.relative_to(REPO)) for p in sorted((REPO/'ucm/sparse/prophetkv').glob('*.py')))+('ucm/sparse/factory.py','ucm/sparse/blend/blend.py','ucm/sparse/blend/selection.py')
FATAL=re.compile(r'Traceback|\[UC\]\[E\]|load kv cache failed|dump kv cache failed|CUDA out of memory|EngineDeadError|ERROR\s|\b(?:RuntimeError|AssertionError|ValueError):')
STOP=threading.Event();ACTIVE={};MUTEX=threading.RLock();PROGRESS=threading.Lock()

@contextmanager
def lock():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}

def runtime_hashes():
    return {str(f):sha(f) for f in sorted((INSTALLED.parent/'vllm').rglob('*.py'))}

def cpu_env():
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(CUDA_VISIBLE_DEVICES='',PROPHETKV_REPO=str(REPO),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0')
    return env

def environment(gpu):
    check(gpu)
    env=cpu_env();env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu],SELECTOR_PHYSICAL_GPU=str(gpu),
        SELECTOR_UCM_ROOT=str(ROOT/'private_ucm'),CUDA_HOME=str(REPO/'.tools/cuda-12.6'),
        LD_LIBRARY_PATH=str(REPO/'.tools/cuda-12.6/lib64'),ENABLE_SPARSE='TRUE',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        VLLM_ALLOW_INSECURE_SERIALIZATION='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        PATH=str(PYTHON.parent)+os.pathsep+os.environ['PATH'])
    return env

def prepare():
    with lock():
        if (ROOT/'protocol.json').exists():return verify()
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Preparation must expose no GPUs')
        generate_sources()
        samples,datasets=prepare_samples(sha,dump)
        private=ROOT/'private_ucm/ucm'
        if private.exists():shutil.rmtree(private)
        shutil.copytree(INSTALLED,private,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('blend.py','selection.py'):
            shutil.copy2(REPO/'ucm/sparse/blend'/name,private/'sparse/blend'/name)
        shutil.copytree(REPO/'ucm/sparse/prophetkv',private/'sparse/prophetkv',ignore=shutil.ignore_patterns('__pycache__'))
        with (private/'sparse/factory.py').open('a') as out:
            out.write('\nUcmSparseFactory.register_sparse_method("ProphetKV", "ucm.sparse.prophetkv.prophetkv", "ProphetKV")\n')
        for name in SOURCES:
            target=ROOT/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
        reference=ROOT/'reference';reference.mkdir(exist_ok=True)
        for name in ('LICENSE','NOTICE'):
            source=REPO/'.downloads/QCFuse'/name
            if source.exists():shutil.copy2(source,reference/name)
        p=dict(study='ProphetKV UCM/vLLM port',schema_version=1,model=str(MODEL),model_revision=MODEL.name,
            model_hashes=hashes(MODEL),runtime_file_hashes=runtime_hashes(),runtime_versions={n:version(n) for n in ('uc-manager','vllm','torch','transformers')},
            max_model_len=max(((m['tokens']+191)//64)*64 for m in samples),
            model_length_policy='prompt length plus 128, rounded to 64; identical across methods',gpu_memory_utilization=.93,max_output_tokens=128,thinking_enabled=False,dtype='bfloat16',tensor_parallel_size=1,
            cases=list(CASES),tasks=list(TASKS),context_targets=list(LENGTHS),chunk_sizes=list(CHUNKS),samples_per_task=10,
            pilot_samples_per_task=0,performance_gate_required=False,
            requested_chunk_sizes=list(CHUNKS),aligned_chunk_sizes={str(c):((c+63)//64)*64 for c in CHUNKS},
            scope='Only missing CSV cases; historical baseline/P20 retained for VT/CWE/multivalue. Fresh NIAH-single baseline/P20/P30/P40.',measured_requests=200,samples=samples,dataset_hashes=datasets,
            sources={n:sha(REPO/n) for n in SOURCES},private_ucm_hashes=hashes(private),installed_ucm_hashes=hashes(INSTALLED),
            gpu_uuids=GPU_UUIDS,prompt_format='native non-thinking chat; numbered block-padded chunks; fresh 256 suffix',
            timing_source=TIMING,storage='buffered local warm cache',created_at=time.time(),
            paper='https://arxiv.org/html/2602.02579v3',kernel_reference='uYanJX/QCFuse@38795d91d900debb5f2df23bd7296b3ff97960c8',
            ruler_generator_hashes=hashes(REPO/'.downloads/RULER/scripts/data'),
            selection='question-only context softmax, head/query mean then 36-layer FP32 sum; floor budget; position-stable ties',
            rate_denominator='cached tokens excluding exact first chunk and fresh suffix',
            gpu_affinity='task/row assignment identical across lengths, chunk sizes and configurations')
        dump(ROOT/'protocol.json',p)
        dump(ROOT/'execution.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),gpu_uuids=GPU_UUIDS,
            sample_gpus={m['id']:m['physical_gpu'] for m in samples},sources=p['sources']))
        dump(ROOT/'prompt_manifest.json',samples)
        dump(ROOT/'dataset_manifest.json',dict(seed=42,hashes=datasets,selection='source row ordinals 0–9'))
        dump(ROOT/'progress.json',dict(state='prepared',validated=0,target=200))
        return p

def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in p['sources'].items():
        if sha(REPO/name)!=digest or sha(ROOT/'source'/name)!=digest:raise ValueError('Source changed: '+name)
    if hashes(ROOT/'private_ucm/ucm')!=p['private_ucm_hashes']:raise ValueError('Private implementation changed')
    if hashes(INSTALLED)!=p['installed_ucm_hashes']:raise ValueError('Shared installation changed')
    if hashes(MODEL)!=p['model_hashes']:raise ValueError('Checkpoint changed')
    if runtime_hashes()!=p['runtime_file_hashes']:raise ValueError('Patched runtime changed')
    for path,digest in p['dataset_hashes'].items():
        if sha(path)!=digest:raise ValueError('Dataset changed')
    for m in p['samples']:
        if sha(m['input_path'])!=m['input_sha256']:raise ValueError('Frozen prompt changed')
    if sum(len(m['required_cases']) for m in p['samples'])!=200:raise ValueError('Scope mismatch')
    return p

def validate(path,meta,case,smoke=False):
    r=json.loads(path.read_text());log=path.with_suffix('.log').read_text()
    if FATAL.search(log) or 'WORKER_COMPLETE' not in log:raise ValueError('Unclean worker log')
    expected=dict(sample_id=meta['id'],case=case,input_sha256=meta['input_sha256'],prompt_sha256=meta['prompt_sha256'],
        protocol_sha256=sha(ROOT/'protocol.json'),physical_gpu=meta['physical_gpu'],gpu_uuid=GPU_UUIDS[meta['physical_gpu']],
        prompt_tokens=meta['tokens'],smoke=smoke,max_output_tokens=16 if smoke else 128,timing_source=TIMING,
        cache_unchanged=True,online_mask_reused=False)
    if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Record identity/protocol mismatch')
    if any(not math.isfinite(r[k]) or r[k]<=0 for k in ('ttft_seconds','generation_seconds')) or r['ttft_seconds']>r['generation_seconds']:
        raise ValueError('Invalid timing')
    if not 0<r['output_tokens']<=r['max_output_tokens'] or r['output_tokens']!=len(r['output_token_ids']) or r['finish_reason'] not in ('length','stop'):
        raise ValueError('Invalid generation')
    sample=load_sample(meta['input_path'])
    if r['references']!=sample['source_metadata']['references'] or score(sample,r['prediction'])[0]!=r['score']:raise ValueError('Scoring mismatch')
    if any(w['visible_uuid']!=r['gpu_uuid'] or not Path(w['ucm_path']).is_relative_to(ROOT/'private_ucm') for w in r['worker_imports']):raise ValueError('Worker provenance mismatch')
    dp=path.with_suffix('.diagnostics.json')
    if sha(dp)!=r['diagnostics_sha256']:raise ValueError('Diagnostics changed')
    diag=[d for w in json.loads(dp.read_text()) for d in w['diagnostics']]
    if case=='baseline':
        if diag or r['num_cached_tokens']!=0:raise ValueError('Baseline cache reuse')
        return r
    measured=log.split('MEASURE_BEGIN',1)[1].split('MEASURE_END',1)[0]
    hits=re.findall(r'request_id: measured-1-0,.*?req_stage: BlendStage\.(\w+), first chunk prefix hit: (\d+), chunks cache total hit: (\d+)',measured)
    b=meta['boundaries'];eligible=b[-2]-b[1];ratio=int(case.split('-')[-1])/100;count=int(eligible*ratio)
    if hits!=[('CACHE_BLEND',str(b[1]//64),str(eligible//64))]:raise ValueError('Incomplete measured cache hit evidence')
    if not r['cache_verification']['complete']:raise ValueError('Cache verification missing')
    layers=[d for d in diag if d.get('kind') in ('layer_counts','fusion_audit')]
    selection=[d for d in diag if d not in layers]
    if len(selection)!=1 or len(layers)!=36:raise ValueError('Missing request or layer diagnostics')
    d=selection[0]
    if d['request_id']!='measured-1-0' or d['eligible_count']!=eligible or d['selected_count']!=count:raise ValueError('Invalid selection budget')
    selected=d['selected_positions'];scores=d['scores'];chosen=set(selected)
    if len(chosen)!=count or not chosen<=set(range(b[1],b[-2])) or len(scores)!=eligible or any(not math.isfinite(x) or x<0 for x in scores):raise ValueError('Invalid selection')
    if 0<count<eligible and min(scores[i-b[1]] for i in chosen)<max(s for i,s in enumerate(scores) if i+b[1] not in chosen):raise ValueError('Selection not top-ranked')
    if case.startswith('prophetkv'):
        if d['kind']!='prophetkv_selection' or d['question_positions']!=meta['query']['positions'] or d['probe_layers']!=36 or d['alignment_count']!=36:raise ValueError('Probe mismatch')
        expected_selected=sorted(sorted(range(eligible),key=lambda i:(-scores[i],i))[:count])
        if selected!=[i+b[1] for i in expected_selected]:raise ValueError('Unstable tie handling')
    for i,event in enumerate(layers):
        if event['layer']!=f'model.layers.{i}.self_attn.attn':raise ValueError('Layer coverage mismatch')
        projection=eligible+256 if case.startswith('cacheblend') and i<=1 else count+256
        attention=eligible+256 if case.startswith('cacheblend') and i==0 else count+256
        if event['projection_tokens']!=projection or event['attention_tokens']!=attention or event['ffn_tokens']!=attention:
            raise ValueError('Actual compute count mismatch')
        if smoke and not all(event.get(k) is True for k in ('k_write_verified','v_write_verified','skipped_preserved','prefix_preserved','suffix_verified','causal_attention_verified')):
            raise ValueError('Numerical cache/model audit failed')
    return r

def valid(path,meta,case,smoke=False):
    try:
        receipt=json.loads(path.with_suffix('.validated.json').read_text())
        if any(sha(path.with_suffix(suffix))!=receipt[key] for suffix,key in [('.json','record_sha256'),('.log','log_sha256'),('.diagnostics.json','diagnostics_sha256')]):return False
        validate(path,meta,case,smoke);return True
    except (OSError,ValueError,KeyError,TypeError):return False

def all_records(p):
    rows=[]
    for meta in p['samples']:
        for case in CASES:
            path=ROOT/'records'/meta['id']/f'{case}.json'
            if valid(path,meta,case):rows.append(json.loads(path.read_text()))
    return rows

def progress(p,state):
    with PROGRESS:
        # Counts use validation receipts; full semantic validation occurs on acceptance/resume/report.
        n=sum(1 for m in p['samples'] for c in CASES if (ROOT/'records'/m['id']/f'{c}.validated.json').exists())
        dump(ROOT/'progress.json',dict(state=state,validated=n,target=200,updated_at=time.time()))
        return n

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
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:raise RuntimeError('Owned process did not exit')
    if group_alive(child.pid):raise RuntimeError('Owned engine group remains alive')

def idle(gpu):
    release(gpu)
    while processes(gpu):
        dump(ROOT/f'gpu-{gpu}.json',dict(state='waiting_for_unrelated_process',gpu=gpu,pids=processes(gpu)))
        if STOP.wait(5):raise InterruptedError('Stopped')
    check(gpu)

def cleanup_cache(gpu):
    with MUTEX:
        if gpu in ACTIVE:raise RuntimeError('Cannot clean live cache')
    cache=CACHE/f'gpu-{gpu}'
    if cache.exists():shutil.rmtree(cache)

def launch(meta,case,path,smoke=False):
    gpu=meta['physical_gpu'];cache=CACHE/f'gpu-{gpu}';path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(2):
        if STOP.is_set():raise InterruptedError('Stopped')
        idle(gpu)
        if path.exists() or path.with_suffix('.log').exists():
            archive=path.parent/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
            for f in path.parent.glob(path.stem+'.*'):
                if f.is_file():f.rename(archive/f.name)
        if attempt and case=='populate':cleanup_cache(gpu)
        cmd=[str(PYTHON),'-u',str(ROOT/'source/benchmarks/prophetkv_csv/cacheblend_prophetkv.py'),'--protocol',str(ROOT/'protocol.json'),
            '--sample',meta['input_path'],'--case',case,'--cache-dir',str(cache),'--output',str(path)]
        if smoke:cmd.append('--smoke')
        env=environment(gpu);failure=None;child=None
        try:
            with path.with_suffix('.log').open('w') as log:
                child=subprocess.Popen(cmd,cwd='/tmp',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with MUTEX:ACTIVE[gpu]=child
                dump(ROOT/f'gpu-{gpu}.json',dict(state='running',pid=child.pid,pgid=child.pid,command=cmd,gpu=gpu,
                    uuid=GPU_UUIDS[gpu],sample=meta['id'],case=case,smoke=smoke,attempt=attempt,started_at=time.time()))
                print(f'start gpu={gpu} sample={meta["id"]} case={case} smoke={smoke} pid={child.pid}',flush=True)
                last=time.monotonic();size=0
                while child.poll() is None:
                    content=path.with_suffix('.log').read_text(errors='replace')
                    if len(content)!=size:last=time.monotonic();size=len(content)
                    if STOP.is_set() or FATAL.search(content) or time.monotonic()-last>600:
                        failure='stop' if STOP.is_set() else 'fatal log or 600-second watchdog';terminate(child);break
                    STOP.wait(1)
                code=child.wait()
        finally:
            if child is not None:terminate(child)
            with MUTEX:ACTIVE.pop(gpu,None)
            dump(ROOT/f'gpu-{gpu}.json',dict(state='exited',gpu=gpu,sample=meta['id'],case=case,ended_at=time.time()))
            if not STOP.is_set():reserve(gpu)
        if STOP.is_set():raise InterruptedError('Stopped')
        try:
            if code or failure:raise RuntimeError(f'Worker failed: {code}, {failure}')
            if case=='populate':
                r=json.loads(path.read_text());log=path.with_suffix('.log').read_text()
                if not r['cache_verification']['complete'] or FATAL.search(log) or 'WORKER_COMPLETE' not in log:raise ValueError('Invalid cache build')
            else:
                validate(path,meta,case,smoke)
                dump(path.with_suffix('.validated.json'),dict(record_sha256=sha(path),log_sha256=sha(path.with_suffix('.log')),
                    diagnostics_sha256=sha(path.with_suffix('.diagnostics.json')),validated_at=time.time()))
            print(f'validated gpu={gpu} sample={meta["id"]} case={case} smoke={smoke}',flush=True)
            return
        except (OSError,ValueError,KeyError,RuntimeError) as error:
            print(f'attempt_failed {meta["id"]} {case}: {error}',flush=True)
            if attempt==1:raise

def equivalence(directory):
    import torch
    a=torch.load(directory/'baseline.model-audit.pt',weights_only=True)[0]
    b=torch.load(directory/'prophetkv-100.model-audit.pt',weights_only=True)[0]
    if set(a)!=set(range(36)) or set(a)!=set(b):raise ValueError('Missing full-model audit layers')
    errors=[]
    for i in range(36):
        error=(a[i]-b[i]).abs()
        # BF16 kernels/position relocation introduce rounding. Check relative RMS
        # and p99, retaining max errors; exact equality is not asserted.
        rel=float(error.square().mean().sqrt()/a[i].square().mean().sqrt().clamp_min(1e-6))
        passed=rel<.03
        errors.append(dict(layer=i,relative_rms=rel,max_abs_error=float(error.max()),passed=passed))
    receipt=dict(complete=True,passed=all(r['passed'] for r in errors),control='100% vs no-cache',
        compared='all 256 suffix hidden states after every decoder layer',tolerance='relative RMS < 0.03 per layer',layers=errors)
    dump(directory/'equivalence.json',receipt)
    if not receipt['passed']:raise ValueError('100% full-model numerical comparison failed')

def numerical():
    gate=ROOT/'gates/numerical.json'
    directory=ROOT/'numerical';directory.mkdir(exist_ok=True)
    if gate.exists():
        receipt=json.loads(gate.read_text())
        if (receipt.get('complete') and receipt.get('protocol_sha256')==sha(ROOT/'protocol.json')
            and receipt['cpu_sha256']==sha(directory/'cpu.log')
            and receipt['gpu_sha256']==sha(directory/'gpu.log')):return
    with (directory/'cpu.log').open('w') as log:
        subprocess.run([str(PYTHON),'-m','unittest','discover','-s',str(ROOT/'source/benchmarks/prophetkv_csv'),'-p','test_prophetkv.py','-v'],
            cwd='/tmp',env=cpu_env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    idle(3)
    try:
        with (directory/'gpu.log').open('w') as log:
            subprocess.run([str(PYTHON),str(ROOT/'source/benchmarks/prophetkv_csv/test_prophetkv.py'),'--gpu'],cwd='/tmp',env=environment(3),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    finally:reserve(3)
    result=json.loads((directory/'gpu.log').read_text().splitlines()[-1])
    if not result['complete']:raise ValueError('GPU numerical gate failed')
    dump(gate,dict(complete=True,protocol_sha256=sha(ROOT/'protocol.json'),cpu_sha256=sha(directory/'cpu.log'),gpu_sha256=sha(directory/'gpu.log'),gpu=result))

def smoke(p,length,chunk):
    directory=ROOT/f'smoke/{length}-{chunk}';gate=directory/'validation.json'
    if gate.exists():
        receipt=json.loads(gate.read_text())
        if (receipt.get('complete') and receipt.get('protocol_sha256')==sha(ROOT/'protocol.json')
            and all((ROOT/name).exists() and sha(ROOT/name)==digest for name,digest in receipt['artifacts'].items())):return
    progress(p,f'model_audits_{length}_{chunk}')
    meta=next(m for m in p['samples'] if m['context_target']==length and m['chunk_size']==chunk and m['label']=='cwe' and m['source_row']==CHUNKS.index(chunk)%2)
    gpu=meta['physical_gpu']
    cleanup_cache(gpu)
    try:
        launch(meta,'populate',directory/'populate.json')
        for case in ('baseline','prophetkv-20',*CONTROLS,*[c for c in CASES if c not in ('baseline','prophetkv-20')]):
            path=directory/f'{case}.json'
            if not valid(path,meta,case,True):launch(meta,case,path,True)
        equivalence(directory)
        dump(gate,dict(complete=True,protocol_sha256=sha(ROOT/'protocol.json'),cases=[*CASES,*CONTROLS],
            numerical_equivalence_sha256=sha(directory/'equivalence.json'),
            artifacts={str(f.relative_to(ROOT)):sha(f) for f in directory.rglob('*')
                if f.is_file() and f!=gate and 'attempts' not in f.parts}))
    finally:cleanup_cache(gpu)

def order(meta,pilot):
    cases=['baseline','prophetkv-20',*[c for c in CASES if c not in ('baseline','prophetkv-20')]] if pilot else list(CASES)
    offset=(meta['source_row']+CHUNKS.index(meta['chunk_size']))%len(cases)
    return cases[offset:]+cases[:offset]


def per_gpu(p,gpu,samples,pilot):
    for meta in samples:
        if meta['physical_gpu']!=gpu:continue
        pending=[case for case in order(meta,pilot) if case in meta['required_cases'] and not valid(ROOT/'records'/meta['id']/f'{case}.json',meta,case)]
        if not pending:continue
        if STOP.is_set():raise InterruptedError('Stopped')
        cleanup_cache(gpu)
        try:
            if any(c!='baseline' for c in pending):launch(meta,'populate',ROOT/'cache-builds'/meta['id']/f'{time.time_ns()}.json')
            for case in pending:
                launch(meta,case,ROOT/'records'/meta['id']/f'{case}.json')
                progress(p,'pilot' if pilot else 'sweep')
        finally:cleanup_cache(gpu)

def phase(p,samples,name,pilot=False):
    dump(ROOT/'phase.json',dict(phase=name,started_at=time.time(),sample_ids=[m['id'] for m in samples]))
    with (ROOT/'phases.jsonl').open('a') as out:out.write(json.dumps(dict(phase=name,event='start',at=time.time()))+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(per_gpu,p,g,samples,pilot) for g in GPU_UUIDS}
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                if future.exception():STOP.set()
                future.result()
    with (ROOT/'phases.jsonl').open('a') as out:out.write(json.dumps(dict(phase=name,event='end',at=time.time()))+'\n')

def stop(*_):STOP.set()

def run(pilot_only=False):
    with lock():
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Supervisor must be CPU-only')
        from prophetkv_report import report
        p=verify();state='failed'
        status=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),identity=identity(os.getpid()),command=sys.argv,started_at=time.time(),state='running')
        dump(ROOT/'supervisor.json',status)
        try:
            dependency=REPO/'.results/prophetkv-qwen3-ruler10-64k-chunks-gpu34-20260918'
            progress(p,'waiting_for_active_64k_sweep')
            print('Queued: waiting for the existing 1200-request GPU3/4 sweep to complete and release its lock.',flush=True)
            while True:
                ready=False
                try:
                    certificate=json.loads((dependency/'final/validation.json').read_text())
                    cleanup=json.loads((dependency/'cleanup.json').read_text())
                    supervisor=json.loads((dependency/'supervisor.json').read_text())
                    exited=identity(supervisor['pid'])!=supervisor['identity']
                    with (dependency/'run.lock').open('a') as lockfile:
                        fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        ready=(certificate.get('complete') and certificate.get('validated')==1200
                               and certificate.get('cleanup_sha256')==sha(dependency/'cleanup.json')
                               and cleanup.get('complete') and exited)
                except (OSError,ValueError,KeyError):pass
                if ready:break
                if STOP.wait(10):raise InterruptedError('Stopped while waiting')
            for gpu in GPU_UUIDS:reserve(gpu)
            progress(p,'numerical_validation')
            numerical()
            from prophetkv_report import report
            for length in LENGTHS:
                smoke(p,length,512)
                samples=[m for m in p['samples'] if m['context_target']==length]
                phase(p,samples,f'csv-backfill-{length}-512')
                report(all_records(p),'sweep')
            verify()
            rows=all_records(p)
            if len(rows)!=200:raise RuntimeError('Incomplete final matrix')
            report(rows,'complete',True);state='complete'
        except BaseException as error:
            STOP.set();state='stopped' if isinstance(error,InterruptedError) else 'failed'
            dump(ROOT/'failure.json',dict(error=repr(error),at=time.time()))
            from prophetkv_report import report
            report(all_records(p),state)
            raise
        finally:
            STOP.set()
            with MUTEX:children=list(ACTIVE.values())
            for child in children:terminate(child)
            for gpu in GPU_UUIDS:
                release(gpu);cleanup_cache(gpu)
            dump(ROOT/'cleanup.json',dict(complete=True,state=state,at=time.time(),owned_workers_exited=True,
                placeholders_released=True,caches_removed=not any(CACHE.glob('gpu-*'))))
            progress(p,state);dump(ROOT/'supervisor.json',{**status,'state':state,'ended_at':time.time()})
            if state=='complete':dump(ROOT/'final/validation.json',dict(complete=True,validated=200,
                protocol_sha256=sha(ROOT/'protocol.json'),cleanup_sha256=sha(ROOT/'cleanup.json'),
                gates=hashes(ROOT/'gates'),smoke_certificates={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'smoke').glob('*/validation.json')},artifacts=hashes(ROOT/'final')))

def detach(pilot_only=False):
    with lock():verify()
    cmd=['nohup',str(PYTHON),'-u',str(Path(__file__).resolve()),'pilot' if pilot_only else 'run']
    with (ROOT/'supervisor.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=REPO,env=cpu_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/'nohup-launch.json',dict(pid=child.pid,command=cmd,start_new_session=True,stdin='/dev/null',at=time.time()))
    print(f'Detached supervisor PID {child.pid}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','pilot','run','detach','resume','status','report'))
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command in ('detach','resume'):detach()
    elif args.command=='pilot':run(True)
    elif args.command=='run':run()
    elif args.command=='status':print((ROOT/'progress.json').read_text())
    else:
        with lock():
            from prophetkv_report import report
            p=verify();report(all_records(p),json.loads((ROOT/'progress.json').read_text())['state'])
