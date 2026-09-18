#!/usr/bin/env python3
"""Append niah_multivalue without changing the original pinned experiment."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_prophetkv as base
import prophetkv_data as data
import prophetkv_report as reporting
from prophetkv_common import dump, sha, load_sample

ROOT=base.ROOT
EXT=ROOT/'scope-multivalue'
SOURCE='benchmarks/prophetkv64k/run_with_multivalue.py'
TASKS=(*data.TASKS,'niah_multivalue')
TARGET=1200
original_verify=base.verify
original_report=reporting.report


def prepare():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only preparation required')
    EXT.mkdir(exist_ok=True)
    if (EXT/'amendment.json').exists():return verify()
    p=original_verify()
    previous=data.TASKS
    try:
        data.TASKS=('niah_multivalue',)
        samples,datasets=data.prepare_samples(sha,dump)
    finally:data.TASKS=previous
    assert len(samples)==40
    for m in samples:
        s=load_sample(m['input_path'])
        assert s['label']=='niah_multivalue' and s['source_row'] in range(10)
        assert s['physical_gpu']==3+s['source_row']%2
        assert s['boundaries'][1]==((s['chunk_size']+63)//64)*64
        assert all(s['tokens']-256<=q<s['tokens'] for q in s['query']['positions'])
    target=EXT/'source'/SOURCE;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(base.REPO/SOURCE,target)
    amendment=dict(schema_version=1,created_at=time.time(),tasks=list(TASKS),target=TARGET,
        added_task='niah_multivalue',source_rows=list(range(10)),added_requests=200,
        original_protocol_sha256=sha(ROOT/'protocol.json'),original_execution_sha256=sha(ROOT/'execution.json'),
        source_sha256=sha(base.REPO/SOURCE),samples=samples,dataset_hashes=datasets,
        gpu_uuids=base.GPU_UUIDS,
        inference='Original pinned worker, private UCM, checkpoint and generation settings unchanged.',
        provenance='Records retain original protocol hash; added input hashes and scope are bound by this amendment.',
        schedule='Append niah_multivalue after the original five tasks within each chunk phase.',
        timing_caveat='Supervisor restarted for scope amendment; interrupted cases rerun. Record transition phase for timing interpretation.')
    dump(EXT/'amendment.json',amendment)
    dump(EXT/'combined-prompt-manifest.json',p['samples']+samples)
    print('Prepared 40 additional frozen layouts and 200 requests.',flush=True)
    return verify()


def verify():
    p=original_verify()
    a=json.loads((EXT/'amendment.json').read_text())
    assert a['original_protocol_sha256']==sha(ROOT/'protocol.json')
    assert a['original_execution_sha256']==sha(ROOT/'execution.json')
    assert a['source_sha256']==sha(base.REPO/SOURCE)==sha(EXT/'source'/SOURCE)
    assert a['target']==TARGET and a['tasks']==list(TASKS)
    assert a['gpu_uuids']=={str(k):v for k,v in base.GPU_UUIDS.items()}
    for path,digest in a['dataset_hashes'].items():assert sha(path)==digest
    for m in a['samples']:
        assert sha(m['input_path'])==m['input_sha256']
        load_sample(m['input_path'])
    samples=p['samples']+a['samples']
    assert len(samples)==len({m['id'] for m in samples})==240
    assert len(samples)*len(base.CASES)==TARGET
    for m in a['samples']:
        assert m['label']=='niah_multivalue' and m['context_target']==65536
        assert m['physical_gpu']==3+m['source_row']%2
    preserved=EXT/'preserved-records.json'
    if preserved.exists():
        for path,digest in json.loads(preserved.read_text())['hashes'].items():assert sha(ROOT/path)==digest
    return {**p,'samples':samples,'tasks':list(TASKS),'measured_requests':TARGET,'scope_amendment_sha256':sha(EXT/'amendment.json')}


def progress(p,state):
    with base.PROGRESS:
        n=sum((ROOT/'records'/m['id']/f'{c}.validated.json').exists() for m in p['samples'] for c in base.CASES)
        dump(ROOT/'progress.json',dict(state=state,validated=n,target=TARGET,updated_at=time.time(),
            scope_amendment='scope-multivalue/amendment.json'))
        return n


def report(rows,state,final=False):
    reporting.TASKS=TASKS
    directory=original_report(rows,state,final)
    for path in (directory/'REPORT.md',ROOT/'REPORT.md'):
        text=path.read_text().replace('/1,000 validated measurements','/1,200 validated measurements')
        text+='\nScope amendment adds ten niah_multivalue source prompts across all four layouts and five methods. Original measured records are preserved. Supervisor restart and concurrency phases are recorded under scope-multivalue/.\n'
        path.write_text(text)
    return directory


base.progress=progress


def run():
    with base.lock():
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only supervisor required')
        signal.signal(signal.SIGTERM,base.stop);signal.signal(signal.SIGINT,base.stop)
        p=verify();state='failed'
        status=dict(pid=os.getpid(),pgid=os.getpgrp(),sid=os.getsid(0),identity=base.identity(os.getpid()),
            command=sys.argv,started_at=time.time(),state='running',scope_amendment_sha256=sha(EXT/'amendment.json'))
        dump(ROOT/'supervisor.json',status)
        try:
            for gpu in base.GPU_UUIDS:base.reserve(gpu)
            progress(p,'checking_preserved_validation')
            base.numerical()
            for chunk in base.CHUNKS:base.smoke(p,65536,chunk)
            for chunk in base.CHUNKS:
                samples=[m for m in p['samples'] if m['chunk_size']==chunk]
                base.phase(p,samples,f'sweep-multivalue-added-65536-{chunk}')
                report(base.all_records(p),'sweep')
            verify();rows=base.all_records(p)
            if len(rows)!=TARGET:raise RuntimeError('Incomplete expanded matrix')
            report(rows,'complete',True);state='complete'
        except BaseException as error:
            base.STOP.set();state='stopped' if isinstance(error,InterruptedError) else 'failed'
            dump(ROOT/'failure.json',dict(error=repr(error),at=time.time()))
            report(base.all_records(p),state)
            raise
        finally:
            base.STOP.set()
            with base.MUTEX:children=list(base.ACTIVE.values())
            for child in children:base.terminate(child)
            for gpu in base.GPU_UUIDS:base.release(gpu);base.cleanup_cache(gpu)
            dump(ROOT/'cleanup.json',dict(complete=True,state=state,at=time.time(),owned_workers_exited=True,
                placeholders_released=True,caches_removed=not any(base.CACHE.glob('gpu-*'))))
            progress(p,state);dump(ROOT/'supervisor.json',{**status,'state':state,'ended_at':time.time()})
            if state=='complete':
                dump(ROOT/'final/validation.json',dict(complete=True,validated=TARGET,protocol_sha256=sha(ROOT/'protocol.json'),
                    scope_amendment_sha256=sha(EXT/'amendment.json'),cleanup_sha256=sha(ROOT/'cleanup.json'),
                    gates=base.hashes(ROOT/'gates'),smoke_certificates={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'smoke').glob('*/validation.json')},
                    artifacts=base.hashes(ROOT/'final')))


def detach():
    with base.lock():
        p=verify()
        # Capture prior state and receipt-bound records once, after old workers exit.
        if not (EXT/'preserved-records.json').exists():
            saved={};count=0
            for m in p['samples']:
                for case in base.CASES:
                    path=ROOT/'records'/m['id']/f'{case}.json'
                    if base.valid(path,m,case):
                        count+=1
                        for suffix in ('.json','.log','.diagnostics.json','.validated.json'):
                            f=path.with_suffix(suffix);saved[str(f.relative_to(ROOT))]=sha(f)
            dump(EXT/'preserved-records.json',dict(validated=count,hashes=saved,at=time.time()))
            history=EXT/'previous-supervisor';history.mkdir(exist_ok=True)
            for name in ('supervisor.json','progress.json','cleanup.json','failure.json','nohup-launch.json','detachment.json','gpu-3.json','gpu-4.json'):
                f=ROOT/name
                if f.exists():shutil.copy2(f,history/name)
        for name in ('failure.json','cleanup.json'):
            f=ROOT/name
            if f.exists():f.unlink()
        progress(p,'expanded_scope_queued')
    cmd=['nohup',str(base.PYTHON),'-u',str(Path(__file__).resolve()),'run']
    with (ROOT/'supervisor.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=base.REPO,env=base.cpu_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(ROOT/'nohup-launch.json',dict(pid=child.pid,command=cmd,start_new_session=True,stdin='/dev/null',at=time.time(),
        scope_amendment_sha256=sha(EXT/'amendment.json')))
    print(f'Detached expanded supervisor PID {child.pid}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','detach','resume','run','status','report'))
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command in ('detach','resume'):detach()
    elif args.command=='run':run()
    elif args.command=='status':print((ROOT/'progress.json').read_text())
    else:
        with base.lock():
            p=verify();report(base.all_records(p),json.loads((ROOT/'progress.json').read_text())['state'])
