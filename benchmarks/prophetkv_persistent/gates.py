"""Fail-closed numerical, isolation, endurance and restart qualification."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time
import run as r
from prophetkv_common import dump,sha
from cacheblend_prophetkv import CASES


def command(cmd,gpu,log_path):
    r.idle(gpu);log_path.parent.mkdir(parents=True,exist_ok=True)
    env=r.environment(gpu,log_path.with_suffix('.scheduler.json'))
    env['PROPHETKV_REFERENCE_HELPERS']=str(r.ROOT/'source')
    with log_path.open('w') as log:
        child=subprocess.Popen(cmd,cwd='/tmp',env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with r.MUTEX:r.ACTIVE[gpu]=child
        try:
            last=time.monotonic();size=0
            while child.poll() is None:
                content=log_path.read_text(errors='replace')
                if len(content)!=size:last=time.monotonic();size=len(content)
                if r.STOP.is_set() or r.val.FATAL.search(content) or time.monotonic()-last>600:
                    r.terminate(child);break
                r.STOP.wait(1)
            if child.wait()!=0:raise RuntimeError('Gate command failed: '+str(log_path))
        finally:
            r.terminate(child)
            with r.MUTEX:r.ACTIVE.pop(gpu,None)


def entry(meta,path):return dict(meta,output=str(path))


def signature(path):
    record=json.loads(path.read_text())
    diagnostics=json.loads(path.with_suffix('.diagnostics.json').read_text())
    events=[dict(d) for w in diagnostics for d in w['diagnostics']]
    for d in events:
        for key in ('request_id','probe_seconds'):d.pop(key,None)
    return {k:record[k] for k in ('output_token_ids','score','num_cached_tokens')},events


def qualification(p,gpu,audits):
    import torch
    directory=r.ROOT/'gates'/f'gpu-{gpu}';directory.mkdir(parents=True,exist_ok=True)
    numerical=directory/'numerical.json'
    if not numerical.exists():
        command([str(r.PYTHON),str(r.ROOT/'source/test_prophetkv.py'),'--gpu'],gpu,directory/'numerical.log')
        dump(numerical,dict(complete=True,log_sha256=sha(directory/'numerical.log')))
    for meta in audits:
        if meta['physical_gpu']!=gpu:continue
        folder=directory/'smoke'/meta['id']
        for case in (*CASES,'prophetkv-0','prophetkv-100'):
            r.session(p,gpu,case,[entry(meta,folder/(case+'.json'))],phase=-1,smoke=True)
        native=folder/'native-prefix-reference.json'
        if not native.exists():
            command([str(r.PYTHON),str(r.ROOT/'source/dense_prefix_reference.py'),
                '--protocol',str(r.ROOT/'protocol.json'),'--sample',meta['input_path'],'--output',str(native)],gpu,native.with_suffix('.log'))
        reference=json.loads(native.read_text())
        if reference['num_cached_tokens']!=meta['boundaries'][1] or reference['audit_sha256']!=sha(native.with_suffix('.model-audit.pt')):
            raise ValueError('Native reference invalid')
        load=lambda case:torch.load(folder/(case+'.model-audit.pt'),weights_only=True)[0]
        sparse=load('prophetkv-100');dense=load('native-prefix-reference');full=load('baseline')
        if set(sparse)!=set(range(36)) or set(dense)!=set(sparse):raise ValueError('Missing reference layers')
        rows=[]
        for i in range(36):
            if not torch.equal(sparse[i],dense[i]):raise ValueError('100% native dense-prefix mismatch')
            rows.append(dict(layer=i,bitwise_equal=True,no_cache_relative_rms=float(
                (sparse[i]-full[i]).square().mean().sqrt()/full[i].square().mean().sqrt().clamp_min(1e-6))))
        dump(folder/'equivalence.json',dict(complete=True,layers=rows,fresh_suffix_tokens=meta['fresh_suffix_tokens']))
    samples=[m for m in p['samples'] if m['physical_gpu']==gpu]
    if gpu<=3:
        case=CASES[gpu-1]
        short=min(samples,key=lambda m:m['tokens']);long=max(samples,key=lambda m:m['tokens'])
        sequence=[short,long,short]+samples[:17]
        entries=[entry(m,directory/'endurance'/f'{i:02d}'/(case+'.json')) for i,m in enumerate(sequence)]
        r.session(p,gpu,case,entries,phase=-2)
        records=[json.loads(Path(e['output']).read_text()) for e in entries]
        if len({x['session_id'] for x in records})!=1:raise ValueError('Endurance requires one engine initialization')
        if signature(Path(entries[0]['output']))!=signature(Path(entries[2]['output'])):
            raise ValueError('A-B-A isolation mismatch')
        for i in (0,1):
            isolated=entry(sequence[i],directory/'isolated'/str(i)/(case+'.json'))
            r.session(p,gpu,case,[isolated],phase=-3)
            if signature(Path(entries[i]['output']))!=signature(Path(isolated['output'])):
                raise ValueError('Persistent/isolated tokens, scores, selections or counts differ')
        # Allocation is read after full retirement; compare at matched prompt sizes
        # and require no retained tensors across the twenty-request sequence.
        allocated=[x['retirement']['workers'][0]['allocated_bytes'] for x in records]
        if max(allocated)-min(allocated)>16*1024*1024:
            raise ValueError('GPU live allocations accumulate after retirement')
        if any(x['cache_files_after_retirement'] or x['cache_disk_bytes']>64*1024**3 for x in records):
            raise ValueError('Unbounded storage')
        dump(directory/'endurance.json',dict(complete=True,case=case,requests=20,
            engine_initializations=1,allocated_bytes=allocated,A_B_A_equal=True,
            isolated_equal=True,short_tokens=short['tokens'],long_tokens=long['tokens']))
    else:
        entries=[entry(m,directory/'failure-resume'/str(i)/'baseline.json') for i,m in enumerate(samples[:2])]
        r.session(p,gpu,'baseline',entries,phase=-4,fail_after=1)
        records=[json.loads(Path(e['output']).read_text()) for e in entries]
        if records[0]['session_id']==records[1]['session_id'] or records[1]['engine_start_count']!=2:
            raise ValueError('Failure/resume was not exercised')
        dump(directory/'failure-resume.json',dict(complete=True,accepted_request_preserved=True,
            engine_starts=2,requests=2,records={e['output']:sha(e['output']) for e in entries}))


def gates(p):
    directory=r.ROOT/'gates';directory.mkdir(exist_ok=True)
    certificate=directory/'validation.json'
    if certificate.exists():
        c=json.loads(certificate.read_text())
        if c['protocol_sha256']!=sha(r.ROOT/'protocol.json'):raise ValueError('Gate protocol changed')
        for path,digest in c['artifacts'].items():
            if sha(directory/path)!=digest:raise ValueError('Gate artifact changed')
        return
    cpu=directory/'cpu.log'
    with cpu.open('w') as log:
        subprocess.run([str(r.PYTHON),'-m','unittest','discover','-s',str(r.ROOT/'source'),'-p','test_*.py','-v'],
            cwd='/tmp',env=r.cpu_env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    ruler=[m for m in p['samples'] if m['dataset']=='ruler'];lb=[m for m in p['samples'] if m['dataset']=='longbench_v2']
    audits=list({m['id']:m for m in (max(ruler,key=lambda m:m['tokens']),
        max(lb,key=lambda m:m['tokens']),max(lb,key=lambda m:m['fresh_suffix_tokens']))}.values())
    if max(m['fresh_suffix_tokens'] for m in audits)!=871:raise ValueError('Missing 871-token query regression')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(qualification,p,gpu,audits) for gpu in r.GPU_UUIDS}
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                if f.exception():r.STOP.set()
                f.result()
    dump(certificate,dict(complete=True,protocol_sha256=sha(r.ROOT/'protocol.json'),
        artifacts=r.hashes(directory),at=time.time(),audited_prompts=[m['id'] for m in audits],
        endurance_requests_per_configuration=20,normal_sweep_engine_initializations=12))
