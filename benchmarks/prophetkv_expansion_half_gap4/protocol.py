"""Independent protocol, frozen source snapshots and preservation of previous runs."""
import json
import shutil
import time
from pathlib import Path
from importlib.metadata import version
from prophetkv_gpu import ROOT, REPO, GPU_UUIDS, identity
from prophetkv_common import sha,dump,load_sample
from settings import PARENT,PRIOR_EXTENSION,CASES,PARAMETERS,MAX_OUTPUT_TOKENS
import data


def prepare_protocol(source, hashes):
    inputs=data.verify()
    parent=json.loads((PARENT/'protocol.json').read_text())
    for old in (PARENT,PRIOR_EXTENSION):
        for name in ('supervisor','reporter'):
            state=json.loads((old/(name+'.json')).read_text())
            if identity(state['pid'])==state['identity']:raise RuntimeError('Previous process still live')
        if not json.loads((old/'final/validation.json').read_text())['complete']:raise ValueError('Parent incomplete')
    if hashes(data.MODEL)!=parent['model_hashes']:raise ValueError('Checkpoint changed')
    for path,digest in parent['runtime_file_hashes'].items():
        if sha(path)!=digest:raise ValueError('Patched runtime changed')
    preserved={}
    for old in (PARENT,PRIOR_EXTENSION):
        for f in sorted(old.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and 'interim' not in f.parts and f.suffix not in ('.pyc','.lock'):
                preserved[str(f)]=sha(f)
    for name in ('prophetkv_with_expansion_results.txt','prophetkv_with_expansion_ratios_results.txt'):
        preserved[str(REPO/name)]=sha(REPO/name)
    dump(ROOT/'preserved-prior.json',dict(hashes=preserved,created_at=time.time()))
    private=ROOT/'private_ucm/ucm'
    shutil.copytree(PARENT/'private_ucm/ucm',private,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    from private_runtime import adapt
    adapt(private)
    shutil.copytree(source,ROOT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    samples=[]
    for meta in inputs['samples']:
        path=ROOT/meta['input_file'];sample=load_sample(path)
        samples.append({k:v for k,v in sample.items() if k not in ('token_ids','formatted_text')}
            |dict(input_path=str(path),input_sha256=sha(path)))
    p=dict(study='Qwen3-4B-Instruct expansion half-budget gap4 output256',schema_version=1,
        model=str(data.MODEL),model_revision=data.MODEL.name,model_hashes=parent['model_hashes'],
        runtime_file_hashes=parent['runtime_file_hashes'],runtime_versions={n:version(n) for n in parent['runtime_versions']},
        sources=hashes(ROOT/'source'),private_ucm_hashes=hashes(private),
        parent_private_ucm_hashes=parent['private_ucm_hashes'],
        private_adaptation='variable fresh suffix >=256 to preserve full question/choices; unchanged operations at 256',
        input_manifest_sha256=sha(ROOT/'input-manifest.json'),preserved_prior_sha256=sha(ROOT/'preserved-prior.json'),
        samples=samples,measured_requests=len(samples)*len(CASES),max_model_len=inputs['max_model_len'],
        max_output_tokens=MAX_OUTPUT_TOKENS,tensor_parallel_size=1,dtype='bfloat16',eager=True,
        thinking_enabled=False,decoding=dict(temperature=0.,top_p=1.),rope_scaling=None,
        cases=list(CASES),case_parameters=PARAMETERS,gpu_uuids=GPU_UUIDS,
        phases={str(gpu):list(CASES[(gpu-1)%len(CASES):]+CASES[:(gpu-1)%len(CASES)]) for gpu in GPU_UUIDS},
        qualification='explicitly waived; no model smoke/qualification requests',
        engine_policy='one persistent engine/GPU/method, 28 normal initializations plus bounded retries',
        fresh_suffix_policy=inputs['fresh_suffix_policy'],chunk_size=4096,
        longbench_eligibility=inputs['longbench_eligibility'],ruler_selection=inputs['ruler_selection'],
        longbench_count=inputs['longbench_count'],truncated=False,
        timing_source='engine_step_first_token_monotonic',timing=parent['timing'],
        storage='buffered local warm cache; offline construction/readiness recorded separately',
        timing_cohort_note='All measurements are fresh; configuration order rotates across GPUs. Concurrent work and method phases can affect timings.',
        dummy_holders=False,created_at=time.time())
    dump(ROOT/'protocol.json',p)
    return p


def verify_preserved():
    saved=json.loads((ROOT/'preserved-prior.json').read_text())
    for name,digest in saved['hashes'].items():
        if sha(name)!=digest:raise ValueError('Previous artifact changed: '+name)
    return len(saved['hashes'])


def verify_protocol(source,hashes):
    p=json.loads((ROOT/'protocol.json').read_text())
    if sha(ROOT/'preserved-prior.json')!=p['preserved_prior_sha256']:raise ValueError('Preservation manifest changed')
    if sha(ROOT/'input-manifest.json')!=p['input_manifest_sha256']:raise ValueError('Input manifest changed')
    if hashes(source)!=p['sources'] or hashes(ROOT/'source')!=p['sources']:raise ValueError('Frozen sources changed')
    if hashes(ROOT/'private_ucm/ucm')!=p['private_ucm_hashes']:raise ValueError('Private runtime changed')
    for path,digest in p['runtime_file_hashes'].items():
        if sha(path)!=digest:raise ValueError('Patched runtime changed')
    for meta in p['samples']:
        if sha(meta['input_path'])!=meta['input_sha256']:raise ValueError('Frozen input changed')
    if p['cases']!=list(CASES) or p['case_parameters']!=PARAMETERS or p['max_output_tokens']!=256:
        raise ValueError('Configuration changed')
    if p['measured_requests']!=len(p['samples'])*7 or len(p['samples'])!=700+p['longbench_count']:
        raise ValueError('Scope changed')
    return p
