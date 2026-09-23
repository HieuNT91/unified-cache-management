"""Freeze an isolated extension and retain the completed parent byte-for-byte."""
import json
import shutil
import time
from pathlib import Path
from importlib.metadata import version
from prophetkv_common import sha, dump, load_sample
from prophetkv_gpu import ROOT, REPO, GPU_UUIDS, identity
from settings import PARENT, CASES, PARAMETERS


def verify_parent_certificate():
    certificate=json.loads((PARENT/'final/validation.json').read_text())
    if not certificate['complete'] or certificate['validated']!=400:
        raise ValueError('Parent experiment is incomplete')
    for name,digest in certificate['artifacts'].items():
        if sha(PARENT/'final'/name)!=digest:raise ValueError('Parent report changed: '+name)
    if sha(PARENT/'protocol.json')!=certificate['protocol_sha256']:
        raise ValueError('Parent protocol changed')
    if sha(PARENT/'cleanup.json')!=certificate['cleanup_sha256']:
        raise ValueError('Parent cleanup changed')
    for name in ('supervisor','reporter'):
        state=json.loads((PARENT/(name+'.json')).read_text())
        if identity(state['pid'])==state['identity']:
            raise RuntimeError('Parent '+name+' remains live')
    if not json.loads((PARENT/'cleanup.json').read_text())['owned_workers_exited']:
        raise ValueError('Parent engines have not exited')
    rows=json.loads((PARENT/'final/records.json').read_text())
    if len(rows)!=400 or len({(r['sample_id'],r['case']) for r in rows})!=400:
        raise ValueError('Parent records are incomplete or duplicated')
    return rows


def retained_rows():
    rows=verify_parent_certificate()
    for row in rows:
        path=PARENT/'records'/row['sample_id']/(row['case']+'.json')
        receipt=json.loads(path.with_suffix('.validated.json').read_text())
        for suffix,key in (('.json','record_sha256'),('.log','log_sha256'),('.diagnostics.json','diagnostics_sha256')):
            if sha(path.with_suffix(suffix))!=receipt[key]:raise ValueError('Retained record changed')
        if json.loads(path.read_text())!=row:raise ValueError('Parent raw record/report disagreement')
    return rows


def prepare_extension(source, hashes):
    from dataclasses import asdict
    from validation import ExpansionConfig
    retained=retained_rows()
    parent=json.loads((PARENT/'protocol.json').read_text())
    for path,digest in parent['runtime_file_hashes'].items():
        if sha(path)!=digest:raise ValueError('Patched runtime changed: '+path)
    if hashes(Path(parent['model']))!=parent['model_hashes']:
        raise ValueError('Checkpoint changed')
    if hashes(PARENT/'private_ucm/ucm')!=parent['private_ucm_hashes']:
        raise ValueError('Parent private runtime changed')
    for case,parameters in PARAMETERS.items():
        if asdict(ExpansionConfig(**parameters))!=parameters:raise ValueError('Invalid parameters: '+case)
    for meta in parent['samples']:
        sample=load_sample(meta['input_path'])
        if sha(meta['input_path'])!=meta['input_sha256'] or sample['tokens']>65536:
            raise ValueError('Frozen input changed or exceeds limit')
    private=ROOT/'private_ucm/ucm'
    shutil.copytree(PARENT/'private_ucm/ucm',private,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    # Inference, selector, connector and retirement code are byte-identical.
    if hashes(private)!=parent['private_ucm_hashes']:raise ValueError('Private runtime copy differs')
    shutil.copytree(source,ROOT/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    preserved=hashes(PARENT)
    preserved['../parent-root-report.txt']=sha(REPO/'prophetkv_with_expansion_results.txt')
    dump(ROOT/'preserved-parent.json',dict(parent=str(PARENT),hashes=preserved,records=len(retained),at=time.time()))
    p={key:value for key,value in parent.items() if key not in
       ('sources','private_ucm_hashes','cases','expansion_parameters','instrumentation_recovery','phases')}
    p.update(study='ProphetKV expansion 30/40/50 percent extension',
             parent=str(PARENT),parent_protocol_sha256=sha(PARENT/'protocol.json'),
             preserved_parent_sha256=sha(ROOT/'preserved-parent.json'),
             sources=hashes(ROOT/'source'),private_ucm_hashes=hashes(private),
             runtime_versions={n:version(n) for n in parent['runtime_versions']},
             cases=list(CASES),case_parameters=PARAMETERS,measured_requests=600,
             retained_requests=400,combined_requests=1000,
             phases={str(gpu):list(CASES[(gpu-1)%3:]+CASES[:(gpu-1)%3]) for gpu in GPU_UUIDS},
             engine_policy='one persistent engine per GPU/configuration; 12 normal initializations',
             retained_record_paths=[str(PARENT/'records'/r['sample_id']/(r['case']+'.json')) for r in retained],
             timing_cohort_note='Baseline and 20% retained from the completed parent; 30/40/50% measured later. Same prompts/GPU assignments; timing cohorts differ.',
             parameter_policy='User explicitly selected anchor ratios .225/.30/.375 for total ratios .30/.40/.50; all other parameters unchanged.',
             created_at=time.time())
    dump(ROOT/'protocol.json',p)
    dump(ROOT/'prompt-manifest.json',p['samples'])
    return p


def verify_preserved():
    saved=json.loads((ROOT/'preserved-parent.json').read_text())
    for name,digest in saved['hashes'].items():
        path=REPO/'prophetkv_with_expansion_results.txt' if name=='../parent-root-report.txt' else PARENT/name
        if sha(path)!=digest:raise ValueError('Preserved parent artifact changed: '+name)
    return len(saved['hashes'])


def verify_extension(source, hashes):
    p=json.loads((ROOT/'protocol.json').read_text())
    if sha(ROOT/'preserved-parent.json')!=p['preserved_parent_sha256']:
        raise ValueError('Parent preservation manifest changed')
    verify_parent_certificate()
    verify_preserved()
    if hashes(ROOT/'source')!=p['sources'] or hashes(REPO/'benchmarks/prophetkv_with_expansion_ratios')!=p['sources']:
        raise ValueError('Frozen extension sources changed')
    recovery=ROOT/'warmup-recovery'
    amendment=json.loads((recovery/'amendment.json').read_text())
    if sha(ROOT/'protocol.json')!=amendment['protocol_sha256']:
        raise ValueError('Recovery protocol changed')
    if hashes(source)!=amendment['sources'] or hashes(recovery/'source')!=amendment['sources']:
        raise ValueError('Recovery sources changed')
    if sha(recovery/'preserved-records.json')!=amendment['preserved_records_sha256']:
        raise ValueError('Recovery preservation manifest changed')
    preserved=json.loads((recovery/'preserved-records.json').read_text())
    for name,digest in preserved['hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Pre-recovery record changed: '+name)
    if hashes(ROOT/'private_ucm/ucm')!=p['private_ucm_hashes']:
        raise ValueError('Frozen private runtime changed')
    if sha(PARENT/'protocol.json')!=p['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    if sha(PARENT/'input-manifest.json')!=p['input_manifest_sha256']:
        raise ValueError('Frozen prompt manifest changed')
    for path,digest in p['runtime_file_hashes'].items():
        if sha(path)!=digest:raise ValueError('Patched runtime changed')
    for meta in p['samples']:
        if sha(meta['input_path'])!=meta['input_sha256']:raise ValueError('Frozen input changed')
    if (len(p['samples'])!=200 or p['measured_requests']!=600 or p['combined_requests']!=1000
            or p['cases']!=list(CASES) or p['case_parameters']!=PARAMETERS):
        raise ValueError('Extension scope or parameters changed')
    expected={str(gpu):list(CASES[(gpu-1)%3:]+CASES[:(gpu-1)%3]) for gpu in GPU_UUIDS}
    if p['phases']!=expected:raise ValueError('Method scheduling changed')
    return p
