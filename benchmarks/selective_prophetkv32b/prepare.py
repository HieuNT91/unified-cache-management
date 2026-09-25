"""Freeze the exact completed cohort and prior artifact hashes; CPU only."""
import hashlib,json,shutil,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
R=H.parents[1]
D=R/'.results/selective-prophetkv32b-40samples-output256-20260924'
OLD=R/'.analysis/qwen3-32b-layer-search-40samples-20260924'
sys.path.insert(0,str(H/'runtime'))
from common import load,dump,CASES,identity,group_alive

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def prepare():
    assert not (D/'protocol.json').exists()
    assert load(OLD/'final-validation.json')['complete']
    for name in ['supervisor','reporter','active']:
        s=load(OLD/(name+'.json'))
        assert identity(s['pid'])!=s['identity'] and not group_alive(s['pid'])
    old=load(OLD/'protocol.json'); samples=[]
    tasks=load(OLD/'manifest.json')['tasks']
    for row in range(5):
        for task in tasks:
            m=next(m for m in old['samples'] if m['label']==task and m['source_row']==row)
            target=D/'inputs'/Path(m['input_path']).name;target.parent.mkdir(exist_ok=True)
            shutil.copy2(m['input_path'],target);assert sha(target)==m['input_sha256']
            prior=load(OLD/'samples'/m['id']/'record.json')
            samples.append(m|dict(input_path=str(target),prior_capture_stem=str(OLD/'captures'/prior['request_id'].replace(':','_'))))
    old.update(study='Selective ProphetKV Qwen3-32B on selection cohort',samples=samples,cases=list(CASES),
        measured_requests=360,max_new_tokens=256,max_model_len=65920,num_gpu_blocks=1031,
        selective_layers=[45,48,50,56,58],fusion='ascending layer sum FP32, TP all-reduce then divide by four',
        qualification='Disabled; ordinary initialization and priming only',same_selection_cohort=True,
        created_at=time.time())
    old['settings'].update(root=str(D),cache='/tmp/ucm-selective-prophetkv32b-20260924',samples=5,
                           scope=[dict(length=65536,task=t) for t in tasks])
    old.update(selection='question-only context-softmax; selected layers ascending FP32 sum; TP head mean; stable global top-k',
        timing='submission to first token; includes probe/transfer/selection/fusion; excludes load/build/readiness/prime/export/retirement',
        preserved_engine_window=65920,position_table_extension='Append 384 positions with unchanged factor-2 frequencies and magnitude; original 65536 entries bitwise preserved')
    old['memory_adaptation'].update(gpu_cache_blocks=1031,
        applies_to='all nine configurations',
        numerical_gate='retained per-layer memory-adaptation checks during normal model initialization; no separate model qualification',
        block_policy='ceil((max prompt 65664 + output256) / 64) plus one null block')
    old.update(max_output_tokens=256, samples_per_task_length=5, scope=old['settings']['scope'])
    old['datasets']=sorted({m['dataset_path'] for m in samples if 'dataset_path' in m})
    old['dataset_output_reserve_note']='128 is the historical dataset construction reserve; exact prompts are retained and actual generation cap is 256'
    dump(D/'protocol.json',old)
    pins={str(f):sha(f) for f in OLD.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.lock','.pyc')}
    dump(D/'preserved-prior.json',pins)
    dump(D/'prior-exit.json',dict(complete=True,prior_root=str(OLD),processes_exited=True,at=time.time()))
    print('Frozen 40 prompts and',len(pins),'prior artifacts')

def pin():
    files=[f for f in H.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']
    files += [D/'protocol.json',D/'preserved-prior.json'] + list((D/'inputs').glob('*'))
    dump(D/'implementation.json',{str(f):sha(f) for f in files})
if __name__=='__main__': {'prepare':prepare,'pin':pin}[sys.argv[1]]()
