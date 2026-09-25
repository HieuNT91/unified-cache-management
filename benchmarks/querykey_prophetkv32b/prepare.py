"""Prepare immutable inputs before cancellation; freeze control inventory afterward."""
import hashlib,shutil,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
D=R/'.results/querykey-prophetkv32b-mk2-mk3-cwe-5samples-output256-20260924'
VANILLA=R/'.results/vanilla-prophetkv32b-40samples-output256-20260924'
SELECTIVE=R/'.results/selective-prophetkv32b-40samples-output256-20260924'
STUDY=R/'.analysis/qwen3-32b-layer-search-40samples-20260924'
sys.path.insert(0,str(H/'runtime'))
from common import load,dump
from query_config import CASES,TASKS,schedule,configuration

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def prepare():
    from transformers import AutoTokenizer
    from spans import map_span
    assert not D.exists();D.mkdir();(D/'private_ucm').symlink_to(H/'private_ucm',target_is_directory=True)
    p=load(VANILLA/'protocol.json');tok=AutoTokenizer.from_pretrained(p['model'],local_files_only=True)
    samples=[];sources={}
    for m in p['samples']:
        if m['label'] not in TASKS:continue
        target=D/'inputs'/Path(m['input_path']).name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(m['input_path'],target);assert sha(target)==m['input_sha256']
        if m['source_path'] not in sources:sources[m['source_path']]=[__import__('json').loads(x) for x in Path(m['source_path']).read_text().splitlines()]
        focus=map_span(tok,m,load(target),sources[m['source_path']][m['source_row']])
        sidecar=D/'spans'/(m['id']+'.json');dump(sidecar,focus)
        evidence=D/'evidence'/(m['id']+'.json');evidence.parent.mkdir(exist_ok=True)
        shutil.copy2(STUDY/'samples'/m['id']/'evidence.json',evidence)
        samples.append(m|dict(input_path=str(target),focus=focus,focus_manifest=str(sidecar),evidence_path=str(evidence)))
    assert len(samples)==15 and [m['label'] for m in samples]==list(TASKS)*5
    for k in ['scoring_layers','comparison_root']:p.pop(k,None)
    p.update(study='Query-key attention ablation: MK2/MK3/CWE',samples=samples,cases=list(CASES),
        scope=[dict(length=65536,task=t) for t in TASKS],measured_requests=420,combined_with_baseline=435,
        focused_fresh_measurements=210,missing_controls='frozen after transition',
        fusion='all64: ascending FP32 sum /64; selected5: ascending FP32 sum; unchanged TP sum /4',
        selection='Only query rows vary; mean query/head softmax over all cached context keys; stable global top-k',
        layer_scopes={'all64':list(range(64)),'selected5':[45,48,50,56,58]},query_scopes=['focus','full_question'],
        baseline_policy='Retain latest vanilla baseline for the same 15 prompts',created_at=time.time())
    p['settings'].update(root=str(D),cache='/tmp/ucm-querykey-prophetkv32b-20260924',scope=p['scope'],samples=5)
    p['memory_adaptation']['applies_to']='all configurations'
    dump(D/'protocol.json',p)
    print('Prepared 15 byte-identical inputs:',[(m['label'],m['source_row'],len(m['focus']['token_ids'])) for m in samples])

def retained_inventory(p):
    import suite
    result=[]
    for m in p['samples']:
        for case in ('baseline',*CASES):
            if case!='baseline' and configuration(case)['query_scope']=='focus':continue
            cfg=None if case=='baseline' else configuration(case)
            root=SELECTIVE if cfg and cfg['layer_scope']=='selected5' else VANILLA
            oldcase='baseline' if cfg is None else cfg['method']+'-'+str(cfg['budget'])
            path=root/'records'/m['id']/(oldcase+'.json');receipt=path.with_suffix('.validated.json')
            if not receipt.exists():
                assert case!='baseline','Latest baseline incomplete';continue
            cert=load(receipt);assert cert['complete'] and cert['sha256']
            for name,h in cert['sha256'].items():assert sha(name)==h,name
            oldp=load(root/'protocol.json');oldm=next(x for x in oldp['samples'] if x['id']==m['id'])
            assert oldm['input_sha256']==m['input_sha256']==sha(oldm['input_path'])
            record=suite.validate(root,oldp,oldm,oldcase,path)
            assert record['max_model_len']==65920 and record['max_output_tokens']==256
            result.append(dict(sample_id=m['id'],case=case,path=str(path),source_case=oldcase,
                source_root=str(root),receipt=str(receipt),receipt_sha256=sha(receipt),sha256=cert['sha256']))
    return result

def freeze():
    from common import identity,group_alive
    assert load(D/'transition/cancellation.json')['complete']
    for root in [VANILLA,SELECTIVE]:
        for n in ['supervisor','reporter','active']:
            s=load(root/(n+'.json'));assert identity(s['pid'])!=s['identity'] and not group_alive(s['pid'])
    p=load(D/'protocol.json');retained=retained_inventory(p);fresh=schedule(p['samples'],retained)
    assert len(fresh)+len(retained)==435 and sum(x['case'].startswith('focus-') for x in fresh)==210
    dump(D/'control-inventory.json',dict(retained=retained,fresh=fresh,target_new=len(fresh),combined=435,frozen_at=time.time()))
    pins={}
    for root in [VANILLA,SELECTIVE,R/'benchmarks/vanilla_prophetkv32b',R/'benchmarks/selective_prophetkv32b']:
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.lock','.pyc'):pins[str(f)]=sha(f)
    for name,h in load(VANILLA/'preserved-prior.json').items():
        assert sha(name)==h,name
        pins[name]=h
    dump(D/'preserved-prior.json',pins)
    print('Frozen:',len(retained),'retained;',len(fresh),'new including',len(fresh)-210,'missing controls')

def pin():
    files=[f for f in H.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']
    files += [D/'protocol.json',D/'preserved-prior.json',D/'control-inventory.json',D/'cpu-validation.json',D/'cpu-report-validation.json',D/'cpu-control-reuse-validation.json',D/'transition/cancellation.json']
    files += [f for name in ['inputs','spans','evidence'] for f in (D/name).glob('*')]
    dump(D/'implementation.json',{str(f):sha(f) for f in files})
if __name__=='__main__':{'prepare':prepare,'freeze':freeze,'pin':pin}[sys.argv[1]]()
