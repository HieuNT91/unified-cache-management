"""Freeze the completed selective experiment and copy its exact ordered inputs."""
import hashlib,json,shutil,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
D=R/'.results/vanilla-prophetkv32b-40samples-output256-20260924'
OLD=R/'.results/selective-prophetkv32b-40samples-output256-20260924'
sys.path.insert(0,str(H/'runtime'))
from common import load,dump,CASES,identity,group_alive

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def prepare():
    assert not D.exists()
    certificate=load(OLD/'final-validation.json');assert certificate['complete'] and certificate['validated']==360
    for name,h in certificate['artifact_sha256'].items():assert sha(name)==h
    for name in ['supervisor','reporter','active']:
        s=load(OLD/(name+'.json'));assert identity(s['pid'])!=s['identity'] and not group_alive(s['pid'])
    D.mkdir();(D/'private_ucm').symlink_to(H/'private_ucm',target_is_directory=True)
    p=load(OLD/'protocol.json');samples=[]
    for m in p['samples']:
        target=D/'inputs'/Path(m['input_path']).name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(m['input_path'],target);assert sha(target)==m['input_sha256']
        samples.append(m|dict(input_path=str(target)))
    p.pop('selective_layers',None)
    p.update(study='Vanilla all-layer mean ProphetKV matched to selective 40-prompt comparison',samples=samples,cases=list(CASES),
        scoring_layers=list(range(64)),fusion='ascending native FP32 sum divided by64; TP all-reduce and divide by4',
        selection='question/head mean normalized over all cached keys, equal mean of all64 layers, stable global top-k',
        comparison_root=str(OLD),baseline_policy='40 fresh baseline measurements; prior baseline preserved for cross-run comparison',
        created_at=time.time())
    p['settings'].update(root=str(D),cache='/tmp/ucm-vanilla-prophetkv32b-20260924')
    dump(D/'protocol.json',p)
    # Includes prior source manifests and the historical all-layer capture hashes.
    pins=load(OLD/'preserved-prior.json')
    for root in [OLD,R/'benchmarks/selective_prophetkv32b']:
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.lock','.pyc'):pins[str(f)]=sha(f)
    dump(D/'preserved-prior.json',pins)
    dump(D/'prior-exit.json',dict(complete=True,prior_root=str(OLD),validated=360,processes_exited=True,at=time.time()))
    print('Frozen 40 exact inputs and',len(pins),'prior artifacts')

def pin():
    files=[f for f in H.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']
    files += [D/'protocol.json',D/'preserved-prior.json']+list((D/'inputs').glob('*'))
    dump(D/'implementation.json',{str(f):sha(f) for f in files})
if __name__=='__main__':{'prepare':prepare,'pin':pin}[sys.argv[1]]()
