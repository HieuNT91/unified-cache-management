"""Authorized handoff, after CPU validation; signal only verified owned processes."""
import os,signal,time,shutil
from pathlib import Path
from prepare import H,D,VANILLA,SELECTIVE,sha
from common import load,dump,identity,group_alive,selected_devices,busy_devices

def snapshot(root,target):
    target.mkdir(parents=True,exist_ok=True)
    for f in root.iterdir():
        if f.is_file() and f.suffix in ('.json','.log'):shutil.copy2(f,target/f.name)
    active=load(root/'active.json');log=Path(active['log'])
    shutil.copy2(log,target/'active-session.log')
    return active

def transition():
    assert load(D/'cpu-validation.json')['complete'] and load(D/'cpu-report-validation.json')['complete']
    directory=D/'transition';assert not directory.exists();directory.mkdir()
    states={name:load(VANILLA/(name+'.json')) for name in ['progress','supervisor','reporter','active','detachment','supervisor-launch','reporter-launch']}
    print(Path(states['active']['log']).read_text(errors='replace')[-2000:],flush=True)
    # Completed historical jobs must stay exited; current identities override historical PIDs.
    for n in ['supervisor','reporter','active']:
        s=load(SELECTIVE/(n+'.json'));assert identity(s['pid'])!=s['identity'] and not group_alive(s['pid'])
    _,devices=selected_devices([1,2,3,4]);assert devices==load(D/'protocol.json')['gpu_devices']
    apps=busy_devices(devices);owned=[]
    active=states['active']
    for uuid,pid,*_ in apps:
        pid=int(pid);assert os.getpgid(pid)==active['pid'],'Unrecognized GPU workload; no cancellation performed'
        owned.append(dict(uuid=uuid,pid=pid,identity=identity(pid),pgid=os.getpgid(pid)))
    for n in ['supervisor','reporter','active']:
        s=states[n];assert identity(s['pid'])==s['identity'],n
        assert str(VANILLA).encode() in bytes.fromhex(s['identity']['cmd']) or b'benchmarks/vanilla_prophetkv32b/run.py' in bytes.fromhex(s['identity']['cmd'])
    snapshot(VANILLA,directory/'before')
    pins={}
    for receipt in (VANILLA/'records').rglob('*.validated.json'):
        cert=load(receipt);assert cert['complete']
        for name,h in cert['sha256'].items():assert sha(name)==h; pins[name]=h
        pins[str(receipt)]=sha(receipt)
    dump(directory/'accepted-before.json',pins)
    dump(directory/'workload-identities.json',dict(states=states,gpu_processes=owned,at=time.time()))
    signals=[]
    for n in ['reporter','supervisor']:
        s=states[n];assert identity(s['pid'])==s['identity'];os.kill(s['pid'],signal.SIGTERM)
        signals.append(dict(name=n,pid=s['pid'],identity=s['identity'],signal='SIGTERM',at=time.time()))
    dump(directory/'signals.json',signals)
    deadline=time.monotonic()+55
    while time.monotonic()<deadline:
        if all(identity(states[n]['pid'])!=states[n]['identity'] for n in ['supervisor','reporter']) and not group_alive(active['pid']):break
        time.sleep(.25)
    assert all(identity(states[n]['pid'])!=states[n]['identity'] for n in ['supervisor','reporter'])
    assert not group_alive(active['pid']) and not busy_devices(devices)
    cleanup=load(VANILLA/'cleanup.json');assert cleanup['complete'] and cleanup['engine_exited'] and cleanup['cache_removed']
    assert all(sha(name)==h for name,h in pins.items()),'Accepted artifact changed'
    snapshot(VANILLA,directory/'after')
    accepted={};incomplete={}
    for folder in (VANILLA/'records').iterdir():
        if not folder.is_dir():continue
        for f in folder.iterdir():
            if not f.is_file():continue
            case=f.name.split('.')[0]
            if (folder/(case+'.validated.json')).exists():accepted[str(f)]=sha(f)
            else:
                dest=directory/'incomplete'/folder.name/f.name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(f,dest);incomplete[str(f)]=dict(sha256=sha(f),preserved_copy=str(dest));assert sha(dest)==sha(f)
    dump(directory/'accepted-after.json',accepted);dump(directory/'incomplete.json',incomplete)
    dump(directory/'cancellation.json',dict(complete=True,signals=signals,owned_engines_exited=True,gpu_compute_empty=True,
        accepted_count=len(list((VANILLA/'records').rglob('*.validated.json'))),accepted_preserved=True,incomplete_preserved_separately=True,
        cleanup=cleanup,final_progress=load(VANILLA/'progress.json'),at=time.time()))
    print('Transition complete; accepted records and incomplete artifacts preserved',flush=True)
if __name__=='__main__':transition()
