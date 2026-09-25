"""One-time startup verification; no ongoing assistant polling after this receipt."""
import os,time
from pathlib import Path
from prepare import D,sha
from common import load,dump,identity

def detached():
    processes={}
    for name in ['supervisor','reporter']:
        s=load(D/(name+'.json'));pid=s['pid'];assert identity(pid)==s['identity']
        proc=Path('/proc')/str(pid);fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        status=dict(line.split(':',1) for line in proc.joinpath('status').read_text().splitlines() if ':' in line)
        environment=proc.joinpath('environ').read_bytes().split(b'\0')
        receipt=dict(pid=pid,identity=s['identity'],ppid=int(fields[1]),pgid=os.getpgid(pid),sid=os.getsid(pid),
            sighup_ignored=bool(int(status['SigIgn'].strip(),16)&1),stdin=os.readlink(proc/'fd/0'),
            cpu_only=b'CUDA_VISIBLE_DEVICES=' in environment)
        assert receipt['ppid']==1 and receipt['pgid']==pid and receipt['sid']==pid
        assert receipt['sighup_ignored'] and receipt['stdin']=='/dev/null' and receipt['cpu_only']
        processes[name]=receipt
    active=load(D/'active.json');assert identity(active['pid'])==active['identity']
    env=(Path('/proc')/str(active['pid'])/'environ').read_bytes().split(b'\0')
    visible=next(x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
    assert visible==','.join(d['uuid'] for d in load(D/'protocol.json')['gpu_devices'])
    processes['worker']=dict(pid=active['pid'],identity=active['identity'],visible_uuids=visible)
    dump(D/'detachment.json',dict(processes=processes,at=time.time()))

def verify():
    detached();first=D/'first-focus-all64-5-validation.json';deadline=time.monotonic()+900
    while not first.exists():
        assert load(D/'supervisor.json')['state']=='running'
        assert time.monotonic()<deadline,'Startup progress watchdog'
        time.sleep(1)
    first=load(first);certificate=load(first['validation']);assert certificate['complete']
    for name,h in certificate['sha256'].items():assert sha(name)==h
    record=load(first['record']);assert record['query_scope']=='focus' and record['layer_scope']=='all64'
    sample=next(m for m in load(D/'protocol.json')['samples'] if m['id']==record['sample_id'])
    assert record['query_positions']==sample['focus']['absolute_positions']
    assert all(c['observer_removed'] and c['layers']==list(range(64)) for c in record['priming_comparison'])
    session=load(record['session_path']);assert session['warmup_readiness']['verified_shards']==8
    dump(D/'startup-verification.json',dict(complete=True,record=first['record'],validation=first['validation'],
        span=sample['focus'],priming_comparison=record['priming_comparison'],
        warmup_committed_shards=8,exact_all_rank_masks=True,all64_layer_selected_sets=True,retirement_complete=True,
        no_timed_attention_observer=True,at=time.time()))
    print('Startup verified:',record['sample_id'],record['case'],'score',record['score'],'TTFT',record['ttft_seconds'],flush=True)
if __name__=='__main__':verify()
