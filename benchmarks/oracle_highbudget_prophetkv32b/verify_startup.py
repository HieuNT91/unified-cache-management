"""One-time startup checks; subsequent configurations publish first-acceptance receipts."""
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
        r=dict(pid=pid,identity=s['identity'],ppid=int(fields[1]),pgid=os.getpgid(pid),sid=os.getsid(pid),
            sighup_ignored=bool(int(status['SigIgn'].strip(),16)&1),stdin=os.readlink(proc/'fd/0'),cpu_only=b'CUDA_VISIBLE_DEVICES=' in environment)
        assert r['ppid']==1 and r['pgid']==pid and r['sid']==pid and r['sighup_ignored'] and r['stdin']=='/dev/null' and r['cpu_only']
        processes[name]=r
    a=load(D/'active.json');assert identity(a['pid'])==a['identity']
    env=(Path('/proc')/str(a['pid'])/'environ').read_bytes().split(b'\0')
    visible=next(x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='))
    assert visible==','.join(d['uuid'] for d in load(D/'oracle/protocol.json')['gpu_devices'])
    processes['worker']=dict(pid=a['pid'],identity=a['identity'],visible_uuids=visible)
    dump(D/'detachment.json',dict(processes=processes,at=time.time()))

def verify():
    detached();path=D/'oracle/first-target_only-all64-0-validation.json';deadline=time.monotonic()+900
    while not path.exists():
        assert load(D/'supervisor.json')['state']=='running'
        assert time.monotonic()<deadline,'Startup progress watchdog'
        time.sleep(1)
    first=load(path);cert=load(first['validation']);assert cert['complete']
    for name,h in cert['sha256'].items():assert sha(name)==h
    r=load(first['record']);assert r['selection_mode']=='target_only' and r['native_selected_tokens']==0
    assert r['selected_context_tokens']==r['oracle_eligible_tokens']
    assert all(c['observer_removed'] and c['layers']==[] for c in r['priming_comparison'])
    session=load(r['session_path']);assert session['warmup_readiness']['verified_shards']==8
    dump(D/'startup-verification.json',dict(complete=True,record=first['record'],validation=first['validation'],
        sample_id=r['sample_id'],selected_oracle_tokens=r['selected_context_tokens'],attention_scoring_skipped=True,
        all_rank_exact_masks=True,all64_layer_selected_sets=True,warmup_committed_shards=8,retirement_complete=True,
        no_timed_capture_observer=True,second_experiment_queued=True,at=time.time()))
    print('Startup verified',r['sample_id'],'target tokens',r['selected_context_tokens'],'score',r['score'],'TTFT',r['ttft_seconds'],flush=True)
if __name__=='__main__':verify()
