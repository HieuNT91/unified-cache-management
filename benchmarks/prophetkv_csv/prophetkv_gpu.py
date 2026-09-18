"""Owned, UUID-checked idle GPU placeholders for the ProphetKV experiment."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(os.environ.get('PROPHETKV_REPO', Path(__file__).resolve().parents[2]))
ROOT = REPO / '.results/prophetkv-csv-backfill-20260918'
PYTHON = REPO / '.envs/cacheblend/bin/python'
GPU_UUIDS = {3:'GPU-f7a26c8e-4455-7a74-3e75-f02c21d7c9e5',
             4:'GPU-d52293e0-0963-52cc-f251-0827ef68b02f'}

def check(gpu):
    if gpu not in GPU_UUIDS:
        raise ValueError('Only physical GPUs 3 and 4 are authorized')
    listing = subprocess.check_output(['nvidia-smi','-L'], text=True)
    expected = GPU_UUIDS[gpu]
    if not any(s.startswith(f'GPU {gpu}:') and expected in s for s in listing.splitlines()):
        raise RuntimeError('Physical device identity changed')
    return listing

def processes(gpu):
    rows = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'],text=True)
    return [int(s.split(',')[1]) for s in rows.splitlines() if s.split(',')[0].strip()==GPU_UUIDS[gpu]]

def identity(pid):
    try:
        p=Path(f'/proc/{pid}')
        return {'start':p.joinpath('stat').read_text().split()[21],
                'cmd':p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except (FileNotFoundError,ProcessLookupError):
        return None

def release(gpu):
    path=ROOT/f'placeholder-{gpu}.json'
    if not path.exists():return
    saved=json.loads(path.read_text())
    if identity(saved['pid'])==saved['identity']:
        os.kill(saved['pid'],signal.SIGTERM)
        for _ in range(100):
            if saved['pid'] not in processes(gpu):break
            time.sleep(.1)
        else:raise RuntimeError('Owned placeholder did not release GPU')
    path.unlink()

def reserve(gpu):
    check(gpu)
    path=ROOT/f'placeholder-{gpu}.json'
    if path.exists():
        s=json.loads(path.read_text())
        if identity(s['pid'])==s['identity']:return
    if processes(gpu):return  # never displace another user
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(CUDA_VISIBLE_DEVICES=GPU_UUIDS[gpu],OMP_NUM_THREADS='1')
    with (ROOT/f'placeholder-{gpu}.log').open('a') as log:
        p=subprocess.Popen(['nohup',str(PYTHON),str(Path(__file__).resolve()),'hold',str(gpu)],
            cwd='/tmp',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(.2)
    path.write_text(json.dumps({'pid':p.pid,'gpu':gpu,'uuid':GPU_UUIDS[gpu],
        'identity':identity(p.pid),'started':time.time(),'purpose':'idle placeholder; release before benchmark'}))

def hold(gpu):
    check(gpu)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUIDS[gpu]:raise RuntimeError('Unsafe visibility')
    import torch
    if torch.cuda.device_count()!=1:raise RuntimeError('Unsafe visibility')
    stop=False
    def finish(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,finish)
    # Keep a CUDA allocation and a small periodic operation; no competing work.
    buffer=torch.zeros(256*1024*1024,device='cuda',dtype=torch.uint8)
    print(f'placeholder ready GPU {gpu} UUID {GPU_UUIDS[gpu]} pid {os.getpid()}',flush=True)
    while not stop:
        buffer[:1024].add_(1)
        torch.cuda.synchronize()
        time.sleep(1)
    print('placeholder released',flush=True)

if __name__=='__main__':
    command=sys.argv[1]
    for gpu in ([int(sys.argv[2])] if len(sys.argv)>2 else GPU_UUIDS):
        {'hold':hold,'reserve':reserve,'release':release}[command](gpu)
