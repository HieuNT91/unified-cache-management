"""Physical GPU identity checks; no reservation workloads."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(os.environ.get('PROPHETKV_REPO', Path(__file__).resolve().parents[2]))
ROOT = REPO / '.results/prophetkv-with-expansion-ruler4x50-20260923'
PYTHON = Path('/home/thnguyen/unified-cache-management/.envs/cacheblend/bin/python')
GPU_UUIDS = {1:'GPU-02dade44-3157-21e2-549f-7aa76931eb02',2:'GPU-90021bb8-8e9c-1225-eddd-4c43ce187cc6',3:'GPU-f7a26c8e-4455-7a74-3e75-f02c21d7c9e5',
             4:'GPU-d52293e0-0963-52cc-f251-0827ef68b02f'}

def check(gpu):
    if gpu not in GPU_UUIDS:
        raise ValueError('Only physical GPUs 1–4 are authorized')
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
