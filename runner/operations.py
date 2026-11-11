"""Process ownership, environment and hardware checks for AdaTree inference."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from runner.cache.setups import atomic_json, file_hash, fingerprint
from runner.router.process import identity, alive, group_alive
ROOT = Path(__file__).resolve().parents[1]

def code_hashes():
    """Source provenance only; these hashes do not gate launch or resume."""
    paths=[ROOT/p for p in ('run.sh', 'status.py', 'requirements.txt') if (ROOT/p).is_file()]
    for base in ('runner','ucm','scripts'):
        paths.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json')
                     and '__pycache__' not in p.parts and 'vendor' not in p.parts)
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)}

def legacy_idle(root):
    for path in [root/'supervisor.json',*(root/'processes').glob('*.json')]:
        if not path.exists():continue
        state=json.loads(path.read_text())
        if alive(state) or (path.parent.name=='processes' and group_alive(state['pid'])):
            raise ValueError('Owned supervisor/engine is still alive; refusing a duplicate launch')

def environment(devices=''):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=devices,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',
        ENABLE_SPARSE='TRUE',VLLM_USE_V1='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
        VLLM_ALLOW_INSECURE_SERIALIZATION='1',VLLM_ATTENTION_BACKEND='FLASH_ATTN',PLATFORM='cuda',
        VLLM_USE_REROPE='0',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def terminate_owned(state):
    # Share the stop command's identity guard: a live leader must match both
    # start time and argv; orphan ranks must retain the original session and
    # cannot predate the recorded leader. Zombie leaders are absent from the
    # process snapshot and therefore do not hide valid surviving owned ranks.
    from scripts.launcher.launcher_stop import process_table, members
    pid=state['pid']
    def owned():
        return members(state, process_table())
    if owned():
        os.killpg(pid,signal.SIGTERM)
        end=time.monotonic()+30
        while group_alive(pid) and time.monotonic()<end:time.sleep(.2)
        if owned():os.killpg(pid,signal.SIGKILL)
        end=time.monotonic()+10
        while group_alive(pid) and time.monotonic()<end:time.sleep(.1)
    if group_alive(pid):
        raise RuntimeError('Owned process group remains; automatic retry is blocked')

def cleanup_caches(root, protocol):
    """Remove only dead attempts' explicitly owned cache namespaces, never results."""
    import re
    import shutil
    anchor=(Path(protocol['cache_root'])/fingerprint(str(root))[:16]).resolve()
    removed=[]
    for path in (root/'sessions').glob('*/ownership.json'):
        owner=json.loads(path.read_text())
        if alive(owner) or group_alive(owner['pid']):
            continue
        cache=Path(owner['cache'])
        if (cache.is_symlink() or cache.parent.resolve()!=anchor/f"group{owner['group']}"
                or not re.fullmatch('[a-f0-9]{32}',cache.name)):
            raise ValueError('Cache ownership/path mismatch; no cleanup performed')
        if cache.exists():
            shutil.rmtree(cache)
            removed.append(str(cache))
            atomic_json(path.parent/'cache-cleanup.json',dict(owned_engines_exited=True,cache=str(cache),at=time.time()))
    return removed

def idle(root):
    root=Path(root);legacy_idle(root)
    for path in (root/'sessions').glob('*/ownership.json'):
        state=json.loads(path.read_text())
        if alive(state) or group_alive(state['pid']):raise ValueError('Owned worker/engine remains alive')
    for path in [root/'run.lock',*root.glob('group*.lock')]:
        if not path.exists():continue
        with path.open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ValueError('Owned supervisor/worker lock is held') from error

def model_artifact(root,name):
    # User-selected local model snapshots legitimately symlink into HF blobs.
    # Corpus artifacts still use the stricter resolved-path containment check.
    name=Path(name)
    if name.is_absolute() or '..' in name.parts or not name.parts:
        raise ValueError('Expected a relative model artifact path')
    return Path(root)/name

def check_hardware(settings):
    from runner.router.profiles import allocation
    from runner.runtime.tensor_parallel import protocol_tp
    tp=protocol_tp(settings)
    memory=settings.get('gpu_memory_utilization',.95)
    if not 0 < memory < 1:
        raise ValueError('GPU memory utilization must be between 0 and 1')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    inventory={r[0].strip():dict(name=r[1].strip(),total_mib=float(r[2]),free_mib=float(r[3])) for r in csv.reader(raw.splitlines())}
    requested=sum(settings['groups'],[])
    index=json.loads((Path(settings['model'])/'model.safetensors.index.json').read_text())
    weights=index['metadata']['total_size']/tp
    p=allocation(settings.get('execution_profile',settings['dataset']))
    kv=p['kv_blocks']*64*262144/tp
    # Only reject an impossible weights+KV lower bound here. vLLM measures
    # activation/workspace; validate_initialization checks its actual KV capacity.
    required=(weights+kv)/memory/2**20
    for gpu in requested:
        if gpu not in inventory:raise ValueError('Configured UUID is unavailable')
        if inventory[gpu]['total_mib']<required or inventory[gpu]['free_mib']<required:
            raise ValueError(f'Cannot fit GPU {gpu}: total={inventory[gpu]["total_mib"]:.0f} MiB, '
                f'free={inventory[gpu]["free_mib"]:.0f} MiB, requires at least {required:.0f} MiB '
                'before runtime workspace; no CPU offload or input shortening is applied')
        if inventory[gpu]['free_mib']<inventory[gpu]['total_mib']*memory:
            raise ValueError(f'GPU {gpu} has insufficient free memory for the {memory:.0%} automatic KV budget')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if set(requested)&{r.strip() for r in busy}:raise ValueError('Configured GPUs are occupied; existing processes left untouched')
    return dict(groups=[[dict(uuid=g,**inventory[g]) for g in group] for group in settings['groups']],
                required_mib_per_rank=required,tp=tp,weights_bytes_per_rank=weights,kv_bytes_per_rank=kv,
                gpu_memory_utilization=memory,kv_cache_mode='auto',kv_bytes_are_minimum=True,
                workspace_validation='runtime-kv-capacity-audit')
