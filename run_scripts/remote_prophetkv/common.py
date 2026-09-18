"""CPU-only configuration, provenance, and remote GPU identity checks."""
import csv
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LENGTHS = (8192, 16384, 32768, 65536)
TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3',
         'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
         'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2')
# Four disjoint task/length units per job, 2,800 requests each at n=100.
JOBS = {
    '0': ((8192, 'niah_multivalue'), *((65536, task) for task in TASKS[:3])),
    '1': ((16384, 'niah_multivalue'), *((65536, task) for task in TASKS[3:6])),
    '2': ((32768, 'niah_multivalue'), *((65536, task) for task in TASKS[6:9])),
    '3': tuple((65536, task) for task in TASKS[9:]),
}
CASES = ('baseline', 'prophetkv-5', 'prophetkv-10', 'prophetkv-20',
         'prophetkv-30', 'prophetkv-40', 'prophetkv-50')
CONTROLS = ('prophetkv-0', 'prophetkv-100')
TIMING = 'engine_step_first_token_monotonic'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def settings():
    indices = [int(x) for x in os.environ.get('GPU_INDICES', '0,1').split(',')]
    if len(set(indices)) != 2 or len(indices) != 2 or min(indices) < 0:
        raise ValueError('GPU_INDICES must contain exactly two distinct physical indices (TP=2)')
    job = os.environ.get('REMOTE_JOB_ID', 'all')
    scope = scope_for_job(job)
    if job != 'all' and indices != [2 * int(job), 2 * int(job) + 1]:
        raise ValueError('Numbered jobs must use their assigned disjoint GPU pair')
    n = int(os.environ.get('NUM_SAMPLES', '100'))
    chunk = int(os.environ.get('CHUNK_SIZE', '4096'))
    memory = float(os.environ.get('GPU_MEMORY_UTILIZATION', '.90'))
    if n < 1 or chunk < 64 or chunk % 64 or chunk > 4096 or not 0 < memory < 1:
        raise ValueError('Invalid sample count, chunk size (64..4096, multiple of 64), or memory fraction')
    return dict(model=str(Path(os.environ['MODEL_PATH']).expanduser().resolve()),
                root=str(Path(os.environ['RESULT_ROOT']).expanduser().resolve()),
                cache=str(Path(os.environ['CACHE_ROOT']).expanduser().resolve()),
                ruler=str(Path(os.environ['RULER_ROOT']).expanduser().resolve()),
                samples=n, chunk=chunk, indices=indices, memory=memory,
                job_id=job, scope=[dict(length=length, task=task) for length, task in scope])


def scope_for_job(job):
    if job == 'all':
        return tuple(unit for units in JOBS.values() for unit in units)
    if job not in JOBS:
        raise ValueError('REMOTE_JOB_ID must be all, 0, 1, 2, or 3')
    return JOBS[job]


def inventory():
    listing = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                   '--format=csv,noheader,nounits'], text=True)
    devices = {}
    for index, uuid, name, memory in csv.reader(raw.splitlines(), skipinitialspace=True):
        index = int(index)
        if not any(line.startswith(f'GPU {index}:') and uuid in line for line in listing.splitlines()):
            raise RuntimeError('nvidia-smi inventory disagrees with physical GPU listing')
        devices[index] = dict(index=index, uuid=uuid, name=name, memory_mib=int(memory))
    return listing, devices


def selected_devices(indices):
    listing, devices = inventory()
    chosen = [devices[index] for index in indices]
    if any('A800' not in x['name'] for x in chosen):
        raise RuntimeError('These launch scripts target the remote A800 server; selected device is not A800')
    return listing, chosen


def verify_gpu_visibility():
    expected = json.loads(os.environ['REMOTE_GPU_DEVICES'])
    _, actual = selected_devices([d['index'] for d in expected])
    if actual != expected or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(d['uuid'] for d in expected):
        raise RuntimeError('Remote physical GPU identity/UUID visibility changed')
    return expected


def busy_devices(devices):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    allowed = {d['uuid'] for d in devices}
    return [row for row in csv.reader(raw.splitlines(), skipinitialspace=True)
            if len(row) >= 2 and row[0] in allowed]


def identity(pid):
    try:
        base = Path('/proc') / str(pid)
        # The comm field may itself contain spaces/parentheses.
        fields = base.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(start=fields[19], cmd=base.joinpath('cmdline').read_bytes().hex())
    except (OSError, IndexError):
        return None


def group_alive(pgid):
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == pgid:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def rope_for_length(length):
    # Extra chat/marker/padding tokens can put a 64K-target prompt above 65536.
    # Use the model card's documented 4x configuration, with no checkpoint edits.
    return (dict(rope_type='yarn', factor=4.0, original_max_position_embeddings=32768)
            if length == 65536 else None)
