"""CPU-only configuration, provenance, and remote GPU identity checks."""
import csv
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = Path('/home/thnguyen/spark/unified-cache-management')
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
CASES = ('baseline',) + tuple(f'selective_prophetkv-{r}' for r in (5,10,20,30,40,50,60,1))
CONTROLS = ('prophetkv-0', 'prophetkv-100')
TIMING = 'engine_step_first_token_monotonic'
ENGINE_POLICY = 'persistent-per-method-and-rope'


def engine_group(length):
    return 'yarn64k' if length == 65536 else 'native'


def model_limit(protocol, sample):
    """Use the same fixed allocation for every method and retry in a RoPE group."""
    group = engine_group(sample['context_target'])
    sizes = [m['tokens'] for m in protocol.get('samples', [sample])
             if engine_group(m['context_target']) == group]
    required = ((max(sizes) + 256 + 63) // 64) * 64
    if required > 65920:
        raise ValueError('New prompts exceed the preserved 65920-token allocation')
    return 65920


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def settings():
    return dict(model='/home/thnguyen/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137',
                root=str(REPO / '.results/qwen3-32b-prophetkv-niahqa8x10-yarn2-no-smoke-20260921'),
                cache=str(REPO / '.cache/qwen3-32b-prophetkv-yarn2-niahqa-no-smoke'),
                ruler='/home/thnguyen/unified-cache-management/.downloads/RULER',
                samples=10, chunk=4096, indices=[1,2,3,4], memory=.95,
                job_id='local-tp4', scope=[dict(length=65536,task=t) for t in ('niah_multikey_3','niah_multikey_2','niah_multikey_1','qa_1','qa_2','niah_single_1','niah_single_2','niah_single_3')])


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
    if indices != [1,2,3,4] or any('RTX 4500 Ada' not in d['name'] for d in chosen):
        raise RuntimeError('This restart is restricted to physical RTX 4500 Ada GPUs 1-4')
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
    # Requested factor 2; retain all prompt positions through a private table extension.
    return (dict(rope_type='yarn', factor=2.0, original_max_position_embeddings=32768)
            if length == 65536 else None)
