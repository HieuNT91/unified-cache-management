"""CPU-only configuration, provenance, and remote GPU identity checks."""
import csv
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LENGTHS = (65536,)
RULER_TASKS = ('niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
               'niah_multivalue', 'niah_multiquery', 'vt', 'cwe', 'fwe', 'qa_1', 'qa_2')
TASKS = (*RULER_TASKS, 'longbench_v2_non_thinking')
JOBS = {str(i): tuple((65536, task) for task in TASKS) for i in range(2)}
CASES = ('baseline', *(f'expansion-{n}' for n in (10,20,30,40,50)))
PARAMETERS = {f'expansion-{n}': dict(total_ratio=n/100, anchor_ratio=a,
    max_gap=2, score_exponent=.5, window_scale=8., window_exponent=.5,
    min_window=8, max_window=64) for n,a in ((10,.075),(20,.15),(30,.225),(40,.30),(50,.375))}
CONTROLS = ()
TIMING = 'engine_step_first_token_monotonic'
ENGINE_POLICY = 'persistent-per-method-yarn2-65792'


def engine_group(length):
    return 'yarn2-65792'


def model_limit(protocol, sample):
    return 65792


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def settings():
    job = os.environ.get('EXPANSION_JOB', '0')
    if job not in JOBS:
        raise ValueError('EXPANSION_JOB must be 0 (GPUs 4,5) or 1 (GPUs 6,7)')
    parent = Path(os.environ['RESULT_ROOT']).expanduser().resolve()
    cache = Path(os.environ['CACHE_ROOT']).expanduser().resolve()
    memory = float(os.environ.get('GPU_MEMORY_UTILIZATION', '.90'))
    if not 0 < memory < 1:
        raise ValueError('Invalid GPU memory fraction')
    return dict(model=str(Path(os.environ['MODEL_PATH']).expanduser().resolve()),
        root=str(parent / ('job-'+job)), cache=str(cache / ('job-'+job)),
        ruler=str(Path(os.environ['RULER_ROOT']).expanduser().resolve()),
        longbench=str(Path(os.environ['LONGBENCH_DATA']).expanduser().resolve()),
        samples=100, chunk=4096, indices=([4,5] if job=='0' else [6,7]), memory=memory,
        job_id=job, input_limit=65536, max_model_len=65792, thinking_enabled=False, max_output_tokens=256,
        scope=[dict(length=65536,task=t) for t in TASKS])


def scope_for_job(job):
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
    if indices not in ([4,5], [6,7]) or any('A800' not in x['name'] or x['memory_mib'] < 75000 for x in chosen):
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
    return dict(rope_type='yarn', factor=2.0, original_max_position_embeddings=32768)
