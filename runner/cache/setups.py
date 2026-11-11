"""Metadata, engine startup and KV retirement helpers used by the runner."""
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from runner.inputs.config import VERSIONS
from runner.reporting.aggregate import EVALUATION_FIELDS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def setup_lock(root, build=False):
    root = Path(root)
    if build:
        root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if build else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Setup is in use; retry after its builder/readers exit') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def manifest_entries(path):
    """Resolve paths once; IDs need not be valid filesystem names."""
    entries, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = row.get('id')
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in seen:
            raise ValueError('Manifest requires unique, nonempty string IDs')
        seen.add(prompt_id)
        input_fields = set(row) - EVALUATION_FIELDS
        if input_fields == {'id', 'prepared'}:
            fields = ('prepared',)
        elif input_fields == {'id', 'context', 'question'}:
            fields = ('context', 'question')
        else:
            raise ValueError('Each row needs id and either prepared or context/question paths')
        entries.append(dict(id=prompt_id, **{key: (path.parent / row[key]).resolve() for key in fields},
                            **{key: row[key] for key in EVALUATION_FIELDS if key in row}))
    if not entries:
        raise ValueError('Manifest is empty')
    return entries


def bytes_per_shard(model, tp):
    cfg = json.loads((model / 'config.json').read_text())
    heads = cfg.get('num_key_value_heads', cfg['num_attention_heads'])
    dim = cfg.get('head_dim', cfg['hidden_size'] // cfg['num_attention_heads'])
    return 64 * max(1, heads // tp) * dim * 2 * 2 * cfg['num_hidden_layers']


def check_environment(tp):
    from importlib.metadata import version
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != tp or len(set(devices)) != tp or any(not d.startswith('GPU-') for d in devices):
        raise ValueError('Set CUDA_VISIBLE_DEVICES to exactly --tp distinct GPU UUIDs')
    for package, expected in VERSIONS.items():
        if version(package).split('+')[0] != expected:
            raise RuntimeError(f'{package} must be {expected}')
    return devices


def start_engine(cfg):
    from ucm.integration.vllm.patch.apply_patch import apply_all_patches
    apply_all_patches()
    from vllm import LLM
    from vllm.config import KVTransferConfig
    options = copy.deepcopy(cfg)
    if 'kv_transfer_config' in options:
        options['kv_transfer_config'] = KVTransferConfig(**options['kv_transfer_config'])
    return LLM(**options)


def retirement(llm, tp):
    from runner.runtime.worker import retire
    receipts = llm.collective_rpc(retire)
    if sorted(r['rank'] for r in receipts) != list(range(tp)) or not all(
            r['quiescent'] and r['transfers']['pending'] == 0 and r['request_bookkeeping'] == 0 for r in receipts):
        raise RuntimeError('Request retirement failed')
    return receipts


def verify_retired(llm, tp, rid, scheduler_path):
    receipts = retirement(llm,tp)
    if scheduler_path is not None:
        state = json.loads(Path(scheduler_path).read_text())
        if state != dict(request_id=rid,requests_blend_meta=0,requests_meta=0):
            raise RuntimeError('Router scheduler request did not retire')
    return receipts
