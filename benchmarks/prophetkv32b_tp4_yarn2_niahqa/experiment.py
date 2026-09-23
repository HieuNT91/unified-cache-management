"""Freeze the independent eight-task YaRN 2x experiment."""
import hashlib
import os
from pathlib import Path
import shutil
import time

from common import HERE, REPO, CASES, dump, load, identity, rope_for_length, model_limit

ORIGINAL = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-20260921'
EXTENSION = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-extension-20260921'
PREVIOUS_SOURCE = REPO / 'benchmarks/prophetkv32b_tp4_yarn2'
PARENT = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-yarn2-20260921'
SMALL = Path('/home/thnguyen/unified-cache-management/.results/prophetkv-expanded5-ruler13x100-longbenchv2-20260918')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_hashes(hashes, base=Path('/')):
    for name, expected in hashes.items():
        if sha256(base / name) != expected:
            raise ValueError(f'Frozen artifact changed: {base / name}')


def verify_unchanged_inference():
    names = ['worker.py', 'reference.py', 'persistent_worker.py', 'cacheblend_ruler.py',
             'prophetkv_common.py', 'lifecycle.py', 'persistent_connector.py', 'memory_worker.py', 'rope_window.py']
    names += [str(p.relative_to(HERE)) for p in (HERE / 'method').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for name in names:
        if (HERE / name).read_bytes() != (PREVIOUS_SOURCE / name).read_bytes():
            raise ValueError(f'Inference implementation changed: {name}')
    return names


def verify_frozen(root):
    amendment = load(root / 'experiment.json')
    check_hashes(amendment['preserved_sha256'])
    check_hashes(load(root / 'provenance.json')['sha256'], root)
    for path in (root / 'source').rglob('*'):
        if path.is_file() and path.read_bytes() != (HERE / path.relative_to(root / 'source')).read_bytes():
            raise ValueError(f'Executed source differs from snapshot: {path}')
    p = load(root / 'protocol.json')
    if p['rope_scaling_64k'] != rope_for_length(65536) or p['rope_scaling_64k']['factor'] != 2:
        raise ValueError('Wrong RoPE factor')
    if os.environ.get('VLLM_ALLOW_LONG_MAX_MODEL_LEN') != '1':
        raise ValueError('Exact prompt preservation requires the documented length override')


def prepare_experiment(cfg, root):
    from suite import check_orphan, progress, verify, preflight, encode
    if (root / 'protocol.json').exists():
        verify_frozen(root)
        return verify(cfg, root)
    check_orphan(PARENT)
    for name in ('supervisor.json', 'reporter.json'):
        state = load(PARENT / name)
        if identity(state['pid']) == state['identity']:
            raise RuntimeError('Previous experiment still has a live process')
    if not load(PARENT / 'final/integrity-validation.json')['complete']:
        raise ValueError('Parent experiment is incomplete')
    runtime, _, _, devices, tokenizer = preflight(cfg)
    unchanged = verify_unchanged_inference()
    preserved = dict(load(PARENT / 'experiment.json')['preserved_sha256'])
    preserved.update({str(PARENT / n): h for n, h in load(PARENT / 'provenance.json')['sha256'].items()})
    for folder in ('records', 'final', 'yarn-comparison'):
        for path in (PARENT / folder).rglob('*'):
            if path.is_file():
                preserved[str(path)] = sha256(path)
    for name in ('qwen3_32b_yarn2_results.txt', 'qwen3_32b_yarn_comparison.txt'):
        preserved[str(REPO / name)] = sha256(REPO / name)
    check_hashes(preserved)
    previous = load(PARENT / 'protocol.json')
    for key in ('model', 'samples', 'chunk', 'indices', 'memory'):
        if cfg[key] != previous['settings'][key]:
            raise ValueError(f'Unexpected configuration change: {key}')
    small = load(SMALL / 'protocol.json')
    preserved[str(SMALL / 'protocol.json')] = sha256(SMALL / 'protocol.json')
    samples, datasets, source_hashes = [], [], {}
    for unit in cfg['scope']:
        task = unit['task']
        selected = [m for m in small['samples'] if m['label'] == task and m['source_row'] < cfg['samples']]
        selected.sort(key=lambda m: m['source_row'])
        if [m['source_row'] for m in selected] != list(range(cfg['samples'])):
            raise ValueError(f'Missing or duplicate source rows for {task}')
        sources = {m['source_path'] for m in selected}
        if len(sources) != 1:
            raise ValueError('Source rows span multiple datasets')
        source = Path(sources.pop())
        digest = sha256(source)
        if any(m['source_metadata']['source_sha256'] != digest for m in selected):
            raise ValueError(f'Source dataset changed: {source}')
        source_hashes[str(source)] = digest
        rows = [__import__('json').loads(line) for line in source.read_text().splitlines()]
        target = root / 'datasets/65536' / task / 'validation.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        datasets.append(str(target))
        for ordinal, meta in enumerate(selected):
            row = rows[ordinal]
            if row['outputs'] != meta['source_metadata']['references']:
                raise ValueError('Reference mismatch with prior source row')
            sample = encode(tokenizer, row, cfg, 65536, ordinal, task)
            sample['source_path'] = str(target)
            path = root / 'samples' / (sample['id'] + '.json')
            dump(path, sample)
            samples.append({k: v for k, v in sample.items() if k != 'token_ids'} | dict(input_path=str(path)))
    preserved.update(source_hashes)
    shutil.copytree(PARENT / 'private_ucm', root / 'private_ucm', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    protocol = previous | dict(study='Qwen3-32B YaRN 2x NIAH and QA comparison', settings=cfg,
        cases=list(CASES), scope=cfg['scope'], samples=samples, measured_requests=480,
        datasets=datasets, runtime_versions=runtime, gpu_devices=devices, created_at=time.time(),
        qualification='Fresh native-prefix reference, 0/100% controls and A-B-A/fresh-engine isolation for all eight tasks')
    model_limit(protocol, samples[0])
    dump(root / 'protocol.json', protocol)
    dump(root / 'prompt_manifest.json', samples)
    dump(root / 'preparation.json', cfg)
    dump(root / 'experiment.json', dict(preserved_sha256=preserved, parent_root=str(PARENT),
        new_requests=480, unchanged_inference_sources=unchanged, changed_setting='Eight new tasks, source rows 0–9 each',
        source_dataset_sha256=source_hashes, created_at=time.time()))
    shutil.copytree(HERE, root / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {}
    for folder in ('source', 'samples', 'datasets', 'private_ucm'):
        for path in (root / folder).rglob('*'):
            if path.is_file():
                hashes[str(path.relative_to(root))] = sha256(path)
    for name in ('protocol.json', 'prompt_manifest.json', 'experiment.json'):
        hashes[name] = sha256(root / name)
    dump(root / 'provenance.json', dict(sha256=hashes, frozen_at=time.time()))
    verify(cfg, root)
    verify_frozen(root)
    progress(root, protocol, 'prepared')
    print(f'Prepared 480 measurements; prompt lengths {min(s["tokens"] for s in samples)}–{max(s["tokens"] for s in samples)}; {len(preserved)} preserved artifacts', flush=True)
    return protocol

