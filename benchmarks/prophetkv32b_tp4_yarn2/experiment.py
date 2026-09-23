"""Freeze an independent six-method YaRN 2x experiment and its 4x comparison."""
import hashlib
import os
from pathlib import Path
import shutil
import time

from common import HERE, REPO, CASES, dump, load, identity, rope_for_length, model_limit

ORIGINAL = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-20260921'
EXTENSION = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-extension-20260921'
PREVIOUS_SOURCE = REPO / 'benchmarks/prophetkv32b_tp4_extension'


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
             'prophetkv_common.py', 'lifecycle.py', 'persistent_connector.py']
    names += [str(p.relative_to(HERE)) for p in (HERE / 'method').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for name in names:
        if (HERE / name).read_bytes() != (PREVIOUS_SOURCE / name).read_bytes():
            raise ValueError(f'Inference implementation changed: {name}')
    new = (HERE / 'memory_worker.py').read_text()
    insertion = ('        from rope_window import ensure_rope_window\n'
                 '        rope_audits = [ensure_rope_window(layer.self_attn.rotary_emb, self.model_config.max_model_len)\n'
                 '                       for layer in model.model.layers]\n')
    normalized = new.replace(insertion, '').replace('layers=audits, rope_windows=rope_audits))', 'layers=audits))')
    if normalized != (PREVIOUS_SOURCE / 'memory_worker.py').read_text():
        raise ValueError('Memory worker changes extend beyond the position table')
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
    from suite import check_orphan, progress, verify
    if (root / 'protocol.json').exists():
        verify_frozen(root)
        return verify(cfg, root)
    for parent in (ORIGINAL, EXTENSION):
        check_orphan(parent)
        for name in ('supervisor.json', 'reporter.json'):
            state = load(parent / name)
            if identity(state['pid']) == state['identity']:
                raise RuntimeError(f'Previous process still running: {parent / name}')
        if not load(parent / 'final/integrity-validation.json')['complete']:
            raise ValueError('Previous experiment is incomplete')
    preserved = {str(ORIGINAL / name): digest for name, digest in load(EXTENSION / 'extension.json')['parent_sha256'].items()}
    preserved.update({str(EXTENSION / name): digest for name, digest in load(EXTENSION / 'provenance.json')['sha256'].items()})
    check_hashes(preserved)
    for folder in ('records', 'final', 'combined'):
        for path in (EXTENSION / folder).rglob('*'):
            if path.is_file():
                preserved[str(path)] = sha256(path)
    preserved[str(REPO / 'qwen3_32b_results.txt')] = sha256(REPO / 'qwen3_32b_results.txt')
    unchanged = verify_unchanged_inference()
    original = load(ORIGINAL / 'protocol.json')
    for key in ('model', 'samples', 'chunk', 'indices', 'memory', 'scope'):
        if cfg[key] != original['settings'][key]:
            raise ValueError(f'Settings changed beyond requested RoPE factor: {key}')
    for folder in ('samples', 'datasets', 'private_ucm'):
        shutil.copytree(ORIGINAL / folder, root / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    samples = [m | dict(input_path=str(root / 'samples' / (m['id'] + '.json'))) for m in original['samples']]
    for meta in samples:
        if Path(meta['input_path']).read_bytes() != (ORIGINAL / 'samples' / (meta['id'] + '.json')).read_bytes():
            raise ValueError('Prompt bytes changed')
    protocol = original | dict(study='Qwen3-32B ProphetKV six-method YaRN 2x comparison',
        settings=cfg, cases=list(CASES), samples=samples, measured_requests=120,
        datasets=[str(root / Path(path).relative_to(ORIGINAL)) for path in original['datasets']],
        rope_scaling_64k=rope_for_length(65536), created_at=time.time(),
        nominal_yarn_context=65536, preserved_engine_window=65792,
        position_table_extension='Append 256 positions with unchanged factor-2 frequencies and magnitude; original 65536 entries bitwise preserved',
        qualification='Fresh native-prefix reference, 0/100% controls, A-B-A/fresh engines and 64-layer tensor audits under factor 2')
    if model_limit(protocol, samples[0]) != 65792:
        raise ValueError('Engine allocation changed')
    dump(root / 'protocol.json', protocol)
    dump(root / 'prompt_manifest.json', samples)
    dump(root / 'preparation.json', cfg)
    dump(root / 'experiment.json', dict(preserved_sha256=preserved, original_root=str(ORIGINAL),
        extension_root=str(EXTENSION), comparison=str(EXTENSION / 'combined/raw_records.jsonl'),
        new_requests=120, comparison_requests=120, unchanged_inference_sources=unchanged,
        changed_setting='rope_scaling.factor: 4.0 -> 2.0', created_at=time.time(),
        window_note='Identical 65216–65664-token prompts and 128-token cap require 65792 table positions; last 256 extrapolate beyond nominal 2x window'))
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
    print(f'Prepared 120 YaRN-2x requests; {len(preserved)} preserved YaRN-4x artifacts', flush=True)
