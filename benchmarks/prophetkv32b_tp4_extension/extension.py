"""Preserve the completed cohort and prepare only the three added ratios."""
import hashlib
from pathlib import Path
import shutil
import time

from common import HERE, REPO, CASES, ENGINE_POLICY, dump, load, identity

PARENT = REPO / '.results/qwen3-32b-prophetkv-vt-cwe10-64k-c4096-tp4-20260921'
PARENT_SOURCE = REPO / 'benchmarks/prophetkv32b_tp4'
ALL_CASES = ('baseline', 'prophetkv-5', 'prophetkv-20', *CASES)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(part)
    return digest.hexdigest()


def check_hashes(hashes, base):
    for name, expected in hashes.items():
        if sha256(base / name) != expected:
            raise ValueError(f'Frozen artifact changed: {base / name}')


def verify_parent(root):
    amendment = load(root / 'extension.json')
    if amendment['parent_root'] != str(PARENT):
        raise ValueError('Unexpected parent cohort')
    check_hashes(amendment['parent_sha256'], PARENT)
    return amendment


def verify_frozen(root):
    verify_parent(root)
    check_hashes(load(root / 'provenance.json')['sha256'], root)
    for frozen in (root / 'source').rglob('*'):
        if frozen.is_file() and (HERE / frozen.relative_to(root / 'source')).read_bytes() != frozen.read_bytes():
            raise ValueError(f'Executed source differs from snapshot: {frozen}')


def verify_unchanged_inference():
    names = ['worker.py', 'persistent_worker.py', 'cacheblend_ruler.py',
             'prophetkv_common.py', 'lifecycle.py', 'persistent_connector.py', 'reference.py']
    names += [str(p.relative_to(HERE)) for p in (HERE / 'method').rglob('*') if p.is_file()]
    for name in names:
        if (HERE / name).read_bytes() != (PARENT_SOURCE / name).read_bytes():
            raise ValueError(f'Inference implementation changed: {name}')
    old = (PARENT_SOURCE / 'memory_worker.py').read_text()
    new = (HERE / 'memory_worker.py').read_text()
    if old.split('        audits = []', 1)[1] != new.split('        audits = []', 1)[1]:
        raise ValueError('Memory adaptation changed')
    if old.split('class Worker', 1)[0] != new.split('class Worker', 1)[0]:
        raise ValueError('Tiling arithmetic changed')
    return names


def prepare_extension(cfg, root):
    from suite import check_orphan, progress, verify
    if (root / 'protocol.json').exists():
        verify_frozen(root)
        return verify(cfg, root)
    check_orphan(PARENT)
    for name in ('supervisor.json', 'reporter.json'):
        state = load(PARENT / name)
        if identity(state['pid']) == state['identity']:
            raise RuntimeError(f'Parent {name} is still live')
    original = load(PARENT / 'protocol.json')
    if not load(PARENT / 'final/integrity-validation.json')['complete']:
        raise ValueError('Parent cohort did not validate')
    parent_hashes = dict(load(PARENT / 'provenance.json')['sha256'])
    check_hashes(parent_hashes, PARENT)
    unchanged = verify_unchanged_inference()
    for folder in ('records', 'final'):
        for path in (PARENT / folder).rglob('*'):
            if path.is_file():
                parent_hashes[str(path.relative_to(PARENT))] = sha256(path)
    for task in ('vt', 'cwe'):
        gate = PARENT / 'smoke/65536' / task
        certificate = load(gate / 'validation.json')
        if not certificate['complete'] or certificate['comparisons'] != [dict(rank=r, bitwise_equal_layers=64) for r in range(4)]:
            raise ValueError('Missing inherited native-prefix qualification')
        for path in gate.rglob('*'):
            if path.is_file() and (path.suffix in ('.json', '.log') or path.parent == gate):
                parent_hashes[str(path.relative_to(PARENT))] = sha256(path)
    for folder in ('samples', 'datasets', 'private_ucm'):
        shutil.copytree(PARENT / folder, root / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    samples = [m | dict(input_path=str(root / 'samples' / (m['id'] + '.json')))
               for m in original['samples']]
    for meta in samples:
        old_path = PARENT / 'samples' / (meta['id'] + '.json')
        if Path(meta['input_path']).read_bytes() != old_path.read_bytes():
            raise ValueError('Prompt bytes differ from parent')
    protocol = original | dict(study='Qwen3-32B ProphetKV 30/40/50% extension',
        settings=cfg, cases=list(CASES), samples=samples,
        datasets=[str(root / Path(path).relative_to(PARENT)) for path in original['datasets']],
        measured_requests=60, created_at=time.time(), parent_root=str(PARENT),
        qualification='Inherited native-prefix controls; fresh A-B-A/fresh-engine and 64-layer tensor audits for each added ratio and task')
    dump(root / 'protocol.json', protocol)
    dump(root / 'prompt_manifest.json', samples)
    dump(root / 'preparation.json', cfg)
    retained = root / 'retained'
    retained.mkdir()
    shutil.copy2(REPO / 'qwen3_32b_results.txt', retained / 'qwen3_32b_results.txt')
    dump(root / 'extension.json', dict(parent_root=str(PARENT), parent_sha256=parent_hashes,
        retained_requests=60, added_requests=60, combined_requests=120,
        retained_cases=original['cases'], added_cases=list(CASES),
        unchanged_inference_sources=unchanged, created_at=time.time(),
        memory_worker_change='Remove unused sharded export before the unchanged numerical audit and tiling; original checkpoint loader remains unchanged',
        timing_caveat='Original measurements retained; added ratios measured in a later sequential cohort'))
    shutil.copytree(HERE, root / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {}
    for folder in ('source', 'samples', 'datasets', 'private_ucm', 'retained'):
        for path in (root / folder).rglob('*'):
            if path.is_file():
                hashes[str(path.relative_to(root))] = sha256(path)
    for name in ('protocol.json', 'prompt_manifest.json', 'extension.json'):
        hashes[name] = sha256(root / name)
    dump(root / 'provenance.json', dict(sha256=hashes, frozen_at=time.time()))
    verify(cfg, root)
    progress(root, protocol, 'prepared', retained=60, combined_target=120)
    print(f'Prepared 60 added requests; {len(parent_hashes)} preserved parent artifacts', flush=True)
    return protocol
