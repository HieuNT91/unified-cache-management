#!/usr/bin/env python3
"""Create an audited source copy for a failed L40 cache-staging run; never launch it.

Frozen settings, protocols, prepared inputs and accepted records are not edited.
Only the three literal substitutions below are allowed in the recovery copy.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

SELF = 'scripts/cache_staging_recovery.py'
RECEIPT = '.cache-staging-recovery.json'
OLD_READY = """    def ready(self):
        from runner.cache import wait_for_cache
        result = wait_for_cache(self.args, self.chunks)
        self.before = self.snapshot()
        actual = {str(p.relative_to(self.cache)) for p in (self.cache/'kv').rglob('*') if p.is_file()}
        if actual != self.files:
            extra, missing = sorted(actual-self.files), sorted(self.files-actual)
            raise RuntimeError('Unexpected files in temporary prompt cache: '
                f'cache={self.cache}, expected={len(self.files)}, actual={len(actual)}, '
                f'extra={len(extra)}, missing={len(missing)}, '
                f'extra_examples={extra[:10]}, missing_examples={missing[:10]}')
        return result
"""
NEW_READY = """    def ready(self):
        from runner.cache import wait_for_cache
        started = time.monotonic()
        self.before = None
        result = wait_for_cache(self.args, self.chunks)
        # PcStore 0.3.0 completes dump tasks after device-to-host transfer;
        # the file queue may still be writing/renaming repeated prefix blocks.
        # Wait only for staging files belonging to this prompt, never ignore
        # or remove them, and take the immutable snapshot after publication.
        staging = {f'kv/.temp/{Path(name).name}' for name in self.files}
        while True:
            paths = list((self.cache/'kv').rglob('*'))
            if any(p.is_symlink() for p in paths):
                raise RuntimeError('Symlink in temporary cache')
            actual = {str(p.relative_to(self.cache)) for p in paths if p.is_file()}
            extra, missing = sorted(actual-self.files), sorted(self.files-actual)
            if not extra and not missing:
                # Recheck sizes after the last rename, including replacements.
                from runner.cache import verify_cache
                result.update(verify_cache(self.args, self.chunks))
                self.before = self.snapshot()
                result['readiness_wait_seconds'] = time.monotonic() - started
                return result
            detail = (
                f'cache={self.cache}, expected={len(self.files)}, actual={len(actual)}, '
                f'extra={len(extra)}, missing={len(missing)}, '
                f'extra_examples={extra[:10]}, missing_examples={missing[:10]}')
            if missing or not set(extra).issubset(staging):
                raise RuntimeError('Unexpected files in temporary prompt cache: ' + detail)
            remaining = self.args.cache_ready_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise RuntimeError('Cache staging did not drain before readiness timeout: ' + detail)
            time.sleep(min(0.25, remaining))
"""
OLD_GUARD = """        if settings['code'] != code_hashes():
            raise ValueError('Implementation changed; keep the original checkout for resume')"""
NEW_GUARD = """        from scripts.cache_staging_recovery import validate_recovery
        validate_recovery(base, settings, code_hashes())"""
OLD_INIT = "            init['validated']=True;atomic_json(self.session/'initialization.json',init)"
NEW_INIT = """            from scripts.cache_staging_recovery import initialization_receipt
            init['cache_staging_recovery']=initialization_receipt(self.root.parent)
            init['validated']=True;atomic_json(self.session/'initialization.json',init)"""
PATCHES = {'runner/sweep.py': (OLD_READY, NEW_READY),
           'scripts/longbench_a800_data.py': (OLD_GUARD, NEW_GUARD),
           'runner/corpus_runtime.py': (OLD_INIT, NEW_INIT)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def substitute(data, before, after):
    before, after = before.encode(), after.encode()
    if data.count(before) != 1:
        raise ValueError('Unsupported source version: expected exactly one known patch site')
    return data.replace(before, after)


def source_file(root, name):
    path = Path(root)/name
    if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Source path escapes checkout')
    return path


def source_hashes(root):
    paths = [root/'run.py', root/'run.sh']
    for folder in ('runner', 'ucm', 'scripts'):
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.sh', '.json')
                     and '__pycache__' not in p.parts and 'vendor' not in p.parts)
    return {str(p.relative_to(root)): digest(p.read_bytes()) for p in sorted(paths)}


def validate_recovery(base, settings, actual, checkout=None):
    checkout = Path(checkout or Path(__file__).resolve().parents[1]).resolve()
    base = Path(base).resolve()
    receipt = read(checkout/RECEIPT)
    if (receipt['schema'] != 'cache-staging-recovery-v1' or receipt['root'] != str(base)
            or receipt['checkout'] != str(checkout)
            or receipt['settings_sha256'] != digest((base/'settings.json').read_bytes())
            or receipt['plan_sha256'] != digest((base/'plan.json').read_bytes())):
        raise ValueError('Recovery experiment identity changed')
    saved = base/'runtime-recoveries'/receipt['id']/'receipt.json'
    if read(saved) != receipt or receipt['helper_sha256'] != digest((checkout/SELF).read_bytes()):
        raise ValueError('Recovery receipt/helper changed')
    expected = dict(settings['code'])
    if SELF in expected or set(actual) != set(expected) | {SELF}:
        raise ValueError('Recovery source inventory changed')
    patched = {}
    for name, (before, after) in PATCHES.items():
        data = (checkout/name).read_bytes()
        original = substitute(data, after, before)
        if digest(original) != expected[name]:
            raise ValueError('Recovery changed source outside approved patch: ' + name)
        patched[name] = {'before': expected[name], 'after': digest(data)}
        expected[name] = digest(data)
    expected[SELF] = receipt['helper_sha256']
    if actual != expected or receipt['patches'] != patched:
        raise ValueError('Recovery source changed')
    if digest((base/'runtime-recoveries'/receipt['id']/'failure.json').read_bytes()) != receipt['failure_sha256']:
        raise ValueError('Recovery failure evidence changed')
    return receipt


def initialization_receipt(base):
    from scripts.router_control import code_hashes
    receipt = validate_recovery(base, read(Path(base)/'settings.json'), code_hashes())
    path = Path(base)/'runtime-recoveries'/receipt['id']/'receipt.json'
    return dict(id=receipt['id'], receipt=str(path), sha256=digest(path.read_bytes()),
                patches=receipt['patches'], helper_sha256=receipt['helper_sha256'])


def validate_failure(data):
    failure = json.loads(data)
    inv = failure.get('inventory', {})
    extra, expected = inv.get('extra', []), set(inv.get('expected_files', []))
    if (failure.get('phase') != 'cached' or not failure.get('prompt_id')
            or not failure.get('error', '').startswith('Unexpected files in temporary prompt cache:')
            or not extra or inv.get('missing') or inv.get('inventory_errors')):
        raise ValueError('Expected a cached-phase staging-only failure receipt')
    for name in extra:
        parts = Path(name).parts
        if len(parts) != 3 or parts[:2] != ('kv', '.temp'):
            raise ValueError('Failure contains unexpected non-staging files')
        key = parts[-1]
        target = f'kv/{key[:8]}/{key}'
        info = inv.get('actual_files', {}).get(name, {})
        if (len(key) != 32 or any(c not in '0123456789abcdef' for c in key)
                or target not in expected or info.get('symlink') is not False):
            raise ValueError('Failure contains unknown or symlinked staging files')
    return failure


def create(source, destination, base, failure_path):
    source, destination, base = (Path(p).resolve() for p in (source, destination, base))
    settings = read(base/'settings.json')
    if (settings.get('dataset') != 'ruler' or settings.get('execution_profile') != 'ruler-thinking'
            or settings.get('hardware_profile') != 'l40-tp2' or settings.get('tp') != 2):
        raise ValueError('Recovery is restricted to L40 TP2 thinking RULER')
    if (base/'ruler/complete.json').exists():
        raise ValueError('Completed runs cannot be recovered')
    if source_hashes(source) != settings['code']:
        raise ValueError('Original checkout differs from frozen source; do not pull or rewrite pins')
    protected = [source, base] + [Path(settings[k]).resolve() for k in ('prepared', 'model', 'cache_root')]
    if destination.exists() or any(destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise ValueError('Use a new separate recovery source directory')
    failure_path = Path(failure_path).resolve()
    if not failure_path.is_relative_to(base/'ruler'/'sessions'):
        raise ValueError('Failure receipt must belong to this run')
    failure_bytes = failure_path.read_bytes()
    validate_failure(failure_bytes)
    files = {name: source_file(source, name).read_bytes() for name in settings['code']}
    for name, (before, after) in PATCHES.items():
        files[name] = substitute(files[name], before, after)
    files[SELF] = Path(__file__).read_bytes()
    # Import process ownership checks from the verified original source only.
    sys.path.insert(0, str(source))
    from scripts.corpus_control import idle
    from scripts.longbench_a800_data import stage
    with ExitStack() as stack:
        for role in ('ruler', 'features'):
            lock = stack.enter_context((base/role/'launch.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            idle(base/role)
        stage(base, 'ruler', check_code=True)
        if source_hashes(source) != settings['code']:
            raise ValueError('Original source changed during recovery')
        receipt = dict(schema='cache-staging-recovery-v1', id=uuid.uuid4().hex,
            created_at_ns=time.time_ns(), root=str(base), checkout=str(destination),
            source_checkout=str(source), settings_sha256=digest((base/'settings.json').read_bytes()),
            plan_sha256=digest((base/'plan.json').read_bytes()), failure_sha256=digest(failure_bytes),
            helper_sha256=digest(files[SELF]),
            patches={name: dict(before=settings['code'][name], after=digest(files[name])) for name in PATCHES})
        destination.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        payload = json.dumps(receipt, indent=2, sort_keys=True)+'\n'
        (destination/RECEIPT).write_text(payload)
        evidence = base/'runtime-recoveries'/receipt['id']
        evidence.mkdir(parents=True, exist_ok=False)
        (evidence/'failure.json').write_bytes(failure_bytes)
        (evidence/'receipt.json').write_text(payload)
        validate_recovery(base, settings, source_hashes(destination), destination)
    print('Recovery source prepared:', destination)
    print('Original source, settings, plan, inputs and accepted results preserved.')
    print('No process launched. Resume with the recovery copy and the SAME --root:', base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkout', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--failure', type=Path, required=True)
    args = parser.parse_args()
    create(args.source_checkout, args.destination, args.root, args.failure)


if __name__ == '__main__':
    main()
