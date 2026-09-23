"""CPU-only reporting recovery; preserve frozen inference sources and records."""
import hashlib
import inspect
import os
from pathlib import Path
import shutil
import sys
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('Reporting must hide all GPUs')
HERE = Path(__file__).resolve().parent / 'prophetkv32b_yarn2_niahqa_no_smoke'
sys.path.insert(0, str(HERE))
import suite
import finalize
from common import settings, load, dump, identity
from experiment import verify_frozen

root = Path(settings()['root'])
recovery = root / 'report-recovery'
recovery.mkdir(exist_ok=True)
for name in ('supervisor.json', 'reporter.json'):
    state = load(root / name)
    if identity(state['pid']) == state['identity']:
        raise RuntimeError(f'Process remains live: {name}')
suite.check_orphan(root)
verify_frozen(root)
protocol = load(root / 'protocol.json')
assert protocol['qualification'] == 'Disabled by explicit user request; measured sessions only'
for name in ('progress.json', 'reporter.json', 'reporting.log'):
    target = recovery / ('before-' + name)
    if not target.exists():
        shutil.copyfile(root / name, target)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

records = {str(p): digest(p) for p in (root / 'records').glob('*/*.json')
           if p.name in [case + '.json' for case in protocol['cases']]}
assert len(records) == 480
source = inspect.getsource(suite.report)
old = """        for unit in p['scope']:
            length, task = unit['length'], unit['task']
            if not valid_gate(root, p, length, task):
                raise ValueError('Missing smoke gate')
"""
new = """        if p.get('qualification') != 'Disabled by explicit user request; measured sessions only':
            raise ValueError('Qualification waiver is absent from the frozen protocol')
"""
assert source.count(old) == 1
source = source.replace(old, new).replace(
    'Smoke controls (0/100% plus native dense exact-prefix reference) are excluded from measured results. ',
    'Qualification/smoke controls were disabled by explicit user request and were not run. ')
(recovery / 'report_function.py').write_text(source)
namespace = dict(suite.__dict__)
exec(compile(source, str(recovery / 'report_function.py'), 'exec'), namespace)
finalize.report = namespace['report']
finalize.main()
assert all(digest(Path(name)) == expected for name, expected in records.items())
certificate = load(root / 'final/validation.json')
certificate.update(qualification='not run: explicitly waived by user', reporting_recovery=str(recovery))
dump(root / 'final/validation.json', certificate)
state = load(root / 'reporter.json')
state.pop('error', None)
dump(root / 'reporter.json', state | dict(state='complete', recovered_at=time.time(), recovery=str(recovery)))
dump(recovery / 'validation.json', dict(complete=True, measured_records_unchanged=480,
    record_sha256=records, recovery_script_sha256=digest(Path(__file__)),
    report_function_sha256=digest(recovery / 'report_function.py'),
    inference_rerun=False, completed_at=time.time()))
