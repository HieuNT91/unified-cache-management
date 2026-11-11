"""Owned process orchestration for AdaTree runs; no search or training CLI."""
import csv
import re
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from runner.cache.setups import atomic_json, file_hash
from runner.router.process import identity, alive, group_alive
from runner.corpus.records import accepted, publish, record_dir
from runner.operations import code_hashes, environment, terminate_owned, cleanup_caches, idle, check_hardware
from scripts.collection.longbench_a800_data import read
from scripts.collection.collection_v3 import seeds, ordered_rows
ROOT = Path(__file__).resolve().parents[2]

def resolve_groups(devices, expected, tp=2, query_inventory=True):
    tokens = [v.strip() for v in devices.split(',')]
    if len(tokens) != expected or len(set(tokens)) != expected:
        raise ValueError(f'Expected {expected} distinct devices')
    if not query_inventory:
        # L40 assignments are supplied as full UUIDs; hardware is user-managed.
        if expected % tp or any(not re.fullmatch(
                r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', v) for v in tokens):
            raise ValueError('Provide distinct full GPU UUIDs in complete TP groups')
        return [tokens[i:i+tp] for i in range(0, expected, tp)]
    # Legacy configure resolves identities; availability/memory gates run at launch.
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    inventory = {r[0].strip(): r[1].strip() for r in csv.reader(raw.splitlines())}
    values = [inventory.get(v, v) for v in tokens]
    if (len(set(values)) != expected or any(v not in inventory.values() or not re.fullmatch(
            r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', v) for v in values)):
        raise ValueError('Unknown or duplicate GPU identity')
    return [values[i:i+tp] for i in range(0, len(values), tp)]

def progress(base, role, message, group=0):
    atomic_json(Path(base)/role/f'progress-group{group}.json', dict(at=time.time(), message=message))
    print(message, flush=True)

def pending(protocol, rows, root, cases):
    return [r for r in rows if any(accepted(root, c, r, protocol, **({'seed': seed} if seed is not None else {})) is None
                                  for c in cases for seed in seeds(protocol, c))]

def worker(base, role, phase, attempt, group=0, *, stage_loader=None):
    from runner.corpus.runtime import Engine
    from runner.cache.setups import check_environment
    target = 'features' if phase == 'features' else role
    protocol, rows = stage_loader(base, target)
    root = Path(base)/target
    cases = (['probe'] if phase == 'features' else ['nocache'] if phase == 'baseline'
             else [c for c in protocol['scheduled_actions'] if c != 'nocache'])
    rows = ordered_rows(rows, protocol, group)
    remaining = pending(protocol, rows, root, cases)
    if not remaining:
        return
    if check_environment(protocol['tp']) != protocol['groups'][group]:
        raise ValueError('Worker GPU UUIDs differ from the frozen group')
    with (root/f'group{group}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(root, protocol, group, 'baseline' if phase == 'baseline' else 'cached', attempt, remaining)
        definitions = {a['id']: a for a in protocol['actions']}
        failure = None
        case = None
        try:
            for row in remaining:
                case = None
                progress(base, role, f"{phase}: starting {row['id']}", group)
                engine.begin(row)
                for case in cases:
                    for seed in seeds(protocol, case):
                        kwargs = {'seed': seed} if seed is not None else {}
                        if accepted(root, case, row, protocol, **kwargs) is not None:
                            continue
                        folder = record_dir(root, case, row['id'], **kwargs)
                        if folder.exists():
                            history = root/'incomplete'/f'{case}-{row["id"]}-{seed}-{uuid.uuid4().hex}'
                            history.parent.mkdir(exist_ok=True); folder.rename(history)
                        if case == 'probe':
                            result, diagnostics, _ = engine.probe(folder)
                        else:
                            result, diagnostics = engine.answer(definitions[case], case, **kwargs)
                        publish(root, case, row, protocol, result, diagnostics, **kwargs)
                        progress(base, role, f"{case}: accepted {row['id']} seed={seed}", group)
                deletion = engine.end()
                atomic_json(root/'deletions'/f'{phase}-{row["id"]}.json', deletion)
        except BaseException as error:
            failure = error
            try:
                path = engine.record_failure(error, phase, case)
                print(f'Worker failure receipt: {path}', flush=True)
            except Exception as diagnostic_error:
                print(f'Could not save worker failure receipt: {diagnostic_error}', file=sys.stderr, flush=True)
            raise
        finally:
            try:
                engine.close()
            except Exception as shutdown_error:
                if failure is None:
                    raise
                print(f'Engine shutdown also failed: {shutdown_error}', file=sys.stderr, flush=True)

def hardware(protocol):
    if protocol.get('run_profile', protocol.get('reproduction')) == 'adatree-v1.1':
        # Explicit profile validates each assigned UUID and free memory. Busy
        # devices abort the launch; the controller never cancels external jobs.
        return check_hardware(protocol)
    if protocol.get('hardware_profile') == 'rtx4500ada-tree-validation':
        if (protocol.get('schema') != 'tree-prophetkv-thinking-collection-v1'
                or not protocol.get('local_validation') or protocol.get('tp') != 4
                or len(protocol.get('groups', [])) != 1):
            raise ValueError('Local tree validation requires an explicit single TP4 group')
        return check_hardware(protocol)
    if protocol.get('hardware_profile') in ('l40-tp2', 'l40-tp4'):
        from runner.runtime.tensor_parallel import protocol_tp
        from runner.router.profiles import hardware_profile
        tp = 2 if protocol['hardware_profile'] == 'l40-tp2' else 4
        if (protocol_tp(protocol) != tp or len(protocol['groups']) != 8//tp
                or protocol.get('execution_profile') != 'ruler-thinking'
                or protocol.get('hardware_preflight') != 'user-managed'):
            raise ValueError('Invalid L40 thinking deployment')
        # User explicitly owns device availability checks. Do not query GPUs or
        # claim measured memory/identity evidence in this preflight receipt.
        profile = hardware_profile(protocol['hardware_profile'], 'ruler-thinking', tp)
        return dict(hardware_preflight='user-managed', configured_groups=protocol['groups'],
                    tp=tp, gpu_memory_utilization=profile['memory'], kv_cache_mode=profile['kv_cache_mode'])
    receipt = check_hardware(protocol)
    name = 'L20' if protocol['dataset'] == 'ruler' else 'A800'
    if any(name not in device['name'] for group in receipt['groups'] for device in group):
        raise ValueError(f'This deployment requires {name} GPUs; no local GPU fallback')
    return receipt

def engines_idle(root):
    for path in [*(root/'processes').glob('*.json'), *(root/'sessions').glob('*/ownership.json')]:
        saved = read(path)
        if alive(saved) or group_alive(saved['pid']):
            raise RuntimeError('An owned worker/engine remains alive')
    for path in root.glob('group*.lock'):
        with path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

def worker_failure(receipt, code, reason=None):
    """Surface the worker's evidence instead of replacing it with an exit code.

    Read a bounded suffix: full vLLM logs can be large, and diagnostic failures
    must never hide the original worker exit or interfere with peer cleanup.
    """
    prefix = reason or f"{receipt['phase']} group{receipt['group']} failed ({code})"
    path = Path(receipt['log'])
    try:
        with path.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size-32768))
            content = stream.read(32768).decode('utf-8', errors='replace')
        excerpt = '\n'.join(content.splitlines()[-80:])
        detail = f'Worker log tail (up to 80 lines / 32 KiB):\n{excerpt}' if excerpt else 'Worker log is empty.'
    except OSError as error:
        detail = f'Could not read worker log: {error}'
    return RuntimeError(f'{prefix}; accepted results retained.\n'
                        f'Worker PID: {receipt["pid"]}; GPU UUIDs: {", ".join(receipt["gpu_uuids"])}\n'
                        f'Worker log: {path}\n{detail}\n'
                        'Inspect the worker error before resuming; the exit code alone does not identify the cause.')

def execute_phase(base, role, phase, state, *, run_loader=None, stage_loader=None, entrypoint=None):
    settings, _, _ = run_loader(base)
    target = 'features' if phase == 'features' else role
    protocol, rows = stage_loader(base, target)
    cases = ['probe'] if phase == 'features' else ['nocache'] if phase == 'baseline' else [c for c in protocol['scheduled_actions'] if c != 'nocache']
    if not pending(protocol, rows, base/target, cases):
        engines_idle(base/target)
        return
    engines_idle(base/target); engines_idle(base/role)
    atomic_json(base/target/'hardware.json', hardware(protocol))
    cleanup_caches(base/target, protocol)
    children = []; attempt = uuid.uuid4().hex
    try:
        for group, devices in enumerate(protocol['groups']):
            subset = [r for r in rows if r['ordinal'] % len(protocol['groups']) == group]
            if not pending(protocol, subset, base/target, cases):
                continue
            command = [sys.executable, '-u', str(Path(entrypoint or __file__).resolve()), 'worker', '--root', str(base),
                       '--role', role, '--phase', phase, '--attempt', attempt, '--group', str(group)]
            log_path = base/role/f'{phase}-{attempt}-group{group}.log'
            with log_path.open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment(','.join(devices)),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt = dict(pid=child.pid, identity=identity(child.pid), phase=phase, group=group,
                           command=command, log=str(log_path), gpu_uuids=list(devices))
            children.append((child, receipt))
            # Keep feature worker ownership in the launching role for stop/resume.
            atomic_json(base/role/'processes'/f'{attempt}-group{group}.json', receipt)
        state.update(phase=phase, workers=[r for _, r in children]); atomic_json(base/role/'supervisor.json', state)
        last = {r['group']: time.monotonic() for _, r in children}; changed = dict.fromkeys(last, 0)
        while True:
            running = False
            for child, receipt in children:
                code = child.poll(); group = receipt['group']
                if code is not None:
                    if code:
                        state['failed_worker'] = dict(receipt, exit_code=code)
                        raise worker_failure(receipt, code)
                    continue
                running = True
                path = base/role/f'progress-group{group}.json'
                if path.exists() and path.stat().st_mtime_ns > changed[group]:
                    changed[group] = path.stat().st_mtime_ns; last[group] = time.monotonic()
                if time.monotonic()-last[group] > settings['watchdog_seconds']:
                    state['failed_worker'] = dict(receipt, exit_code=None, reason='progress watchdog expired')
                    raise worker_failure(receipt, None, reason=f'Progress watchdog expired for group{group}')
            if not running:
                break
            time.sleep(2)
    finally:
        for child, receipt in children:
            if child.poll() is None or group_alive(child.pid):
                terminate_owned(receipt)
            child.wait(timeout=10)
            atomic_json(base/role/'exits'/f'{attempt}-group{receipt["group"]}.json', dict(receipt,
                        exit_code=child.returncode, owned_engines_exited=not group_alive(child.pid)))
    engines_idle(base/role); engines_idle(base/target)
    cleanup_caches(base/target, protocol)
    if pending(protocol, rows, base/target, cases):
        raise RuntimeError('Workers exited without all scheduled records')

def detach(base, role, resume=False, cached_only=False, *, stage_loader=None, entrypoint=None):
    stage_loader(base, role)
    target = base/role
    with (target/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle(target)
        if role != 'extra':
            idle(base/'features')
        if (target/'complete.json').exists():
            raise ValueError('Completed launcher scope cannot be restarted')
        if (target/'supervisor.json').exists() and not resume:
            raise ValueError('Existing attempt; use resume for missing work')
        command = ['nohup', sys.executable, '-u', str(Path(entrypoint or __file__).resolve()), 'supervise', '--root', str(base), '--role', role]
        if cached_only:
            command.append('--cached-only')
        helper = '''import json,subprocess,sys
c=json.load(sys.stdin)
with open(c['log'],'a') as log:
 p=subprocess.Popen(c['command'],cwd=c['cwd'],env=c['env'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid,flush=True)
'''
        result = subprocess.run([sys.executable, '-c', helper], input=json.dumps(dict(command=command,
            cwd=str(ROOT), env=environment(), log=str(target/'supervisor.log'))), text=True, capture_output=True, check=True)
        pid = int(result.stdout.strip())
        atomic_json(target/'launch.json', dict(pid=pid, identity=identity(pid), command=command, resume=resume))
        for _ in range(200):
            path = target/'supervisor.json'
            if path.exists() and read(path)['pid'] == pid:
                break
            if identity(pid) is None:
                raise RuntimeError('Supervisor exited; inspect supervisor.log')
            time.sleep(.1)
        else:
            raise RuntimeError('Startup receipt timed out; inspect before retrying')
        proc = Path('/proc')/str(pid)
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        ignored = int(next(s.split()[1] for s in (proc/'status').read_text().splitlines() if s.startswith('SigIgn:')), 16)
        if (int(fields[1]) != 1 or int(fields[3]) != pid or not ignored & 1
                or os.readlink(proc/'fd/0') != '/dev/null'
                or b'CUDA_VISIBLE_DEVICES=' not in (proc/'environ').read_bytes().split(b'\0')):
            raise RuntimeError('Detachment verification failed; inspect before retrying')
        atomic_json(target/'detachment.json', dict(pid=pid, identity=identity(pid), parent_pid=1, session_id=pid,
            sighup_ignored=True, stdin='/dev/null', cpu_only=True, resume=resume))
        print(f'Detached {role} CPU supervisor PID={pid}; log: {target}/supervisor.log', flush=True)
