#!/usr/bin/env python3
"""A800/L20 TP2 and L40 thinking TP2/TP4 controls, then attention collection."""
import argparse
import csv
import re
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.setups import atomic_json, file_hash
from runner.router_process import identity, alive, group_alive
from runner.corpus_records import accepted, publish, record_dir
from scripts.router_control import code_hashes, environment, terminate_owned, cleanup_caches
from scripts.corpus_control import idle, check_hardware
from scripts.longbench_a800_data import read, freeze, load, stage, SCHEDULE, ACTIONS, publish_protocols, scheduled, ruler_samples


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


def configure(args):
    thinking = getattr(args,'thinking',False)
    requested_hardware = getattr(args, 'hardware_profile', None)
    l40 = requested_hardware in ('l40-tp2', 'l40-tp4')
    required_tp = 4 if requested_hardware == 'l40-tp4' else 2
    if args.tp != required_tp or args.seed != 42:
        raise ValueError(f'This collection requires TP{required_tp} and seed42')
    if l40 and (args.role != 'ruler' or not thinking):
        raise ValueError('L40 requires the thinking RULER profile')
    if thinking and args.role != 'ruler':
        raise ValueError('Thinking RULER requires the ruler role')
    extend_from = getattr(args, 'extend_from', None)
    if extend_from and not (args.role == 'ruler' and (
            (l40 and thinking and args.samples_per_task == 200) or
            (not l40 and not thinking and args.samples_per_task == 500))):
        raise ValueError('--extend-from requires L40 thinking target200 or L20 non-thinking target500')
    count = ruler_samples(dict(samples_per_task=args.samples_per_task, execution_profile='ruler-thinking' if thinking else 'ruler'))
    dataset = 'ruler' if args.role == 'ruler' else 'longbench-v2'
    selected = (resolve_groups(args.devices, 8, tp=required_tp, query_inventory=False) if l40
                else resolve_groups(args.devices, 10 if dataset == 'ruler' else 4))
    hardware_profile = requested_hardware if l40 else 'l20-tp2' if dataset == 'ruler' else 'server'
    settings = dict(schema='tp2-data-config-v2', dataset=dataset, tp=args.tp, model=str(args.model), data=str(args.data),
                    prepared=str(args.prepared), cache_root=str(args.cache_root),
                    samples_per_task=count if dataset == 'ruler' else 100, seed=42, hardware_profile=hardware_profile, code=code_hashes(), watchdog_seconds=7200, automatic_training=False)
    if l40:
        settings['hardware_preflight'] = 'user-managed'
    if thinking:
        from runner.thinking_budget import PROTOCOL, THINKING_CAP, WINDOW, SAMPLING
        from scripts.ruler import CAPS
        settings.update(execution_profile='ruler-thinking', evaluation_protocol=PROTOCOL,
                        thinking_cap=THINKING_CAP, answer_caps=CAPS, engine_window=WINDOW, sampling=SAMPLING)
    if extend_from:
        from scripts.ruler_extension import configure_extension
        settings['extension'] = configure_extension(args, settings, selected)
        settings['samples_per_task'] = settings['extension']['target_samples_per_task']-settings['extension']['start']
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'configure.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.role == 'extra' and not (args.root/'settings.json').exists():
            raise ValueError('Configure primary first')
        if not (args.root/'settings.json').exists() and any(p.name != 'configure.lock' for p in args.root.iterdir()):
            raise ValueError('Use a new empty experiment directory')
        freeze(args.root/'settings.json', settings)
        for other in args.root.glob('*/devices.json'):
            if other.parent.name != args.role and set(sum(read(other)['groups'], [])) & set(sum(selected, [])):
                raise ValueError('Primary/extra GPU assignments overlap')
        freeze(args.root/args.role/'devices.json', dict(tp=args.tp, groups=selected))
        if (args.root/'plan.json').exists():
            publish_protocols(args.root)
    print(f'Configured {args.role}: {len(selected)} TP{args.tp} groups; training remains a separate command.')


def progress(base, role, message, group=0):
    atomic_json(Path(base)/role/f'progress-group{group}.json', dict(at=time.time(), message=message))
    print(message, flush=True)


def pending(protocol, rows, root, cases):
    return [r for r in rows if any(accepted(root, c, r, protocol) is None for c in cases)]


def worker(base, role, phase, attempt, group=0):
    from runner.corpus_runtime import Engine
    from runner.setups import check_environment
    target = 'features' if phase == 'features' else role
    protocol, rows = stage(base, target, check_code=True)
    root = Path(base)/target
    cases = (['probe'] if phase == 'features' else ['nocache'] if phase == 'baseline'
             else [c for c in protocol['scheduled_actions'] if c != 'nocache'])
    rows = [r for r in rows if r['ordinal'] % len(protocol['groups']) == group]
    remaining = pending(protocol, rows, root, cases)
    if not remaining:
        return
    if check_environment(protocol['tp']) != protocol['groups'][group]:
        raise ValueError('Worker GPU UUIDs differ from the frozen group')
    with (root/f'group{group}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(root, protocol, group, 'baseline' if phase == 'baseline' else 'cached', attempt, remaining)
        definitions = {a['id']: a for a in ACTIONS}
        failure = None
        case = None
        try:
            for row in remaining:
                case = None
                progress(base, role, f"{phase}: starting {row['id']}", group)
                engine.begin(row)
                for case in cases:
                    if accepted(root, case, row, protocol) is not None:
                        continue
                    folder = record_dir(root, case, row['id'])
                    if folder.exists():
                        history = root/'incomplete'/f'{case}-{row["id"]}-{uuid.uuid4().hex}'
                        history.parent.mkdir(exist_ok=True); folder.rename(history)
                    if case == 'probe':
                        result, diagnostics, _ = engine.probe(folder)
                    else:
                        result, diagnostics = engine.answer(definitions[case], case)
                    publish(root, case, row, protocol, result, diagnostics)
                    progress(base, role, f"{case}: accepted {row['id']}", group)
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
    if protocol.get('hardware_profile') in ('l40-tp2', 'l40-tp4'):
        from runner.tensor_parallel import protocol_tp
        from runner.tree_profiles import hardware_profile
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


def execute_phase(base, role, phase, state):
    settings, _, _ = load(base, check_code=True)
    target = 'features' if phase == 'features' else role
    protocol, rows = stage(base, target)
    cases = ['probe'] if phase == 'features' else ['nocache'] if phase == 'baseline' else [c for c in scheduled(settings, role) if c != 'nocache']
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
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--root', str(base),
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


def wait_extra(base):
    """Primary owns no GPU engine while waiting; failed/stopped dependencies halt."""
    while True:
        path = base/'extra/supervisor.json'
        state = read(path) if path.exists() else None
        if (base/'extra/complete.json').exists():
            if state and alive(state):
                time.sleep(2); continue
            idle(base/'extra')
            return
        if state and (state.get('state') == 'failed' or not alive(state)):
            raise RuntimeError('Extra group failed/stopped; resume it, then resume primary')
        time.sleep(5)


def supervise(base, role):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state_root = base/role
    with (state_root/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), identity=identity(os.getpid()), state='running', role=role, started_at=time.time())
        atomic_json(state_root/'supervisor.json', state)
        try:
            # Do not ask idle() to lock our own run.lock during execute_phase.
            for phase in (('baseline', 'cached') if role != 'extra' else ('cached',)):
                execute_phase(base, role, phase, state)
            protocol, rows = stage(base, role)
            freeze(state_root/'controls-complete.json', dict(answers=len(rows)*len(protocol['scheduled_actions']),
                   protocol_sha256=file_hash(state_root/'protocol.json'), owned_engines_exited=True))
            if role != 'extra':
                state.update(state='waiting-for-extra', phase='waiting'); atomic_json(state_root/'supervisor.json', state)
                if role == 'primary':
                    wait_extra(base)
                from scripts.longbench_a800_report import report
                report(base, final=True, emit=False)
                state.update(state='collecting-features'); atomic_json(state_root/'supervisor.json', state)
                execute_phase(base, role, 'features', state)
                freeze(base/'features/complete.json', dict(probes=len(rows), owned_engines_exited=True,
                       protocol_sha256=file_hash(base/'features/protocol.json')))
                from scripts.router_dataset import export_collection
                export_collection(base, base/('ruler-data.json' if role == 'ruler' else 'longbench-data.json'))
            freeze(state_root/'complete.json', dict(complete=True, owned_engines_exited=True,
                   plan_sha256=file_hash(base/'plan.json'), feature_collection=role != 'extra'))
            state.update(state='complete', finished_at=time.time())
        except BaseException as error:
            state.update(state='failed', error=str(error), finished_at=time.time()); raise
        finally:
            atomic_json(state_root/'supervisor.json', state)


def detach(base, role, resume=False):
    stage(base, role, check_code=True)
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
        command = ['nohup', sys.executable, '-u', str(Path(__file__).resolve()), 'supervise', '--root', str(base), '--role', role]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('configure', 'prepare', 'detach', 'resume', 'stop', 'status', 'status_same_count', 'report', 'merge', 'supervise', 'worker'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--thinking', action='store_true', help='Separate thinking RULER protocol')
    parser.add_argument('--extend-from', type=Path, help='Parent RULER run: L40 thinking 30->200 or L20 non-thinking 200->500; collect only new ordinals')
    parser.add_argument('--hardware-profile', choices=('l40-tp2', 'l40-tp4'),
                        help='Eight explicit L40 UUIDs, thinking TP2/TP4; user checks GPU availability')
    parser.add_argument('--role', choices=('primary', 'extra', 'ruler'), required=True)
    for name in ('model', 'data', 'prepared', 'cache-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--devices', default='')
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--samples-per-task', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--group', type=int, default=0)
    parser.add_argument('--phase', choices=('baseline', 'cached', 'features'))
    parser.add_argument('--attempt')
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.command != 'worker' and os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise ValueError('Coordinator must be CPU-only')
    if args.command == 'configure':
        if any(getattr(args, k) is None for k in ('model', 'data', 'prepared', 'cache_root')):
            parser.error('configure requires model/data/prepared/cache-root paths')
        configure(args)
    elif args.command == 'prepare':
        settings = read(args.root/'settings.json')
        if settings['code'] != code_hashes():
            raise ValueError('Implementation changed since configure')
        from scripts.longbench_a800_data import prepare
        with (args.root/'prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(args.root)
    elif args.command in ('detach', 'resume'):
        detach(args.root, args.role, args.command == 'resume')
    elif args.command == 'merge':
        from scripts.ruler_extension import merge_extension
        merge_extension(args.root)
    elif args.command == 'stop':
        from scripts.launcher_stop import stop
        stop(args.root, Path(__file__).name, args.root/args.role)
    elif args.command == 'supervise':
        supervise(args.root, args.role)
    elif args.command == 'worker':
        worker(args.root, args.role, args.phase, args.attempt, args.group)
    else:
        from scripts.longbench_a800_report import report
        report(args.root, args.command == 'status_same_count')


if __name__ == '__main__':
    main()
