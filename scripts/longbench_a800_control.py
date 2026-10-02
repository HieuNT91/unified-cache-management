#!/usr/bin/env python3
"""Two independent TP4 control launchers, followed by attention collection."""
import argparse
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
from scripts.corpus_control import groups, idle, check_hardware
from scripts.longbench_a800_data import read, freeze, load, stage, SCHEDULE, ACTIONS


def configure(args):
    selected = groups(args.gpu_a, args.gpu_b)
    if len(selected) != 2:
        raise ValueError('Both GPU_A (physical 0-3) and GPU_B (physical 4-7) UUID groups are required')
    settings = dict(schema='longbench-a800-config-v1', model=str(args.model), data=str(args.data),
                    prepared=str(args.prepared), cache_root=str(args.cache_root), groups=selected,
                    code=code_hashes(), watchdog_seconds=7200, automatic_training=False)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root/'configure.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (args.root/'settings.json').exists() and any(p.name != 'configure.lock' for p in args.root.iterdir()):
            raise ValueError('Use a new empty experiment directory')
        freeze(args.root/'settings.json', settings)
    print('Configured 503 prompts, 4527 fixed answers, then 503 attention probes; training is a separate command.')


def progress(base, role, message):
    atomic_json(Path(base)/role/'progress.json', dict(at=time.time(), message=message))
    print(message, flush=True)


def pending(protocol, rows, root, cases):
    return [r for r in rows if any(accepted(root, c, r, protocol) is None for c in cases)]


def worker(base, role, phase, attempt):
    from runner.corpus_runtime import Engine
    from runner.setups import check_environment
    target = 'features' if phase == 'features' else role
    protocol, rows = stage(base, target, check_code=True)
    root = Path(base)/target
    cases = (['probe'] if phase == 'features' else ['nocache'] if phase == 'baseline'
             else [c for c in protocol['scheduled_actions'] if c != 'nocache'])
    remaining = pending(protocol, rows, root, cases)
    if not remaining:
        return
    if check_environment(4) != protocol['groups'][0]:
        raise ValueError('Worker GPU UUIDs differ from the frozen group')
    with (root/'group0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(root, protocol, 0, 'baseline' if phase == 'baseline' else 'cached', attempt, remaining)
        definitions = {a['id']: a for a in ACTIONS}
        try:
            for row in remaining:
                progress(base, role, f"{phase}: starting {row['id']}")
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
                    progress(base, role, f"{case}: accepted {row['id']}")
                deletion = engine.end()
                atomic_json(root/'deletions'/f'{phase}-{row["id"]}.json', deletion)
        finally:
            engine.close()


def hardware(protocol):
    receipt = check_hardware(protocol)
    if any('A800' not in device['name'] for group in receipt['groups'] for device in group):
        raise ValueError('This deployment requires A800 GPUs; no local GPU fallback')
    return receipt


def engines_idle(root):
    for path in [*(root/'processes').glob('*.json'), *(root/'sessions').glob('*/ownership.json')]:
        saved = read(path)
        if alive(saved) or group_alive(saved['pid']):
            raise RuntimeError('An owned worker/engine remains alive')
    for path in root.glob('group*.lock'):
        with path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def execute_phase(base, role, phase, state):
    settings, _, _ = load(base, check_code=True)
    target = 'features' if phase == 'features' else role
    protocol, rows = stage(base, target)
    cases = ['probe'] if phase == 'features' else ['nocache'] if phase == 'baseline' else [c for c in SCHEDULE[role] if c != 'nocache']
    if not pending(protocol, rows, base/target, cases):
        return
    engines_idle(base/target)
    engines_idle(base/role)
    atomic_json(base/target/'hardware.json', hardware(protocol))
    cleanup_caches(base/target, protocol)
    attempt = uuid.uuid4().hex
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--root', str(base),
               '--role', role, '--phase', phase, '--attempt', attempt]
    with (base/role/f'{phase}-{attempt}.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment(','.join(protocol['groups'][0])),
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=child.pid, identity=identity(child.pid), phase=phase, command=command)
    atomic_json(base/role/'processes'/f'{attempt}.json', receipt)
    state.update(phase=phase, worker=receipt); atomic_json(base/role/'supervisor.json', state)
    last = time.monotonic(); changed = 0
    try:
        while child.poll() is None:
            path = base/role/'progress.json'
            if path.exists() and path.stat().st_mtime > changed:
                changed = path.stat().st_mtime; last = time.monotonic()
            if time.monotonic()-last > settings['watchdog_seconds']:
                raise RuntimeError('Progress watchdog expired; inspect logs before resume')
            time.sleep(2)
    except BaseException:
        terminate_owned(receipt); child.wait(timeout=10); raise
    if group_alive(child.pid):
        terminate_owned(receipt)
    atomic_json(base/role/'exits'/f'{attempt}.json', dict(receipt, exit_code=child.returncode,
                                                       owned_engines_exited=not group_alive(child.pid)))
    if child.returncode:
        raise RuntimeError(f'{phase} worker failed ({child.returncode}); accepted results retained, use resume')
    cleanup_caches(base/target, protocol)
    if pending(protocol, rows, base/target, cases):
        raise RuntimeError('Worker exited without all scheduled records')


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
            for phase in (('baseline', 'cached') if role == 'primary' else ('cached',)):
                execute_phase(base, role, phase, state)
            protocol, rows = stage(base, role)
            freeze(state_root/'controls-complete.json', dict(answers=len(rows)*len(SCHEDULE[role]),
                   protocol_sha256=file_hash(state_root/'protocol.json'), owned_engines_exited=True))
            if role == 'primary':
                state.update(state='waiting-for-extra', phase='waiting'); atomic_json(state_root/'supervisor.json', state)
                wait_extra(base)
                from scripts.longbench_a800_report import report
                report(base, final=True, emit=False)
                state.update(state='collecting-features'); atomic_json(state_root/'supervisor.json', state)
                execute_phase(base, role, 'features', state)
                freeze(base/'features/complete.json', dict(probes=len(rows), owned_engines_exited=True,
                       protocol_sha256=file_hash(base/'features/protocol.json')))
                from scripts.router_dataset import export_longbench
                export_longbench(base, base/'router-data.json')
            freeze(state_root/'complete.json', dict(complete=True, owned_engines_exited=True,
                   plan_sha256=file_hash(base/'plan.json'), feature_collection=role == 'primary'))
            state.update(state='complete', finished_at=time.time())
        except BaseException as error:
            state.update(state='failed', error=str(error), finished_at=time.time()); raise
        finally:
            atomic_json(state_root/'supervisor.json', state)


def detach(base, role, resume=False):
    load(base, check_code=True)
    target = base/role
    with (target/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle(target)
        if role == 'primary':
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
    parser.add_argument('command', choices=('configure', 'prepare', 'detach', 'resume', 'stop', 'status', 'status_same_count', 'report', 'supervise', 'worker'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--role', choices=('primary', 'extra'), required=True)
    for name in ('model', 'data', 'prepared', 'cache-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--gpu-a'); parser.add_argument('--gpu-b')
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
    elif args.command == 'stop':
        from scripts.launcher_stop import stop
        stop(args.root, Path(__file__).name, args.root/args.role)
    elif args.command == 'supervise':
        supervise(args.root, args.role)
    elif args.command == 'worker':
        worker(args.root, args.role, args.phase, args.attempt)
    else:
        from scripts.longbench_a800_report import report
        report(args.root, args.command == 'status_same_count')


if __name__ == '__main__':
    main()
