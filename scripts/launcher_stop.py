"""CPU-only stop for receipt-owned router/corpus sessions (Linux, stdlib only)."""
import fcntl
import json
import os
from pathlib import Path
import signal
import time


def process_table():
    table = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            command = (path.parent/'cmdline').read_bytes()
            # Exit can clear cmdline after the stat read but before state turns Z.
            # Kernel threads also have no userspace command/owned launch identity.
            if not command:
                continue
            table[int(path.parent.name)] = dict(
                state=fields[0], parent=int(fields[1]), group=int(fields[2]),
                session=int(fields[3]), identity=dict(
                    start=fields[19], cmd=command.hex()))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return table


def members(receipt, table):
    pid, saved = receipt['pid'], receipt['identity']
    group = {p: row for p, row in table.items() if row['group'] == pid}
    if not group and pid not in table:
        return {}
    if not saved or (pid in table and table[pid]['identity'] != saved):
        raise RuntimeError(f'Process identity changed for PID {pid}; refusing stop')
    if pid in table and table[pid]['group'] != pid:
        raise RuntimeError(f'PID {pid} is not an owned session leader')
    if any(row['session'] != pid or int(row['identity']['start']) < int(saved['start'])
           for row in group.values()):
        raise RuntimeError(f'Process-group ownership changed for PID {pid}')
    if os.getpid() in group or os.getppid() in group:
        raise RuntimeError('Refusing to stop the calling session')
    return group


def send_group(receipt, sig):
    if members(receipt, process_table()):
        try:
            os.killpg(receipt['pid'], sig)
        except ProcessLookupError:
            pass


def receipts(state, root, controller):
    paths = [state/name for name in ('supervisor.json', 'launch.json', 'detachment.json')]
    paths += sorted((state/'processes').glob('*.json'))
    paths += sorted((state/'sessions').glob('*/ownership.json'))
    found = {}
    worker = {'router_control.py': 'runner.router_sweep',
              'corpus_control.py': 'runner.corpus_collect'}.get(controller)
    for path in paths:
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        pid, ident = receipt['pid'], receipt.get('identity')
        if not isinstance(pid, int) or pid <= 1:
            raise ValueError(f'Invalid process receipt: {path}')
        if ident is None:
            if members(dict(pid=pid, identity=None), process_table()):
                raise RuntimeError(f'Missing process identity: {path}')
            continue
        command = os.fsdecode(bytes.fromhex(ident['cmd'])).split('\0')[:-1]
        # Bind saved identity to the requested controller and experiment, without
        # loading a model, prepared data, policy or changed source hashes.
        try:
            target = Path(command[command.index('--root')+1])
        except (ValueError, IndexError):
            raise ValueError(f'Missing experiment identity: {path}') from None
        supervisor = any(Path(arg).name == controller for arg in command) and 'supervise' in command
        is_worker = (worker in command if worker else
                     any(Path(arg).name == controller for arg in command) and 'worker' in command)
        if not target.is_absolute() or target.resolve() != root or not (supervisor or is_worker):
            raise ValueError(f'Process receipt belongs to another experiment: {path}')
        found[(pid, ident['start'])] = dict(pid=pid, identity=ident, supervisor=supervisor)
    return list(found.values())


def wait_exit(owned, seconds):
    deadline = time.monotonic()+seconds
    while True:
        table = process_table()
        if not any(members(row, table) for row in owned):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(.1)


def stop(root, controller, state_dir=None):
    root = Path(root).resolve()
    state = Path(state_dir).resolve() if state_dir is not None else root
    if not state.is_dir() or not any((root/name).is_file() for name in ('settings.json', 'protocol.json')):
        raise ValueError('Not an existing configured experiment/state directory')
    with (state/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owned = receipts(state, root, controller)
        table = process_table()
        for row in owned:
            members(row, table)  # Validate everything before sending any signal.
        supervisors = [row for row in owned if row['supervisor'] and row['pid'] in table]
        frozen = []
        try:
            # Freeze scheduling before discovering children, including a worker
            # spawned immediately before its processes/ receipt was published.
            for row in supervisors:
                send_group(row, signal.SIGSTOP)
                frozen.append(row)
            deadline = time.monotonic()+5
            while frozen:
                table = process_table()
                if all(all(p['state'] in ('T', 't') for p in members(row, table).values()) for row in frozen):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Supervisor did not stop scheduling')
                time.sleep(.05)
            children = {row['pid'] for row in supervisors}
            while True:
                more = {pid for pid, row in table.items() if row['parent'] in children}
                if more <= children:
                    break
                children |= more
            for pid in children:
                pgid = table[pid]['group'] if pid in table else None
                if pgid is not None and not any(row['pid'] == pgid for row in owned):
                    if pgid not in table:
                        raise RuntimeError('Child session has no ownership receipt')
                    row = dict(pid=pgid, identity=table[pgid]['identity'], supervisor=False)
                    members(row, table)
                    owned.append(row)
            for row in supervisors:
                send_group(row, signal.SIGTERM)
                send_group(row, signal.SIGCONT)
            if not wait_exit(supervisors, 5):
                for row in supervisors:
                    send_group(row, signal.SIGKILL)
                if not wait_exit(supervisors, 5):
                    raise RuntimeError('Supervisor remains alive; stop incomplete')
        finally:
            for row in frozen:
                send_group(row, signal.SIGCONT)
        print(f'Stopping owned sessions for {state}', flush=True)
        for row in owned:
            send_group(row, signal.SIGTERM)
        if not wait_exit(owned, 30):
            for row in owned:
                send_group(row, signal.SIGKILL)
            if not wait_exit(owned, 5):
                raise RuntimeError('Owned engines remain alive; stop incomplete')
        for row in receipts(state, root, controller):
            if members(row, process_table()):
                raise RuntimeError('New owned process remains alive; stop incomplete')
        # Detect unrecorded workers/other workflows without signalling them.
        for path in [state/'run.lock', *state.glob('group*.lock')]:
            if path.exists():
                with path.open('a') as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = dict(root=str(root), state_dir=str(state), stopped_at=time.time(),
                       owned_engines_exited=True, processes=owned)
        temporary = state/'stop-receipt.json.tmp'
        temporary.write_text(json.dumps(receipt, indent=2)+'\n')
        temporary.replace(state/'stop-receipt.json')
        print('Owned processes exited. Existing results and caches were preserved.', flush=True)
        return receipt
