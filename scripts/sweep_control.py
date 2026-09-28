#!/usr/bin/env python3
"""Stop only processes belonging to one temporary sweep (standard library only)."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def process_table():
    result = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                continue
            command = os.fsdecode((directory/'cmdline').read_bytes()).split('\0')[:-1]
            result[int(directory.name)] = dict(start=fields[19], parent=int(fields[1]), command=command)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def owned_processes(output, table=None):
    output = Path(output).resolve()
    table = process_table() if table is None else table
    owned = set()
    for pid, row in table.items():
        if pid == os.getpid():
            continue
        command = row['command']
        if 'runner.sweep' in command and '--output' in command:
            value = command[command.index('--output')+1]
            # Launchers always supply an absolute path; direct callers may not.
            base = Path(f'/proc/{pid}/cwd')
            if (base/value).resolve() == output:
                owned.add(pid)
        try:
            env = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
            for value in env:
                if value.startswith(b'PROPHETKV_SCHEDULER_RECEIPT='):
                    receipt = Path(os.fsdecode(value.split(b'=', 1)[1])).resolve()
                    if receipt.name == 'scheduler.json' and output in receipt.parents:
                        owned.add(pid)  # includes reparented engine workers
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
        # The launcher's flock is inherited by its workers. It also identifies
        # the coordinator before it has started any worker.
        try:
            for fd in Path(f'/proc/{pid}/fd').iterdir():
                try:
                    if os.readlink(fd) == str(output/'sweep.lock'):
                        owned.add(pid)
                        break
                except (FileNotFoundError, PermissionError, OSError):
                    pass
        except (FileNotFoundError, PermissionError):
            pass
    while True:
        children = {pid for pid, row in table.items() if row['parent'] in owned}
        if children <= owned:
            break
        owned |= children
    owned.discard(os.getpid())
    return {pid: table[pid] for pid in owned}


def alive(row, pid):
    current = process_table().get(int(pid))
    return current is not None and current['start'] == row['start']


def assert_stopped(output):
    receipt = Path(output)/'stop-receipt.json'
    if not receipt.exists():
        raise RuntimeError('Run scripts/sweep_control.py stop --output YOUR_RUN first')
    previous = json.loads(receipt.read_text())
    if previous['output'] != str(Path(output).resolve()):
        raise RuntimeError('Stop receipt belongs to another directory')
    table = process_table()
    live = [pid for pid, row in previous['processes'].items()
            if int(pid) in table and table[int(pid)]['start'] == row['start']]
    # The resume coordinator itself owns sweep.lock. Its worker children have
    # not started yet. Any runner.sweep process here is an overlapping run.
    coordinator = (os.getppid() if os.environ.get('UCM_SWEEP_COORDINATOR_PID') == str(os.getppid()) else None)
    live += [pid for pid in owned_processes(output, table) if pid != coordinator]
    if live:
        raise RuntimeError(f'Prior sweep processes still alive: {live}')


def stop(output):
    output = Path(output).resolve()
    if not (output/'group-0/plan.json').is_file():
        raise RuntimeError('Not an existing sweep result directory')
    owned = owned_processes(output)
    previous = output/'stop-receipt.json'
    if previous.exists():
        old = json.loads(previous.read_text())['processes']
        for pid, row in old.items():
            if alive(row, pid):
                owned[int(pid)] = row
    print(f'Stopping {len(owned)} owned processes for {output}', flush=True)
    def send(pid, row, sig):
        if alive(row, pid):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
    # Stop coordinators first so they cannot start a second phase or reporter.
    for pid, row in owned.items():
        if any(Path(c).name in ('l20_ruler.sh', 'a800_longbench.sh') for c in row['command']):
            send(pid, row, signal.SIGTERM)
    for pid, row in owned.items():
        if 'runner.sweep' in row['command']:
            send(pid, row, signal.SIGINT)
    for sig, seconds in ((None, 30), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        for pid, row in owned.items():
            if sig is not None:
                send(pid, row, sig)
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            table = process_table()
            if not any(pid in table and table[pid]['start'] == row['start'] for pid, row in owned.items()):
                break
            time.sleep(.5)
    remaining = owned_processes(output)
    if remaining or any(alive(row, pid) for pid, row in owned.items()):
        raise RuntimeError('Some owned processes remain; do not resume yet')
    receipt = dict(output=str(output), stopped_at=time.time(), processes=owned)
    temp = output/'stop-receipt.json.tmp'
    temp.write_text(json.dumps(receipt, indent=2)+'\n')
    temp.replace(previous)
    print('Owned processes exited. Existing results and caches were preserved.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['stop'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stop(args.output)
