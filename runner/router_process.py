"""Linux process identity and owned process-group helpers; no historical job control."""
import os
import time
from pathlib import Path


def identity(pid):
    try:
        root = Path('/proc')/str(pid)
        start = None
        for attempt in range(100):
            stat = (root/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0] == 'Z' or (start is not None and stat[19] != start):
                return None
            start = stat[19]
            command = (root/'cmdline').read_bytes()
            if command:
                return dict(start=start,cmd=command.hex())
            # Newly launched processes may briefly expose an empty cmdline.
            # Never persist a start-only identity as if argv had been observed.
            if attempt < 99:
                time.sleep(.01)
        return None
    except (OSError,IndexError):
        return None


def alive(state):
    return state.get('identity') is not None and identity(state['pid']) == state['identity']


def group_alive(pgid):
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            stat=path.read_text().rsplit(')',1)[1].split()
            if stat[0] != 'Z' and int(stat[2]) == pgid:
                return True
        except (OSError,ValueError,IndexError):
            pass
    return False
