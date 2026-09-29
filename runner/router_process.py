"""Linux process identity and owned process-group helpers; no historical job control."""
import os
from pathlib import Path


def identity(pid):
    try:
        root = Path('/proc')/str(pid)
        stat = (root/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(start=stat[19],cmd=(root/'cmdline').read_bytes().hex())
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
