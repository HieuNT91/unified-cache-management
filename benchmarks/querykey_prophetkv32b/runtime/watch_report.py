"""Detached CPU finalizer; waits for successful inference and owned-engine exit."""
import fcntl
import os
from pathlib import Path
import time

from common import settings, load, dump, identity
from suite import live_supervisor
from finalize import main as finalize


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('The report watcher must expose no GPUs')
    root = Path(settings()['root'])
    with (root / 'reporting.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), identity=identity(os.getpid()),
            pgid=os.getpgrp(), sid=os.getsid(0), state='waiting', started_at=time.time())
        dump(root / 'reporter.json', state)
        try:
            while live_supervisor(root):
                time.sleep(5)
            progress = load(root / 'progress.json')
            if progress['state'] != 'complete' or progress['validated'] != 480:
                raise RuntimeError(f'Inference did not finish: {progress}')
            finalize()
            dump(root / 'reporter.json', state | dict(state='complete', ended_at=time.time()))
        except BaseException as error:
            dump(root / 'reporter.json', state | dict(state='failed', error=str(error), ended_at=time.time()))
            raise


if __name__ == '__main__':
    main()
