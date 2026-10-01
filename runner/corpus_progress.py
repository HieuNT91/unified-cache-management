"""Flushed stderr progress for offline commands; silent outside a CLI session."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import sys
import threading
import time

_ACTIVE = ContextVar('corpus_progress', default=None)


@dataclass
class Stage:
    label: str
    total: object = None
    unit: str = 'items'
    completed: int = 0
    detail: str = ''
    last_printed: int = 0


class Reporter:
    def __init__(self, label, detail='', interval=5):
        self.started = time.monotonic()
        self.interval = interval
        self.stream = sys.stderr
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.current = Stage(label, detail=detail)
        self.thread = threading.Thread(target=self._heartbeat, name='corpus-progress', daemon=True)

    def emit(self, status):
        with self.lock:
            stage = self.current
            seconds = int(time.monotonic()-self.started)
            elapsed = f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'
            count = '' if stage.total is None else f' {stage.completed}/{stage.total} {stage.unit};'
            detail = f'; {stage.detail}' if stage.detail else ''
            try:
                print(f'[{elapsed}] {stage.label}:{count} {status}{detail}', file=self.stream, flush=True)
            except (OSError, ValueError):
                # A closed progress stream must not interrupt fitting or change results.
                pass
            stage.last_printed = stage.completed

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                status = 'progress' if self.current.completed != self.current.last_printed else 'still processing; count unchanged'
                self.emit(status)


@contextmanager
def session(label, detail='', output=None, interval=5):
    reporter = Reporter(label, detail, interval)
    token = _ACTIVE.set(reporter)
    reporter.emit('started')
    reporter.thread.start()
    try:
        yield reporter
    except BaseException:
        reporter.emit('stopped before completion')
        raise
    else:
        with reporter.lock:
            reporter.current = Stage('Complete', detail=f'reports: {output}' if output else '')
            reporter.emit('finished')
    finally:
        reporter.stop.set()
        reporter.thread.join()
        _ACTIVE.reset(token)


@contextmanager
def stage(label, total=None, unit='items'):
    reporter = _ACTIVE.get()
    if reporter is None:
        yield None
        return
    with reporter.lock:
        previous = reporter.current
        state = Stage(label, total, unit)
        reporter.current = state
        reporter.emit('started')
    try:
        yield state
    except BaseException:
        reporter.emit('interrupted')
        raise
    else:
        reporter.emit('finished')
    finally:
        with reporter.lock:
            reporter.current = previous


def detail(text):
    reporter = _ACTIVE.get()
    if reporter is not None:
        with reporter.lock:
            reporter.current.detail = str(text)


def advance():
    reporter = _ACTIVE.get()
    if reporter is not None:
        with reporter.lock:
            reporter.current.completed += 1


def track(items, label, unit='items', describe=str):
    if _ACTIVE.get() is None:
        yield from items
        return
    total = len(items) if hasattr(items, '__len__') else None
    with stage(label, total, unit):
        for item in items:
            detail(describe(item))
            yield item
            advance()
