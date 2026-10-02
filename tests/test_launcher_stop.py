"""Stop real, isolated CPU sessions; never use experiment/GPU processes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import launcher_stop as control


class LauncherStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write(self.root/'settings.json', {})

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def launch(self, controller='corpus_control.py', role='supervise', code=None, root=None):
        command = [sys.executable, '-c', code or 'import time; time.sleep(120)',
                   controller, role, '--root', str(root or self.root)]
        child = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        def cleanup():
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
        self.addCleanup(cleanup)
        return child, dict(pid=child.pid, identity=control.process_table()[child.pid]['identity'])

    def ready(self, path):
        deadline = time.monotonic()+5
        while not path.exists():
            if time.monotonic() >= deadline:
                self.fail('CPU fixture startup timed out')
            time.sleep(.02)

    def test_stop_supervisor_worker_and_preserve_data_and_unrelated_process(self):
        supervisor, saved = self.launch()
        worker, worker_saved = self.launch('runner.corpus_collect', 'worker')
        unrelated, _ = self.launch(root=self.root/'other')
        self.write(self.root/'supervisor.json', dict(saved, state='running'))
        self.write(self.root/'processes/attempt.json', worker_saved)
        for name in ('records/answer.json', 'cache/shard', 'protocol.json'):
            self.write(self.root/name, {'preserve': name})
        original = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = control.stop(self.root, 'corpus_control.py')
        supervisor.wait(timeout=5)
        worker.wait(timeout=5)
        self.assertIsNone(unrelated.poll())
        self.assertTrue(result['owned_engines_exited'])
        for path, contents in original.items():
            self.assertEqual(path.read_bytes(), contents)
        # A second stop is harmless, including stale receipts for exited PIDs.
        control.stop(self.root, 'corpus_control.py')

    def test_extension_stop_leaves_base_controller_and_receipts_untouched(self):
        base, base_saved = self.launch()
        self.write(self.root/'supervisor.json', base_saved)
        child, saved = self.launch('corpus_add_ratios.py')
        state = self.root/'extensions/add5-10'
        self.write(state/'supervisor.json', saved)
        control.stop(self.root, 'corpus_add_ratios.py', state)
        child.wait(timeout=5)
        self.assertIsNone(base.poll())
        self.assertEqual(json.loads((self.root/'supervisor.json').read_text()), base_saved)
        self.assertTrue((state/'stop-receipt.json').exists())
        self.assertFalse((self.root/'stop-receipt.json').exists())

    def test_router_orphan_engine_is_stopped_from_ownership_receipt(self):
        ready = self.root/'ready'
        code = ('import subprocess,sys,time\nfrom pathlib import Path\n'
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"])\n'
                f'Path({str(ready)!r}).write_text(str(p.pid))\n'
                'time.sleep(120)\n')
        leader, saved = self.launch('runner.router_sweep', 'worker', code)
        self.ready(ready)
        orphan = int(ready.read_text())
        self.write(self.root/'sessions/attempt/ownership.json', saved)
        leader.terminate()
        leader.wait(timeout=5)
        self.assertIn(orphan, control.process_table())
        control.stop(self.root, 'router_control.py')
        self.assertNotIn(orphan, control.process_table())

    def test_unpublished_worker_child_is_discovered_before_supervisor_exit(self):
        ready = self.root/'ready'
        code = ('import subprocess,sys,time\nfrom pathlib import Path\n'
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"],start_new_session=True)\n'
                f'Path({str(ready)!r}).write_text(str(p.pid))\n'
                'time.sleep(120)\n')
        child, saved = self.launch(code=code)
        self.write(self.root/'supervisor.json', saved)
        self.ready(ready)
        worker = int(ready.read_text())
        def cleanup():
            try:
                os.killpg(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.addCleanup(cleanup)
        control.stop(self.root, 'corpus_control.py')
        child.wait(timeout=5)
        self.assertNotIn(worker, control.process_table())

    def test_term_resistant_worker_escalates_to_kill(self):
        ready = self.root/'ready'
        code = ('import signal,time\nfrom pathlib import Path\n'
                'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                f'Path({str(ready)!r}).touch()\ntime.sleep(120)')
        child, saved = self.launch('runner.corpus_collect', 'worker', code)
        self.write(self.root/'processes/attempt.json', saved)
        self.ready(ready)
        wait = control.wait_exit
        with patch.object(control, 'wait_exit', side_effect=lambda rows, seconds: wait(rows, min(seconds, .3))):
            control.stop(self.root, 'corpus_control.py')
        self.assertEqual(child.wait(timeout=5), -signal.SIGKILL)

    def test_changed_pid_identity_is_rejected_without_signalling(self):
        child, saved = self.launch()
        saved['identity']['start'] = str(int(saved['identity']['start'])-1)
        self.write(self.root/'supervisor.json', saved)
        with patch.object(control.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                control.stop(self.root, 'corpus_control.py')
            kill.assert_not_called()
        self.assertIsNone(child.poll())

    def test_foreign_experiment_receipt_is_rejected(self):
        child, saved = self.launch(root=self.root/'other')
        self.write(self.root/'supervisor.json', saved)
        with patch.object(control.os, 'killpg') as kill:
            with self.assertRaisesRegex(ValueError, 'another experiment'):
                control.stop(self.root, 'corpus_control.py')
            kill.assert_not_called()
        self.assertIsNone(child.poll())

    def test_launch_lock_blocks_stop(self):
        with (self.root/'launch.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                control.stop(self.root, 'corpus_control.py')
        self.assertFalse((self.root/'stop-receipt.json').exists())

    def test_configured_idle_stop_and_missing_root(self):
        control.stop(self.root, 'router_control.py')
        missing = self.root/'missing'
        with self.assertRaises(ValueError):
            control.stop(missing, 'router_control.py')
        self.assertFalse(missing.exists())


if __name__ == '__main__':
    unittest.main()
