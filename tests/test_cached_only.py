"""CPU coverage for resuming cached answers while feature collection is pending."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import longbench_a800_control as control
from scripts import longbench_a800_data as data
from test_collection_v3 import fixture, commit

ROOT = Path(__file__).resolve().parents[1]


class CachedOnlyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_cached_only_preserves_records_and_later_full_resume(self):
        _, rows = fixture(self.root, 'ruler', thinking=True)
        commit(self.root, rows[0], 'nocache', 0)
        commit(self.root, rows[0], 'prophetkv-1', 0)
        # An incomplete feature artifact must not be deleted or accepted.
        partial = self.root/'features/records/probe/partial/data.tmp'
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b'preserve incomplete probe')
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob('*') if p.is_file()}
        phases = []
        with patch.object(control, 'execute_phase', side_effect=lambda b, r, p, s: phases.append(p)), \
                patch('scripts.longbench_a800_report.report') as report, \
                patch('scripts.router_dataset.export_collection') as export:
            control.supervise(self.root, 'ruler', cached_only=True)
            self.assertEqual(phases, ['cached'])
            report.assert_called_once_with(self.root, final=False, emit=False)
            export.assert_not_called()
            for name in ('ruler/complete.json', 'ruler/controls-complete.json', 'features/complete.json'):
                self.assertFalse((self.root/name).exists(), name)
            receipt = data.read(self.root/'ruler/cached-complete.json')
            self.assertEqual(receipt['answers'], 6500 * 8)
            self.assertNotIn('nocache', receipt['scheduled_actions'])
            state = data.read(self.root/'ruler/supervisor.json')
            self.assertEqual(state['state'], 'cached-complete')
            self.assertEqual(state['selected_phases'], ['cached'])
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
            # Restoring the normal schedule can collect the deferred features.
            phases.clear()
            control.supervise(self.root, 'ruler')
            self.assertEqual(phases, ['baseline', 'features', 'cached'])
            export.assert_called_once()
            self.assertTrue((self.root/'ruler/complete.json').exists())

    def test_failure_does_not_mark_cached_answers_complete(self):
        fixture(self.root)
        with patch.object(control, 'execute_phase', side_effect=RuntimeError('prime failed')), \
                patch('scripts.longbench_a800_report.report') as report:
            with self.assertRaisesRegex(RuntimeError, 'prime failed'):
                control.supervise(self.root, 'primary', cached_only=True)
        report.assert_not_called()
        self.assertFalse((self.root/'primary/cached-complete.json').exists())
        self.assertEqual(data.read(self.root/'primary/supervisor.json')['state'], 'failed')

    def test_cli_passes_cached_only_to_detach_and_supervisor(self):
        fixture(self.root)
        for command, function in (('resume', 'detach'), ('supervise', 'supervise')):
            argv = ['control', command, '--root', str(self.root), '--role', 'primary', '--cached-only']
            with patch.object(sys, 'argv', argv), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), \
                    patch.object(control, function) as called:
                control.main()
            self.assertTrue(called.call_args.kwargs['cached_only'])
            if command == 'resume':
                self.assertTrue(called.call_args.args[2])

    def test_detached_supervisor_command_retains_phase_selection(self):
        fixture(self.root)
        for selected in (False, True):
            def inspect_helper(*args, **kwargs):
                payload = json.loads(kwargs['input'])
                self.assertEqual('--cached-only' in payload['command'], selected)
                self.assertIn('supervise', payload['command'])
                self.assertEqual(payload['env']['CUDA_VISIBLE_DEVICES'], '')
                raise RuntimeError('inspected without spawning')
            with patch.object(control, 'idle'), patch.object(control.subprocess, 'run', side_effect=inspect_helper):
                with self.assertRaisesRegex(RuntimeError, 'without spawning'):
                    control.detach(self.root, 'primary', resume=True, cached_only=selected)

    def test_l40_launcher_reads_phase_selection_only_for_launch_commands(self):
        fake = self.root/'fake-python'
        fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        fake.chmod(0o755)
        env_file = self.root/'server.env'
        for selected in ('0', '1'):
            env_file.write_text(f'PYTHON_BIN="{fake}"\nGPU_DEVICES=GPU-fixture\nPROPHETKV_ONLY={selected}\n')
            for command in ('resume', 'detach', 'stop', 'status', 'configure', 'prepare', 'compute-feature'):
                result = subprocess.run(['bash', str(ROOT/'scripts/launcher/l40_ruler_thinking_data.sh'), command],
                    cwd='/tmp', env={'PATH': os.environ['PATH'], 'UCM_ENV_FILE': str(env_file)},
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                argv = json.loads(result.stdout)
                self.assertEqual('--cached-only' in argv, selected == '1' and command in ('resume', 'detach'))
                self.assertEqual(argv[1], command)


if __name__ == '__main__':
    unittest.main()
