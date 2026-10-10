"""CPU-only recovery: preserve frozen inputs/results and permit only the hotfix."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import cache_staging_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'original'
        self.dest = self.root/'recovered'
        self.run = self.root/'run'
        files = {
            'run.py': '', 'run.sh': '',
            'runner/sweep.py': 'class PromptCache:\n'+recovery.OLD_READY,
            'runner/corpus_runtime.py': 'def initialize():\n    if True:\n        if True:\n'+recovery.OLD_INIT+'\n',
            'scripts/longbench_a800_data.py': 'def stage(*args, **kwargs):\n    pass\ndef load():\n    if True:\n'+recovery.OLD_GUARD+'\n',
            'scripts/corpus_control.py': 'def idle(root):\n    if (root/"busy").exists():\n        raise ValueError("Owned worker remains alive")\n',
        }
        for name, content in files.items():
            path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        for name in ('ruler', 'features'):
            (self.run/name).mkdir(parents=True)
        self.settings = dict(dataset='ruler', execution_profile='ruler-thinking',
            hardware_profile='l40-tp2', tp=2, code=recovery.source_hashes(self.source),
            prepared=str(self.root/'prepared'), model=str(self.root/'model'),
            cache_root=str(self.root/'cache'))
        (self.run/'settings.json').write_text(json.dumps(self.settings))
        (self.run/'plan.json').write_text('{}')
        self.accepted = self.run/'ruler/records/existing/validated.json'
        self.accepted.parent.mkdir(parents=True)
        self.accepted.write_text('{"complete": true}')
        self.failure = self.run/'ruler/sessions/cached-group3-attempt/failure.json'
        self.failure.parent.mkdir(parents=True)
        key = 'a'*32
        self.evidence = dict(phase='cached', prompt_id='niah_single_1-051',
            error='Unexpected files in temporary prompt cache: example',
            inventory=dict(extra=[f'kv/.temp/{key}'], missing=[], inventory_errors=[],
                expected_files=[f'kv/{key[:8]}/{key}'],
                actual_files={f'kv/.temp/{key}': dict(bytes=4202496, symlink=False)}))
        self.failure.write_text(json.dumps(self.evidence))

    def create(self):
        return subprocess.run([sys.executable, recovery.__file__,
            '--source-checkout', str(self.source), '--destination', str(self.dest),
            '--root', str(self.run), '--failure', str(self.failure)],
            capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))

    def validate(self):
        return recovery.validate_recovery(self.run, self.settings,
            recovery.source_hashes(self.dest), self.dest)

    def test_create_preserves_frozen_files_and_records_exact_changes(self):
        protected = [self.run/'settings.json', self.run/'plan.json', self.accepted, self.failure]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected}
        result = self.create()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No process launched', result.stdout)
        self.assertEqual(recovery.source_hashes(self.source), self.settings['code'])
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in protected})
        receipt = self.validate()
        self.assertEqual(set(receipt['patches']), set(recovery.PATCHES))
        self.assertEqual(receipt['failure_sha256'], recovery.digest(self.failure.read_bytes()))
        for name in recovery.PATCHES:
            ast.parse((self.dest/name).read_text())
        repeat = self.create()
        self.assertNotEqual(repeat.returncode, 0)

    def test_modified_runtime_or_receipt_is_rejected(self):
        self.assertEqual(self.create().returncode, 0)
        for name in ('run.py', 'runner/sweep.py', recovery.SELF):
            with self.subTest(name=name):
                path = self.dest/name
                original = path.read_bytes()
                path.write_bytes(original+b'\n# unapproved edit\n')
                with self.assertRaises(ValueError):
                    self.validate()
                path.write_bytes(original)
        record = self.run/'runtime-recoveries'/self.validate()['id']/'receipt.json'
        data = json.loads(record.read_text());data['root'] = '/another/run'
        record.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'receipt/helper'):
            self.validate()

    def test_new_initializations_pin_recovery_and_frozen_settings_stay_required(self):
        self.assertEqual(self.create().returncode, 0)
        control = SimpleNamespace(code_hashes=lambda: recovery.source_hashes(self.dest))
        with patch.object(recovery, '__file__', str(self.dest/recovery.SELF)), \
                patch.dict(sys.modules, {'scripts.router_control': control}):
            pin = recovery.initialization_receipt(self.run)
            self.assertEqual(pin['sha256'], recovery.digest(Path(pin['receipt']).read_bytes()))
            self.assertEqual(pin['patches'], self.validate()['patches'])
            (self.run/'plan.json').write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                recovery.initialization_receipt(self.run)

    def test_busy_or_changed_source_fails_before_creating_copy(self):
        (self.run/'ruler/busy').touch()
        result = self.create()
        self.assertIn('Owned worker', result.stderr)
        self.assertFalse(self.dest.exists())
        (self.run/'ruler/busy').unlink()
        (self.source/'run.py').write_text('# changed')
        result = self.create()
        self.assertIn('differs from frozen source', result.stderr)
        self.assertFalse(self.dest.exists())

    def test_unknown_staging_and_missing_shards_are_not_recoverable(self):
        for change in ('missing', 'unknown', 'symlink'):
            evidence = json.loads(json.dumps(self.evidence))
            inv = evidence['inventory']
            if change == 'missing':
                inv['missing'] = ['kv/missing']
            elif change == 'unknown':
                inv['expected_files'] = []
            else:
                inv['actual_files'][inv['extra'][0]]['symlink'] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery.validate_failure(json.dumps(evidence).encode())

    def test_patch_matches_current_readiness_implementation(self):
        source = (Path(recovery.__file__).resolve().parents[1]/'runner/sweep.py').read_text()
        self.assertEqual(source.count(recovery.NEW_READY), 1)


if __name__ == '__main__':
    unittest.main()
