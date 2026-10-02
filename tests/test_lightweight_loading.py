"""Launch/report metadata paths avoid full-cohort and diagnostic replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runner.setups import atomic_json, file_hash
from scripts.corpus_inputs import original_rows
from scripts.ruler import EVALUATION_PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


class LightweightLoadingTests(unittest.TestCase):
    def test_manifest_loading_never_opens_prompt_or_raw_batches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Payloads deliberately absent: this stage loads inventory, not token data.
            row = dict(id='niah_single_1-000', prepared='samples/a.json',
                       subtask='niah_single_1')
            (root/'manifest.jsonl').write_text(json.dumps(row)+'\n')
            atomic_json(root/'spec.json', dict(seed=42))
            atomic_json(root/'preparation.json', dict(spec={'protocol': EVALUATION_PROTOCOL}, files={
                'manifest.jsonl': file_hash(root/'manifest.jsonl'),
                'samples/a.json': 'b'*64, 'raw/niah_single_1/validation.jsonl': 'c'*64}))
            rows = original_rows(root)
            self.assertEqual(rows[0]['sha256'], 'b'*64)
            self.assertEqual(rows[0]['raw_sha256'], 'c'*64)
            self.assertEqual(rows[0]['ordinal'], 0)
            self.assertEqual(original_rows(root), rows)
            (root/'manifest.jsonl').write_text(json.dumps(dict(row, id='changed'))+'\n')
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                original_rows(root)

    def test_verify_command_removed_from_all_control_clis(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        for name in ('corpus_control.py', 'router_control.py', 'corpus_add_ratios.py'):
            with self.subTest(script=name):
                result = subprocess.run([sys.executable, str(ROOT/'scripts'/name),
                    'verify', '--root', '/unused'], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('invalid choice', result.stderr)

    def test_worker_keeps_input_and_model_binding_at_use_time(self):
        from runner.corpus_runtime import Engine
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_json(root/'config.json', {'model_type': 'qwen3'})
            atomic_json(root/'input.json', {'model_config_sha256': 'wrong'})
            row = dict(prepared='input.json', sha256=file_hash(root/'input.json'))
            engine = Engine.__new__(Engine)
            engine.protocol = dict(prepared=str(root), model=str(root), dataset='ruler')
            with patch.dict(sys.modules, {'runner.worker': SimpleNamespace(drain=object())}), \
                    patch('runner.corpus_runtime.validate'), patch('run.generate') as generate:
                with self.assertRaisesRegex(ValueError, 'Prepared model mismatch'):
                    engine.begin(row)
                row['sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'Input changed'):
                    engine.begin(row)
                generate.assert_not_called()

    def test_resume_defaults_to_fast_without_saved_replay(self):
        from runner import resume
        from test_resume import ResumeTests
        fixture = ResumeTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        del fixture.args.validation
        with patch('run.verify_diagnostics', side_effect=AssertionError('Unexpected replay')):
            receipt = resume.prepare(fixture.args)
        self.assertEqual(receipt['validation']['mode'], 'fast')
        self.assertFalse(receipt['validation']['diagnostic_replay'])
