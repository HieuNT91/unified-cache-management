"""Source updates must not invalidate accepted data or relax run integrity."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from runner.corpus_records import record_dir
from runner.setups import atomic_json, file_hash, fingerprint
from scripts import longbench_a800_control as control
from scripts import longbench_a800_data as data
from scripts import router_control, corpus_control
from test_collection_v3 import fixture, commit


class SourceUpdateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_l40_resume_keeps_baseline_and_schedules_missing_work(self):
        settings, rows = fixture(self.root, 'ruler', thinking=True)
        row = rows[0]; commit(self.root, row, 'nocache', 0)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.root.rglob('*') if p.is_file()}
        with patch.object(router_control, 'code_hashes', return_value={'run.py': 'updated'}):
            protocol, loaded = data.stage(self.root, 'ruler', check_code=True)
            self.assertEqual(loaded, rows)
            self.assertEqual(data.read(self.root/'settings.json')['code'], settings['code'])
            self.assertEqual(control.pending(protocol, [row], self.root/'ruler', ['nocache']), [])
            self.assertEqual(control.pending(protocol, [row], self.root/'ruler', ['prophetkv-1']), [row])
            features, _ = data.stage(self.root, 'features', check_code=True)
            self.assertEqual(control.pending(features, [row], self.root/'features', ['probe']), [row])
            # Exercise the real stage/accepted path; narrow only the worker shard.
            with patch.object(control, 'ordered_rows', return_value=[row]), \
                    patch('runner.corpus_runtime.Engine') as engine, \
                    patch('runner.setups.check_environment') as devices:
                control.worker(self.root, 'ruler', 'baseline', 'updated')
            engine.assert_not_called(); devices.assert_not_called()
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_changed_metadata_and_accepted_result_still_fail(self):
        _, rows = fixture(self.root)
        row = rows[0]; commit(self.root, row, 'nocache', 0)
        protocol, _ = data.stage(self.root, 'primary')
        for name in ('settings.json', 'rows.json', 'primary/protocol.json'):
            path = self.root/name; original = path.read_bytes()
            try:
                value = json.loads(original)
                if isinstance(value, list): value[0]['sha256'] = 'changed'
                else: value['model'] = '/changed'
                atomic_json(path, value)
                with self.assertRaises(ValueError): data.stage(self.root, 'primary', check_code=True)
            finally:
                path.write_bytes(original)
        result = record_dir(self.root/'primary', 'nocache', row['id'], 0)/'result.json'
        result.write_text('{}')
        with self.assertRaises(ValueError):
            control.pending(protocol, [row], self.root/'primary', ['nocache'])

    def test_reconfigure_retains_old_source_receipt_but_rejects_new_parameters(self):
        args = NS(root=self.root/'run', model=self.root/'model', data=self.root/'source',
                  prepared=self.root/'prepared', cache_root=self.root/'cache',
                  devices=','.join(f'GPU-00000000-0000-0000-0000-{j:012x}' for j in range(8)),
                  role='ruler', thinking=True, hardware_profile='l40-tp2', tp=2, seed=42,
                  samples_per_task=500, collection_version='v3', answers_per_question=1)
        with patch.object(control, 'code_hashes', return_value={'old.py': 'old'}):
            control.configure(args)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in args.root.rglob('*.json')}
        with patch.object(control, 'code_hashes', return_value={'new.py': 'new'}):
            control.configure(args)
            args.model = self.root/'another-model'
            with self.assertRaisesRegex(ValueError, 'Frozen'): control.configure(args)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_prepare_dispatch_does_not_gate_on_source(self):
        fixture(self.root)
        argv = ['control', 'prepare', '--root', str(self.root), '--role', 'primary']
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''), \
                patch.object(control, 'code_hashes', return_value={'updated.py': 'new'}), \
                patch.object(data, 'prepare') as prepare:
            control.main()
        prepare.assert_called_once_with(self.root)

    def test_router_source_updates_keep_cohort_and_policy_guards(self):
        prepared = self.root/'prepared'; prepared.mkdir()
        (prepared/'manifest.jsonl').write_text('[]')
        (prepared/'preparation.json').write_text('{}')
        policy_path = self.root/'policy.json'; policy_path.write_text('{}')
        settings = dict(code={'old.py': 'old'}, cohort_sha256=fingerprint('rpkv-original-ruler-13x500-seed42'),
                        prepared=str(prepared), policy=str(policy_path))
        atomic_json(self.root/'settings.json', settings)
        from runner.layout import PROMPT_PROTOCOL
        from scripts.ruler import EVALUATION_PROTOCOL
        policy = dict(prompt_protocol=PROMPT_PROTOCOL, evaluation_protocol=EVALUATION_PROTOCOL)
        with patch.object(router_control, 'code_hashes', return_value={'new.py': 'new'}), \
                patch.object(router_control, 'load_prepared', return_value=[]), \
                patch.object(router_control, 'load_policy', return_value=policy):
            protocol, _ = router_control.load_run(self.root, publish=False)
            self.assertEqual(protocol['code'], settings['code'])
            atomic_json(self.root/'settings.json', dict(settings, cohort_sha256='changed'))
            with self.assertRaisesRegex(ValueError, 'cohort'): router_control.load_run(self.root, publish=False)
            atomic_json(self.root/'settings.json', settings)
            policy['prompt_protocol'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'Policy protocol'): router_control.load_run(self.root, publish=False)

    def test_corpus_source_updates_keep_protocol_guard(self):
        prepared = self.root/'prepared'; prepared.mkdir()
        atomic_json(prepared/'plan.json', {})
        settings = dict(code={'old.py': 'old'}, dataset='ruler', kind='collection', prepared=str(prepared))
        atomic_json(self.root/'settings.json', settings)
        atomic_json(self.root/'tranche.json', dict(limit_per_task=1))
        protocol = dict(settings, plan_sha256=file_hash(prepared/'plan.json'), hardware={'existing': True})
        atomic_json(self.root/'protocol.json', protocol)
        rows = [dict(id=str(i), ordinal=0) for i in range(13)]
        with patch.object(corpus_control, 'code_hashes', return_value={'new.py': 'new'}), \
                patch.object(corpus_control, 'read_rows', return_value=rows):
            self.assertEqual(corpus_control.load_run(self.root, hardware=False), (protocol, rows))
            atomic_json(prepared/'plan.json', {'changed': True})
            with self.assertRaisesRegex(ValueError, 'Frozen protocol'):
                corpus_control.load_run(self.root, hardware=False)


if __name__ == '__main__':
    unittest.main()
