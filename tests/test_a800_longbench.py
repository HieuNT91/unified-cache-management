"""CPU scope, scheduling, shared reports and launcher regressions."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner.setups import atomic_json, file_hash
from scripts import longbench_a800_data as data
from scripts import longbench_a800_control as control
from scripts.longbench_a800_report import summarize, report, records
from runner.corpus_records import NATIVE_ANSWER_VALIDATION

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    prepared = root/'inputs'; prepared.mkdir()
    rows = []; files = {}
    for i in range(503):
        path = f'samples/{i}.json'
        atomic_json(prepared/path, dict(token_ids=[i], source_id=f'source-{i}',
                                       source_metadata=dict(length=data.LENGTHS[i % 3])))
        rows.append(dict(id=f'p{i}', prepared=path, subtask=f'domain-{i % 3}', references=['A'], scoring='longbench_v2'))
        files[path] = file_hash(prepared/path)
    manifest = prepared/'manifest.jsonl'; manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    files['manifest.jsonl'] = file_hash(manifest)
    atomic_json(prepared/'preparation.json', dict(files=files))
    groups = [[f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(j, j+4)] for j in (0, 4)]
    settings = dict(model='/missing-model', data='/missing-data', prepared=str(prepared), cache_root=str(root/'cache'),
                    groups=groups, dataset='longbench-v2', tp=2, code={}, watchdog_seconds=7200)
    atomic_json(root/'settings.json', settings)
    for role, devices in zip(('primary', 'extra'), groups):
        atomic_json(root/role/'devices.json', dict(tp=2, groups=[devices[:2], devices[2:]]))
    with patch('scripts.longbench_v2.prepare'):
        data.prepare(root)
    return settings


def record(row, case, ttft=2.):
    return dict(prompt_id=row['id'], subtask=row['subtask'], method=case, accuracy=1.,
                timings=dict(ttft_seconds=ttft), thinking_tokens=3, answer_tokens=1, control_tokens=1,
                output_tokens=5, output_cap_reached=False, unfinished_thinking=False)


class A800Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_frozen_all503_scope_group_b_has_no_baseline_and_no_fixed_probe(self):
        settings = fixture(self.root)
        _, plan, rows = data.load(self.root)
        self.assertEqual((plan['answers'], plan['probes'], len(rows)), (6036, 503, 503))
        for role in ('primary', 'extra'):
            protocol, _ = data.stage(self.root, role)
            self.assertEqual(protocol['answer_validation'], NATIVE_ANSWER_VALIDATION)
            self.assertNotIn('probe', protocol['scheduled_actions'])
            self.assertEqual(protocol['groups'], [settings['groups'][role == 'extra'][:2], settings['groups'][role == 'extra'][2:]])
        self.assertEqual(data.SCHEDULE['extra'], [f'prophetkv-{p}' for p in (10, 30, 50, 60, 80)])
        protocol, _ = data.stage(self.root, 'features')
        self.assertEqual(protocol['feature_profile']['head_layers'], [7, 15, 23, 31, 39, 47, 55, 63])
        protocol['model'] = '/other-model'; atomic_json(self.root/'features/protocol.json', protocol)
        with self.assertRaisesRegex(ValueError, 'Stage metadata'):
            data.stage(self.root, 'features')

    def test_status_cli_from_other_directory_reports_each_length_without_model(self):
        fixture(self.root)
        config = self.root/'server.env'
        config.write_text(f'PYTHON_BIN="{sys.executable}"\nEXPERIMENT_DIR="{self.root}"\n')
        for role, command in (('primary', 'status'), ('extra', 'status_same_count')):
            result = subprocess.run(['bash', str(ROOT/f'scripts/launcher/a800_longbench_{role}.sh'), command],
                cwd='/tmp', env=dict(os.environ, UCM_ENV_FILE=str(config), EXPERIMENT_DIR=str(self.root), PYTHON_BIN=sys.executable),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for label in ('OVERALL', 'SHORT', 'MEDIUM', 'LONG'):
                self.assertIn(label, result.stdout)
            saved = json.loads((self.root/f'{command}.json').read_text())
            self.assertEqual(saved['expected_answers'], 6036)
            self.assertEqual(len(saved['methods']), 12)

    def test_same_count_intersects_prompt_ids_and_retains_length_denominators(self):
        rows = [dict(id=f'p{i}', subtask='domain', length=data.LENGTHS[i % 3]) for i in range(6)]
        values = {'nocache': [record(r, 'nocache', 4.) for r in rows],
                  'prophetkv-1': [record(r, 'prophetkv-1') for r in rows[:4]],
                  'prophetkv-5': [record(r, 'prophetkv-5') for r in rows[2:]]}
        result = summarize(values, rows, True)
        self.assertEqual(result['matching']['prompt_ids'], ['p2', 'p3'])
        for item in result['methods'].values():
            self.assertEqual(item['overall']['completed'], 2)
            self.assertEqual(item['lengths']['short']['completed'], 1)
            self.assertEqual(item['lengths']['medium']['completed'], 0)
            self.assertEqual(item['lengths']['long']['expected'], 2)
        self.assertEqual(result['methods']['prophetkv-1']['lengths']['long']['ttft_speedup'], 2.)

    def test_workers_only_execute_their_own_missing_cases(self):
        row = dict(id='p', ordinal=0)
        for role, phase, expected in [('primary', 'baseline', ['nocache']),
                ('primary', 'cached', ['prophetkv-5', 'prophetkv-20', 'prophetkv-40', 'prophetkv-70', 'prophetkv-90']),
                ('extra', 'cached', data.SCHEDULE['extra']), ('primary', 'features', ['probe'])]:
            with self.subTest(role=role, phase=phase):
                target = 'features' if phase == 'features' else role
                (self.root/target).mkdir(exist_ok=True)
                protocol = dict(tp=2, groups=[['test-gpu']*2], scheduled_actions=data.SCHEDULE[target])
                engine = Mock(); engine.answer.return_value = ({}, []); engine.probe.return_value = ({}, [], {})
                engine.end.return_value = {}
                def accepted(root, case, *args):
                    return {} if case == 'prophetkv-1' else None
                with patch.object(control, 'stage', return_value=(protocol, [row])), \
                        patch.object(control, 'accepted', side_effect=accepted), patch.object(control, 'publish') as publish, \
                        patch('runner.setups.check_environment', return_value=['test-gpu']*2), \
                        patch('runner.corpus_runtime.Engine', return_value=engine):
                    control.worker(self.root, role, phase, 'attempt')
                self.assertEqual([c.args[1] for c in publish.call_args_list], expected)
                if phase == 'features':
                    engine.answer.assert_not_called(); engine.probe.assert_called_once()
                else:
                    engine.probe.assert_not_called()
                engine.close.assert_called_once(); engine.end.assert_called_once()

    def test_primary_waits_before_features_and_does_not_train_automatically(self):
        fixture(self.root); events = []
        def execute(base, role, phase, state):
            events.append(phase)
        with (patch.object(control, 'execute_phase', side_effect=execute),
                patch.object(control, 'wait_extra', side_effect=lambda _: events.append('wait-extra')),
                patch('scripts.longbench_a800_report.report', side_effect=lambda *a, **k: events.append('final-controls')),
                patch('scripts.router_dataset.export_collection', side_effect=lambda *a: events.append('export'))):
            control.supervise(self.root, 'primary')
        self.assertEqual(events, ['baseline', 'cached', 'wait-extra', 'final-controls', 'features', 'export'])
        self.assertFalse((self.root/'training').exists())
        self.assertEqual(json.loads((self.root/'primary/supervisor.json').read_text())['state'], 'complete')

    def test_failed_dependency_does_not_start_feature_collection(self):
        fixture(self.root); phases = []
        with patch.object(control, 'execute_phase', side_effect=lambda b, r, p, s: phases.append(p)), \
                patch.object(control, 'wait_extra', side_effect=RuntimeError('extra failed')):
            with self.assertRaisesRegex(RuntimeError, 'extra failed'):
                control.supervise(self.root, 'primary')
        self.assertEqual(phases, ['baseline', 'cached'])
        self.assertFalse((self.root/'primary/complete.json').exists())

    def test_committed_result_hash_corruption_is_rejected(self):
        fixture(self.root); _, _, rows = data.load(self.root); row = rows[0]
        folder = self.root/'primary/records/nocache'/row['id']
        atomic_json(folder/'result.json', record(row, 'nocache'))
        protocol, _ = data.stage(self.root, 'primary')
        from runner.corpus_records import protocol_identity
        atomic_json(folder/'validated.json', dict(complete=True, protocol_sha256=protocol_identity(protocol),
                                                 files={'result.json': '0'*64}))
        with self.assertRaisesRegex(ValueError, 'pin or protocol'):
            records(self.root)


if __name__ == '__main__':
    unittest.main()
