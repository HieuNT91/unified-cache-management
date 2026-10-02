"""A800 workflow CPU tests: scope, feature math, report pairing and handoff."""
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
import numpy as np
import torch

from runner.longbench_features import FEATURES, HEAD_LAYERS, DEFINITIONS, features
from runner.setups import atomic_json, file_hash
from scripts.longbench_a800_data import ACTIONS, SCHEDULE, protocol_for
from scripts.longbench_a800_report import summarize
from scripts import longbench_a800_control as control

ROOT = Path(__file__).resolve().parents[1]


def row(i):
    return dict(id=f'p{i}', subtask='domain', ordinal=i, sha256=str(i)*64,
                dataset='longbench-v2', length=('short', 'medium', 'long')[i % 3])


def record(i, score=1., ttft=2.):
    return dict(prompt_id=f'p{i}', subtask='domain', accuracy=score, timings=dict(ttft_seconds=ttft),
                thinking_tokens=3, answer_tokens=2, control_tokens=1, output_tokens=6,
                output_cap_reached=False, unfinished_thinking=False)


class FeatureTests(unittest.TestCase):
    def attention(self, size=100):
        return dict(layers=np.ones((4, 64, size+4), np.float32),
                    heads=np.ones((4, 8, 16, size+4), np.float32),
                    head_layers=list(HEAD_LAYERS), scores=np.ones(size, np.float32))

    def test_uniform_attention_and_prefix_exclusion(self):
        attention = self.attention()
        attention['layers'][..., :4] = 1e6
        attention['heads'][..., :4] = 1e6
        result = features(attention, 4, 104)
        for key, expected in dict(coverage5_median=.05, head_coverage1_p10=.01,
                                  coverage5_min=.05, group_agreement=1., top20_mass=.2).items():
            self.assertAlmostEqual(result[key], expected)
        self.assertEqual(set(result), set(FEATURES))

    def test_poor_heads_survive_tp_aggregation_and_weak_layer_minimum(self):
        attention = self.attention()
        # 1/4 of all captured Q heads miss global top1%, so the pooled p10 is zero.
        attention['heads'][0, :, :, 4] = 0
        attention['layers'][:, 17, 4:9] = 0
        result = features(attention, 4, 104)
        self.assertEqual(result['head_coverage1_p10'], 0.)
        self.assertEqual(result['coverage5_min'], 0.)
        self.assertAlmostEqual(result['coverage5_median'], .05)

    def test_floor_ties_missing_mass_and_corruption(self):
        attention = self.attention(19)
        result = features(attention, 4, 23)
        self.assertEqual(result['coverage5_median'], 0.)
        self.assertEqual(result['head_coverage1_p10'], 0.)
        self.assertAlmostEqual(result['top20_mass'], 3/19)
        attention['layers'].fill(0); attention['heads'].fill(0); attention['scores'].fill(0)
        self.assertTrue(all(value is None for value in features(attention, 4, 23).values()))
        attention['heads'][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError): features(attention, 4, 23)

    def test_per_head_softmax_matches_explicit_gqa_reference(self):
        from ucm.sparse.prophetkv.selection import context_head_importance, context_importance
        torch.manual_seed(3)
        q = torch.randn(7, 8, 4); k = torch.randn(29, 2, 4)
        expected = torch.stack([(q[:, h]@k[:, h//4].T/2).softmax(-1).mean(0) for h in range(8)])
        actual = context_head_importance(q, k, key_tile=6, query_tile=3)
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(actual.mean(0), context_importance(q, k, key_tile=6, query_tile=3))


class ReportTests(unittest.TestCase):
    def test_status_and_same_count_report_each_official_length(self):
        rows = [row(i) for i in range(6)]
        data = {'nocache': [record(i) for i in range(6)],
                'prophetkv-1': [record(i, ttft=1.) for i in (0, 1, 2, 3)],
                'prophetkv-5': [record(i, ttft=.5) for i in (0, 1, 2, 4)]}
        normal = summarize(data, rows)
        self.assertEqual(normal['methods']['prophetkv-1']['lengths']['long']['completed'], 1)
        matched = summarize(data, rows, True)
        self.assertEqual(matched['matching']['prompt_ids'], ['p0', 'p1', 'p2'])
        for item in matched['methods'].values():
            self.assertEqual(item['overall']['completed'], 3)
            self.assertEqual([item['lengths'][s]['completed'] for s in ('short', 'medium', 'long')], [1, 1, 1])
        self.assertEqual(matched['methods']['prophetkv-5']['lengths']['long']['ttft_speedup'], 4.)

    def test_matching_empty_and_unexpected_ids(self):
        data = {'nocache': [record(0)], 'prophetkv-1': [record(1)]}
        report = summarize(data, [row(0), row(1)], True)
        self.assertEqual(report['matching']['matched_samples'], 0)
        self.assertIsNone(report['methods']['nocache']['lengths']['long']['accuracy_percent'])
        with self.assertRaises(ValueError): summarize(data, [row(0)])


class WorkflowTests(unittest.TestCase):
    def test_disjoint_action_scopes_and_same_prepared_inputs(self):
        settings = dict(model='/model', prepared='/inputs', cache_root='/cache', groups=[['A']*4, ['B']*4])
        primary = protocol_for(settings, 'primary', 'f'*64)
        extra = protocol_for(settings, 'extra', 'f'*64)
        probe = protocol_for(settings, 'features', 'f'*64)
        self.assertEqual(primary['scheduled_actions'], ['nocache', 'prophetkv-1', 'prophetkv-5', 'prophetkv-20'])
        self.assertEqual(extra['scheduled_actions'], [f'prophetkv-{p}' for p in (10, 30, 40, 50, 60)])
        self.assertEqual(primary['prepared'], extra['prepared'])
        self.assertEqual(probe['feature_profile'], DEFINITIONS)
        self.assertEqual(probe['groups'], primary['groups'])
        self.assertEqual(len(set(primary['scheduled_actions']+extra['scheduled_actions'])), 9)
        self.assertNotIn('feature_profile', primary)

    def test_primary_handoff_only_after_both_groups_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'primary').mkdir()
            for path in ('plan.json', 'primary/protocol.json', 'features/protocol.json'):
                atomic_json(base/path, {})
            events = []
            with patch.object(control, 'execute_phase', side_effect=lambda b, r, p, s: events.append(p)), \
                 patch.object(control, 'stage', return_value=({}, [row(0)])), \
                 patch.object(control, 'wait_extra', side_effect=lambda b: events.append('wait-extra')), \
                 patch('scripts.longbench_a800_report.report', side_effect=lambda *a, **k: events.append('complete-controls')), \
                 patch('scripts.router_dataset.export_longbench', side_effect=lambda *a: events.append('export')), \
                 patch.object(control.signal, 'signal'):
                control.supervise(base, 'primary')
            self.assertEqual(events, ['baseline', 'cached', 'wait-extra', 'complete-controls', 'features', 'export'])
            self.assertTrue((base/'primary/complete.json').exists())

    def test_failed_extra_prevents_feature_phase_and_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'primary').mkdir()
            atomic_json(base/'primary/protocol.json', {})
            with patch.object(control, 'execute_phase') as execute, \
                 patch.object(control, 'stage', return_value=({}, [row(0)])), \
                 patch.object(control, 'wait_extra', side_effect=RuntimeError('failed')), \
                 patch.object(control.signal, 'signal'):
                with self.assertRaisesRegex(RuntimeError, 'failed'): control.supervise(base, 'primary')
                self.assertEqual([c.args[2] for c in execute.call_args_list], ['baseline', 'cached'])
            self.assertFalse((base/'primary/complete.json').exists())

    def test_worker_resumes_only_missing_actions_and_does_not_probe_controls(self):
        protocol = dict(groups=[['A']*4], scheduled_actions=SCHEDULE['primary'])
        engine = Mock(); engine.answer.return_value = ({}, []); engine.end.return_value = {}
        def prior(root, case, row, protocol): return {} if case == 'prophetkv-1' else None
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(control, 'stage', return_value=(protocol, [row(0)])), \
             patch.object(control, 'accepted', side_effect=prior), patch.object(control, 'publish') as pub, \
             patch('runner.corpus_runtime.Engine', return_value=engine), \
             patch('runner.setups.check_environment', return_value=['A']*4):
            base = Path(tmp); (base/'primary').mkdir()
            control.worker(base, 'primary', 'cached', 'attempt')
        engine.probe.assert_not_called()
        self.assertEqual([c.args[1] for c in engine.answer.call_args_list], ['prophetkv-5', 'prophetkv-20'])
        self.assertEqual(pub.call_count, 2)

    def test_launchers_resolve_config_and_distinct_roles_outside_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fake = root/'fake-python'
            fake.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ["CUDA_VISIBLE_DEVICES"]]))\n')
            fake.chmod(0o755)
            envfile = root/'server.env'
            envfile.write_text(f'PYTHON_BIN="{fake}"\nEXPERIMENT_DIR="/a800 run"\n')
            for role in ('primary', 'extra'):
                for command in ('stop', 'status', 'status_same_count', 'detach'):
                    result = subprocess.run(['bash', str(ROOT/f'scripts/launcher/a800_longbench_{role}.sh'), command],
                        cwd='/tmp', env=dict(PATH=os.environ['PATH'], UCM_ENV_FILE=str(envfile)), capture_output=True, text=True, check=True)
                    argv, cuda = json.loads(result.stdout)
                    self.assertEqual(argv[argv.index('--role')+1], role)
                    self.assertEqual(argv[argv.index('--root')+1], '/a800 run')
                    self.assertEqual(cuda, '')


if __name__ == '__main__': unittest.main()
