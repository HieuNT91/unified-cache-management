"""CPU coverage for separate LongBench tree seeds with shared prepared inputs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from runner.setups import atomic_json, file_hash, fingerprint
from runner.tree_collection import SCHEMA
from scripts.collection_v3 import declaration, seeds
from scripts import tree_prophetkv_data as collection
from scripts import longbench_a800_control as control
from scripts.export_tree_outcomes import export
from test_tree_collection import fixture, ROOT
from test_baseline_breakdown import fixture as report_fixture, commit


class TreeAnswerSeedTests(unittest.TestCase):
    def adopt(self, args):
        with patch('scripts.router_control.code_hashes', return_value={'source': 'fixture'}):
            collection.adopt(args)

    def test_seed23_reuses_original_inputs_and_preserves_seed01_run(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), 'longbench-v2')
            args.percentages = [5,20,40,60]
            self.adopt(args)
            original = args.root
            pins = {p:(file_hash(p),p.stat().st_mtime_ns) for p in original.rglob('*') if p.is_file()}
            later = copy.copy(args)
            later.root = original.with_name('seed23')
            later.cache_root = args.cache_root.with_name('seed23-cache')
            later.answer_seeds = [2,3]
            self.adopt(later)
            settings, plan, rows = collection.load(later.root)
            self.assertEqual(settings['prepared'], str(args.prepared))
            self.assertEqual(seeds(settings), [2,3])
            self.assertEqual((len(rows), plan['answers'], plan['probes']), (503,4024,0))
            self.assertFalse((later.root/'prepared').exists())
            self.assertEqual(collection.stage(later.root, 'tree')[0]['answer_seeds'], [2,3])
            # Resume reads receipts only; it must retain the selected schedule.
            real_read = collection.read
            def read(path):
                if Path(path).parent.name == 'samples':
                    raise AssertionError('Resume reread prepared samples')
                return real_read(path)
            with patch.object(collection, 'read', side_effect=read):
                self.adopt(later)
            later.answer_seeds = [0,1]
            with self.assertRaisesRegex(ValueError, 'parameters differ'):
                self.adopt(later)
            args.answer_seeds = [2,3]
            with self.assertRaisesRegex(ValueError, 'parameters differ'):
                self.adopt(args)
            self.assertEqual(pins, {p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})

    def test_worker_only_seeds23_and_resume_skips_committed_seed2(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), 'longbench-v2')
            args.answer_seeds = [2,3]
            args.percentages = [5,20,40,60]
            self.adopt(args)
            protocol, rows = collection.stage(args.root, 'tree')
            engine = Mock(); engine.answer.return_value = ({}, []); engine.end.return_value = {}
            def accepted(root, case, row, proto, seed=None):
                return {} if case == 'tree-prophetkv-5' and seed == 2 else None
            with patch('runner.corpus_runtime.Engine', return_value=engine), \
                    patch('runner.setups.check_environment', return_value=protocol['groups'][0]), \
                    patch.object(control, 'accepted', side_effect=accepted), \
                    patch.object(control, 'publish') as publish:
                control.worker(args.root, 'tree', 'cached', 'attempt', 0,
                               stage_loader=lambda *_:(protocol, rows[:1]))
            self.assertEqual([(c.args[1],c.kwargs['seed']) for c in engine.answer.call_args_list],
                             [('tree-prophetkv-5',3)] + [(f'tree-prophetkv-{p}',s)
                                                       for p in (20,40,60) for s in (2,3)])
            self.assertEqual(publish.call_count, 7)
            engine.probe.assert_not_called()
            engine.begin.assert_called_once(); engine.end.assert_called_once()

    def test_invalid_or_unprepared_seeds_rejected_and_other_methods_unchanged(self):
        schedule = dict(schema=SCHEMA, dataset='longbench-v2', **declaration('longbench-v2',2,True))
        for selected in ([2,2], [3,2], [-1,2], [2], [], [True,3]):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError,'seed schedule'):
                seeds(dict(schedule, answer_seeds=selected))
        for schema, dataset in (('tp2-data-config-v3','longbench-v2'), (SCHEMA,'ruler')):
            with self.assertRaisesRegex(ValueError,'seed schedule'):
                seeds(dict(schedule,schema=schema,dataset=dataset,answer_seeds=[2,3]))
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), 'longbench-v2'); args.answer_seeds = [3,4]
            with self.assertRaisesRegex(ValueError, 'prepared generation/answer seeds differ'):
                self.adopt(args)
            self.assertFalse((args.root/'settings.json').exists())

    def test_report_export_uses_seed23_receipts_and_original_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target, baseline = report_fixture(root, 'longbench-v2')
            # Configure a fresh fixture before publishing any records.
            settings = json.loads((root/'settings.json').read_text()); settings['answer_seeds'] = [2,3]
            atomic_json(root/'settings.json', settings)
            plan = json.loads((root/'plan.json').read_text()); plan['settings_sha256'] = file_hash(root/'settings.json')
            atomic_json(root/'plan.json', plan)
            target[1].update(answer_seeds=[2,3], plan_sha256=fingerprint(plan))
            atomic_json(root/'tree/protocol.json', target[1])
            for seed in range(4): commit(baseline, 'nocache', seed=seed)
            for case in target[1]['scheduled_actions']:
                for seed in (2,3): commit(target, case, seed=seed)
            with np.load(export(root, quiet=True), allow_pickle=False) as data:
                self.assertEqual(len(data['prompt_id']), 10)
                self.assertEqual(set(data['prompt_id']), {'prompt-0--seed-2','prompt-0--seed-3'})
                self.assertEqual(sum(data['method'] == 'nocache'), 2)

    def test_seed23_env_and_launcher_keep_prepared_path_for_start_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'checkout with spaces'; (root/'scripts/launcher').mkdir(parents=True)
            (root/'config').mkdir()
            for name in ('server_env.sh','a800_longbench_tree_prophetkv.sh'):
                (root/'scripts/launcher'/name).write_text((ROOT/'scripts/launcher'/name).read_text())
            envfile = root/'config/.env.a800.tree-prophetkv.seed23'
            envfile.write_bytes((ROOT/'config/.env.a800.tree-prophetkv.seed23.example').read_bytes())
            fake = root/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            fake.chmod(0o755)
            for command in ('configure','start','resume','status'):
                out = subprocess.run(['bash',str(root/'scripts/launcher/a800_longbench_tree_prophetkv.sh'),command],
                    cwd='/tmp',env=dict(PATH=os.environ['PATH'],PYTHON_BIN=str(fake),UCM_ENV_FILE=str(envfile)),
                    capture_output=True,text=True)
                self.assertEqual(out.returncode,0,out.stderr)
                argv=json.loads(out.stdout)
                self.assertEqual(argv[argv.index('--answer-seeds')+1:],['2','3'])
                self.assertEqual(argv[argv.index('--answers-per-question')+1],'2')
                self.assertTrue(argv[argv.index('--prepared')+1].endswith('tree-prophetkv-2seed-v1/prepared'))
                self.assertTrue(argv[argv.index('--root')+1].endswith('tree-prophetkv-seed23-v1'))
                self.assertTrue(argv[argv.index('--cache-root')+1].endswith('tree-prophetkv-seed23-v1'))


if __name__ == '__main__':
    unittest.main()
