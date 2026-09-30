"""Training/replay use all six measured actions without changing source receipts."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from runner.corpus import Corpus
from runner.corpus_records import protocol_identity
from runner.corpus_train import train
from runner.corpus_training_data import ExtendedCorpus, open_corpus
from runner.setups import atomic_json, file_hash
from runner.tree_policy import DEFAULT_ACTIONS, load
from scripts import corpus_add_ratios as extension
from scripts.corpus_inputs import TASKS
import test_corpus_add_ratios as fixtures
from test_corpus import FakeCorpus


class ExtensionTrainingTests(unittest.TestCase):
    def completed(self, root, tasks=('task',)):
        helper = fixtures.AddRatiosTests()
        collection, rows = helper.fixture(root, groups=2, tasks=tasks)
        engine = helper.fake_engine([])

        class FastAddedActions(engine):
            def answer(self, definition, case, attention):
                record, ds = super().answer(definition, case, attention)
                record['timings']['answer_engine_ttft_seconds'] = .2 if case == 'prophetkv-10' else .4
                return record, ds

        for group in (0, 1):
            with patch('runner.corpus_runtime.Engine', FastAddedActions), \
                    patch('runner.setups.check_environment', return_value=collection.protocol['groups'][group]):
                extension.collect(root, group, 'training-fixture')
        atomic_json(collection.state/'cleanup.json', dict(owned_engines_exited=True))
        extension.finalize(root)
        return collection, rows

    def test_join_snapshot_and_feature_replay_preserve_sources(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            collection, _ = self.completed(Path(td))
            pins = {p: file_hash(p) for p in collection.root.rglob('*') if p.is_file()}
            corpus = open_corpus(collection.root)
            self.assertIsInstance(corpus, ExtendedCorpus)
            self.assertEqual(list(corpus.actions), ['nocache', 'prophetkv-1', 'prophetkv-5', 'prophetkv-10', 'prophetkv-20', 'prophetkv-40'])
            snapshot = corpus.snapshot()
            self.assertEqual(snapshot['counts']['complete'], 2)
            self.assertEqual(len(snapshot['acceptance_hashes']), 14)
            self.assertEqual(snapshot['protocol_sha256'], protocol_identity(collection.protocol))
            self.assertNotIn('task-002', corpus.rows)
            self.assertEqual(corpus.outcome('task-000', 'prophetkv-10')['timings']['answer_engine_ttft_seconds'], .2)
            self.assertEqual(corpus.outcome('task-000', 'nocache')['timings']['answer_engine_ttft_seconds'], 2.)
            features = corpus.features('task-000')
            self.assertEqual(features, corpus.probe('task-000')['features'])
            with patch.object(corpus, 'attention', side_effect=AssertionError('Features should be cached')):
                self.assertEqual(features, corpus.features('task-000'))
            self.assertEqual(pins, {p: file_hash(p) for p in pins})

    def test_no_extension_uses_original_reader_and_direct_training_upgrades(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            with patch('runner.corpus_training_data.Corpus') as original:
                open_corpus(root)
                original.assert_called_once_with(root, None)
            collection, _ = self.completed(root)
            from runner.corpus_training_data import training_view
            base = Corpus.__new__(Corpus)
            base.root, base.prepared = root, Path(collection.protocol['prepared'])
            self.assertIsInstance(training_view(base), ExtendedCorpus)

    def test_incomplete_extension_cannot_silently_train_four_actions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_json(root/'protocol.json', dict(actions=DEFAULT_ACTIONS))
            (root/extension.STATE).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'extension is incomplete'):
                open_corpus(root)
            base = FakeCorpus(root/'base', per_task=2)
            (base.root/extension.STATE).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'extension is incomplete'):
                train(base, 13, root/'training')
            self.assertFalse((root/'training').exists())

    def test_corrupt_original_and_added_records_and_changed_completion_halt(self):
        for kind in ('original', 'added', 'completion', 'missing_receipt', 'derived', 'features'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
                collection, _ = self.completed(Path(td))
                if kind == 'completion':
                    done = extension.read(collection.state/'complete.json')
                    done['new_answers'] -= 1
                    atomic_json(collection.state/'complete.json', done)
                    with self.assertRaisesRegex(ValueError, 'completion counts'):
                        open_corpus(collection.root)
                    continue
                if kind == 'missing_receipt':
                    (collection.root/'records'/'prophetkv-5'/'task-000'/'validated.json').unlink()
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        open_corpus(collection.root)
                    continue
                corpus = open_corpus(collection.root)
                if kind in ('original', 'added'):
                    case = 'nocache' if kind == 'original' else 'prophetkv-10'
                    (collection.root/'records'/case/'task-000'/'result.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError, 'Corrupt'):
                        corpus.outcome('task-000', case)
                elif kind == 'derived':
                    (collection.state/'derived'/'task-000.npz').write_bytes(b'corrupt')
                    with self.assertRaisesRegex(ValueError, 'Derived attention'):
                        corpus.features('task-000')
                else:
                    record = corpus.probe('task-000')
                    record['features'] = dict(record['features'], top1_mass=.987)
                    with patch.object(corpus, 'probe', return_value=record):
                        with self.assertRaisesRegex(ValueError, 'features differ'):
                            corpus.features('task-000')

    def test_original_tree_still_selects_original_reader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protocol = dict(actions=DEFAULT_ACTIONS)
            atomic_json(root/'protocol.json', protocol)
            (root/extension.STATE).mkdir(parents=True)
            tree = dict(actions=DEFAULT_ACTIONS, provenance={'corpus_protocol_sha256': protocol_identity(protocol)})
            with patch('runner.corpus_training_data.Corpus') as original:
                open_corpus(root, tree=tree)
                original.assert_called_once_with(root, None)

    def test_real_six_action_training_export_and_heldout_replay(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            collection, _ = self.completed(root/'corpus', tasks=TASKS)
            original_pins = dict(collection.sources['files'])
            corpus = open_corpus(collection.root)
            summary = train(corpus, 13, root/'training', seed=42, policy_count=1)
            self.assertEqual(summary['training_samples'], 13)
            self.assertEqual(summary['heldout_samples'], 13)
            self.assertEqual(len(summary['actions']), 6)
            self.assertEqual(summary['data_source'], 'completed-add5-10-extension')
            self.assertIn('different sessions', summary['timing'])
            tree = load(root/'training/router1.json')
            self.assertEqual(tree['tree']['action'], 'prophetkv-10')
            self.assertFalse(set(tree['training_ids']) & set(tree['heldout_ids']))
            self.assertEqual(len(tree['heldout_hashes']), 13*7)
            matrix = extension.read(root/'training/matrices.json')
            self.assertTrue(all(len(scores) == 6 for scores in matrix['scores']))
            report = extension.read(root/'training/router1-heldout/report.json')
            self.assertEqual(report['overall']['actions'], {'prophetkv-10': 13})
            self.assertAlmostEqual(report['overall']['estimated_router_ttft_seconds'], .21)
            self.assertEqual(original_pins, {name: file_hash(name) for name in original_pins})
            with self.assertRaises(FileExistsError):
                train(corpus, 13, root/'training', policy_count=1)
            # The CLI's default path works from another checkout without PREPARED_DIR.
            env = {k: v for k, v in os.environ.items() if k not in ('PREPARED_DIR', 'UCM_ENV_FILE')}
            env.update(CUDA_VISIBLE_DEVICES='', PYTHON_BIN=sys.executable,
                       EXPERIMENT_DIR=str(collection.root), UCM_ENV_FILE=str(root/'empty.env'))
            (root/'empty.env').write_text('')
            script = Path(__file__).resolve().parents[1]/'scripts/ruler_corpus.sh'
            run = subprocess.run(['bash', str(script), 'train', '--train-samples', '13', '--policy-count', '1',
                                  '--output', str(root/'cli-training')], cwd='/tmp', env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(extension.read(root/'cli-training/summary.json')['actions']), 6)
            run = subprocess.run(['bash', str(script), 'test', '--tree', str(root/'training/router1.json'),
                                  '--mode', 'offline', '--output', str(root/'cli-heldout')], cwd='/tmp', env=env,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(extension.read(root/'cli-heldout/report.json')['overall']['actions'], {'prophetkv-10': 13})


if __name__ == '__main__':
    unittest.main()
