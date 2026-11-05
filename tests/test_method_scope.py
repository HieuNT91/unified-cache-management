"""Removed controls must fail before generation or input preparation."""
import json
from pathlib import Path
import tempfile
import unittest

from runner.config import engine_config
from runner.corpus_worker import tree_mode
from runner.generation import verify_diagnostics
from runner.tree_policy import inventory
from scripts.longbench_a800_data import prepare
from ucm.sparse.prophetkv.layers import resolve_layers


REMOVED = ('last_traj_4', 'last_traj_4f', 'query_traj_4f', 'naive_reuse')


class MethodScopeTests(unittest.TestCase):
    def test_removed_methods_rejected_before_worker_imports_or_generation(self):
        for method in REMOVED:
            with self.subTest(method=method):
                definition = dict(id=method, method=method, ratio=.2)
                with self.assertRaises(ValueError):
                    engine_config('/unused', method, .2)
                with self.assertRaises(ValueError):
                    resolve_layers(method)
                with self.assertRaises(ValueError):
                    tree_mode(None, 'unused', definition)
                with self.assertRaises(ValueError):
                    verify_diagnostics([], {}, method, .2, 2, range(64))
                with self.assertRaises(ValueError):
                    inventory(dict(actions=[dict(id='nocache', method='baseline', ratio=None), definition]))

    def test_removed_collection_cannot_regenerate_prepared_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = root/'settings.json'
            # Deliberately omit prepared/model paths: the old mode must be
            # rejected before looking up paths or invoking a dataset adapter.
            settings.write_text(json.dumps(dict(dataset='ruler', naive_reuse=True)))
            before = settings.read_bytes()
            with self.assertRaisesRegex(ValueError, 'no longer supported'):
                prepare(root)
            self.assertEqual(settings.read_bytes(), before)
            self.assertEqual(list(root.iterdir()), [settings])


if __name__ == '__main__':
    unittest.main()
