"""CPU coverage for bounded L40 cache loads; no attention changes."""
import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

import numpy as np

from runner.tree_profiles import config
from ucm.sparse.prophetkv.lifecycle import TrackedStore


class BoundedLoadTests(unittest.TestCase):
    def test_full_prompt_waits_between_batches_and_preserves_all_layer_slots(self):
        keys = [i.to_bytes(4, 'little') for i in range(1008)]
        addresses = np.arange(1008 * 128, dtype=np.uint64).reshape(1008, 128)
        received, live = [], []
        def load(ids, shards, pointers):
            self.assertEqual(live, [])
            self.assertLessEqual(len(ids), 64)
            self.assertEqual(len(ids), len(set(ids)))
            received.extend(zip(ids, shards, [list(row) for row in pointers]))
            task = object()
            live.append(task)
            return task
        def wait(task):
            self.assertIs(live.pop(), task)
        backend = types.SimpleNamespace(load_data=load, wait=wait)
        store = TrackedStore(backend, max_load_blocks=64)
        task = store.load_data(keys, [0] * len(keys), addresses)
        self.assertEqual(store.completed, 15)
        self.assertEqual(len(store.pending), 1)
        store.wait(task)
        self.assertEqual(store.drain(), dict(pending=0, completed=16))
        self.assertEqual(received, list(zip(keys, [0] * len(keys), addresses.tolist())))

    def test_repeated_files_keep_separate_destinations_and_no_oversized_read(self):
        calls = []
        def load(ids, shards, pointers):
            self.assertEqual(len(ids), len(set(ids)))
            calls.extend(zip(ids, shards, pointers))
            return object()
        backend = types.SimpleNamespace(load_data=load, wait=lambda task: None)
        store = TrackedStore(backend, max_load_blocks=2)
        keys = [b'a', b'b', b'a', b'a', b'b']
        slots = [[i, i + 20] for i in range(len(keys))]
        store.wait(store.load_data(keys, [0] * 5, slots))
        self.assertEqual(calls, list(zip(keys, [0] * 5, slots)))
        self.assertEqual(store.drain(), dict(pending=0, completed=3))

    def test_failed_batch_stays_pending_and_stops_further_submission(self):
        backend = Mock()
        first = object()
        backend.load_data.return_value = first
        backend.wait.side_effect = RuntimeError('transfer timed out')
        store = TrackedStore(backend, max_load_blocks=1)
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            store.load_data([b'a', b'b'], [0, 0], [[1], [2]])
        backend.load_data.assert_called_once()
        self.assertEqual(store.pending, {id(first): first})
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            store.drain()
        backend.wait.side_effect = None
        self.assertEqual(store.drain(), dict(pending=0, completed=1))

    def test_invalid_metadata_dense_bypass_and_empty_load(self):
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                TrackedStore(Mock(), max_load_blocks=limit)
        backend = Mock()
        store = TrackedStore(backend, max_load_blocks=64)
        for shards, slots in ((None, [[1]]), ([0], None), ([], [[1]]), ([0], [])):
            with self.assertRaises(ValueError):
                store.load_data([b'a'], shards, slots)
        store.wait(store.load_data([], [], []))
        store.router_dense = True
        with self.assertRaisesRegex(RuntimeError, 'dense fallback'):
            store.load_data([b'a'], [0], [[1]])
        backend.load_data.assert_not_called()

    def test_limits_apply_to_l40_only_and_baseline_has_no_connector(self):
        for tp in (2, 4):
            cfg = config('/model', 'ruler-thinking', True, '/cache', hardware=f'l40-tp{tp}', tp=tp)
            extra = cfg['kv_transfer_config']['kv_connector_extra_config']
            if tp == 2:
                self.assertEqual(extra['max_load_blocks'], 64)
            else:
                self.assertNotIn('max_load_blocks', extra)
            self.assertEqual(extra['ucm_connectors'][0]['ucm_connector_config']['timeout_ms'], 120000)
            dense = config('/model', 'ruler-thinking', False, '/cache', hardware=f'l40-tp{tp}', tp=tp)
            self.assertNotIn('kv_transfer_config', dense)
        cfg = config('/model', 'ruler', True, '/cache', hardware='l20-tp2', tp=2)
        self.assertNotIn('max_load_blocks', cfg['kv_transfer_config']['kv_connector_extra_config'])

    def test_connector_propagates_limit_to_every_store_policy(self):
        # Execute the actual override without importing unavailable vLLM/CUDA.
        path = Path(__file__).resolve().parents[1] / 'ucm/integration/vllm/persistent_connector.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_create_store']
        backend = object()
        class Base:
            def _create_store(self, *args, **kwargs):
                return backend
        namespace = dict(UCMBlendConnector=Base, TrackedStore=TrackedStore)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        connector = namespace['PersistentBlendConnector']()
        connector.global_rank = 0
        for settings in ({}, {'temporary_layouts': {'prompt': [0, 64]}},
                         {'persistent_setup': {'cache_dir': '/cache'}}):
            connector.launch_config = dict(settings, max_load_blocks=64)
            store = connector._create_store(None, None)
            self.assertEqual(store.max_load_blocks, 64)
            self.assertIs(store.store, backend)


if __name__ == '__main__':
    unittest.main()
