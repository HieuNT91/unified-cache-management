"""CPU coverage for bounded L40 cache loads; no attention changes."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from runner.tree_profiles import config
from ucm.sparse.prophetkv.lifecycle import TrackedStore

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE = 'ucm.sparse.prophetkv.lifecycle'


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
        with self.assertLogs(LIFECYCLE, level='INFO') as logs:
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                store.load_data([b'a', b'b'], [0, 0], [[1], [2]])
        self.assertIn('wait failed', logs.output[-1])
        self.assertFalse(any('complete' in line for line in logs.output))
        backend.load_data.assert_called_once()
        self.assertEqual(store.pending, {id(first): first})
        self.assertIn(id(first), store._load_timings)
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            store.drain()
        backend.wait.side_effect = None
        self.assertEqual(store.drain(), dict(pending=0, completed=1))
        self.assertEqual(store._load_timings, {})

    def test_progress_includes_submit_time_and_wait_for_final_batch(self):
        # Every transfer is below 120s, yet the complete RPC exceeds 300s.
        # Fake elapsed time exercises this boundary without sleeping or a GPU.
        now = [0.0]
        def load(*args):
            now[0] += 0.25
            return object()
        def wait(task):
            now[0] += 20
        store = TrackedStore(types.SimpleNamespace(load_data=load, wait=wait), max_load_blocks=64)
        with patch(LIFECYCLE + '.monotonic', side_effect=lambda: now[0]):
            with self.assertLogs(LIFECYCLE, level='INFO') as logs:
                task = store.load_data(list(range(1017)), [0] * 1017, [[i] for i in range(1017)])
                self.assertFalse(any('all batches complete' in line for line in logs.output))
                self.assertEqual(store.completed, 15)
                self.assertEqual(len(store._load_timings), 1)
                store.wait(task)
        completed = [line for line in logs.output if 'batch_seconds=' in line]
        self.assertEqual(len(completed), 16)
        self.assertIn('batch=16/16 blocks=57 complete batch_seconds=20.250', completed[-1])
        self.assertIn('all batches complete total_seconds=324.000', logs.output[-1])
        self.assertEqual(store._load_timings, {})
        self.assertEqual(store.drain(), dict(pending=0, completed=16))

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


class L40LauncherTimeoutTests(unittest.TestCase):
    def test_timeout_defaults_and_env_precedence_reach_child_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkout with spaces'
            launchers = root / 'scripts/launcher'
            launchers.mkdir(parents=True)
            for name in ('l40_ruler_thinking_data.sh', 'server_env.sh'):
                shutil.copyfile(ROOT / 'scripts/launcher' / name, launchers / name)
            fake = Path(tmp) / 'fake python'
            # The launcher starts this CPU-only stand-in, which also checks
            # inheritance into its worker subprocess, without loading vLLM.
            fake.write_text(f'#!{sys.executable}\n'
                'import json, os, subprocess, sys\n'
                'key = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"\n'
                'child = subprocess.check_output([sys.executable, "-c", '
                '\'import os; print(os.environ.get("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "unset"))\'], text=True).strip()\n'
                'print(json.dumps([sys.argv[1:], os.environ.get(key, "unset"), child, os.environ.get("CUDA_VISIBLE_DEVICES")]))\n')
            fake.chmod(0o755)
            for command in ('resume', 'detach', 'compute-feature'):
                for tp, file_timeout, inherited, expected in (
                    (2, None, None, '2400'), (2, '2100', None, '2100'),
                    (2, '2100', '2500', '2500'), (4, None, None, 'unset'),
                    (4, '2100', None, '2100')):
                    with self.subTest(command=command, tp=tp, file=file_timeout, inherited=inherited):
                        content = f'PYTHON_BIN="{fake}"\nGPU_DEVICES=GPU-fixture\nTP={tp}\n'
                        if file_timeout is not None:
                            content += f'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS={file_timeout}\n'
                        (root / '.env.l40.thinking').write_text(content)
                        env = {'PATH': os.environ['PATH']}
                        if inherited is not None:
                            env['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS'] = inherited
                        result = subprocess.run(['bash', str(launchers / 'l40_ruler_thinking_data.sh'), command],
                            cwd='/tmp', env=env, text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        argv, timeout, child_timeout, cuda = json.loads(result.stdout)
                        self.assertEqual(argv[1], command)
                        self.assertEqual((timeout, child_timeout), (expected, expected))
                        self.assertEqual(cuda, '')


if __name__ == '__main__':
    unittest.main()
