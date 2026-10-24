"""CPU regression for late prefix rewrites after native D2H retirement."""
import copy
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner.identity import BlockHasher, block_keys, shard_name
from runner.setups import bytes_per_shard
from runner.temporary import TemporaryStore, TreeTemporaryStore
from ucm.sparse.prophetkv.lifecycle import seed_value


class TreeTemporaryStoreTests(unittest.TestCase):
    def store(self):
        backend = Mock()
        backend.dump_data.side_effect = lambda *args: object()
        store = TreeTemporaryStore(backend)
        store.begin_namespace('a' * 32)
        store.writable = True
        return store, backend

    def test_shared_prefix_in_batch_and_later_chunk_is_submitted_once(self):
        store, backend = self.store()
        first = store.dump_data([b'a', b'a', b'b'], [0, 0, 0], [[11], [12], [13]])
        store.wait(first)  # Native file queue may still contain a and b.
        second = store.dump_data([b'a', b'c'], [0, 0], [[21], [22]])
        self.assertEqual(backend.dump_data.call_args_list[0].args,
                         ([b'a', b'b'], [0, 0], [[11], [13]]))
        self.assertEqual(backend.dump_data.call_args_list[1].args, ([b'c'], [0], [[22]]))
        store.wait(second)
        store.wait(store.dump_data([b'a'], [0], [[31]]))
        self.assertEqual(store.drain(), dict(pending=0, completed=2))
        self.assertEqual(backend.wait.call_count, 2)
        self.assertEqual(store.operations['store'], 2)
        backend.lookup.assert_not_called()

    def test_distinct_shards_and_source_rows_are_preserved(self):
        store, backend = self.store()
        store.dump_data([b'a', b'a'], [0, 1], [[10, 11], [20, 21]])
        self.assertEqual(backend.dump_data.call_args.args,
                         ([b'a', b'a'], [0, 1], [[10, 11], [20, 21]]))
        store.drain()

    def test_duplicate_submission_does_not_hide_original_pending_transfer(self):
        store, backend = self.store()
        first = store.dump_data([b'a'], [0], [[1]])
        store.wait(store.dump_data([b'a'], [0], [[2]]))
        self.assertEqual(list(store.pending.values()), [first])
        backend.wait.assert_not_called()
        backend.wait.side_effect = RuntimeError('transfer failed')
        with self.assertRaisesRegex(RuntimeError, 'transfer failed'):
            store.drain()
        self.assertEqual(list(store.pending.values()), [first])
        backend.dump_data.assert_called_once()

    def test_namespace_reset_is_bounded_and_requires_retirement(self):
        store, backend = self.store()
        store.dump_data([b'a'], [0], [[1]])
        store.begin_namespace('a' * 32)
        with self.assertRaisesRegex(RuntimeError, 'Retire'):
            store.begin_namespace('b' * 32)
        store.drain()
        store.begin_namespace('b' * 32)
        self.assertEqual(store.submitted_shards, set())
        store.dump_data([b'a'], [0], [[2]])
        self.assertEqual(backend.dump_data.call_count, 2)

    def test_submission_failure_cannot_be_swallowed_or_retried(self):
        store, backend = self.store()
        backend.dump_data.side_effect = RuntimeError('partially queued')
        with self.assertRaisesRegex(RuntimeError, 'partially queued'):
            store.dump_data([b'a'], [0], [[1]])
        for call in (lambda: store.dump_data([b'a'], [0], [[2]]), store.drain,
                     lambda: store.begin_namespace('b' * 32)):
            with self.assertRaisesRegex(RuntimeError, 'restart the worker'):
                call()
        backend.dump_data.assert_called_once()

    def test_guards_apply_even_to_duplicate_or_empty_dumps(self):
        store, backend = self.store()
        store.dump_data([b'a'], [0], [[1]])
        store.drain()
        store.writable = False
        with self.assertRaisesRegex(RuntimeError, 'read-only'):
            store.dump_data([b'a'], [0], [[1]])
        store.writable = True
        store.router_dense = True
        with self.assertRaisesRegex(RuntimeError, 'dense fallback'):
            store.dump_data([], [], [])
        store.router_dense = False
        with self.assertRaisesRegex(ValueError, 'source rows differ'):
            store.dump_data([b'a'], [], [[1]])
        unbound = TreeTemporaryStore(backend)
        unbound.writable = True
        with self.assertRaisesRegex(RuntimeError, 'namespace'):
            unbound.dump_data([], [], [])
        backend.dump_data.assert_called_once()

    def test_connector_opts_in_only_temporary_tree_and_binds_namespace(self):
        from ucm.integration.vllm.persistent_connector import PersistentBlendConnector
        from ucm.integration.vllm.blend_connector import UCMBlendConnector
        from runner.setup_store import SetupStore
        connector = PersistentBlendConnector.__new__(PersistentBlendConnector)
        connector.global_rank = 0
        backend = Mock()
        for tree in (False, True):
            connector.launch_config = dict(temporary_layouts={'p': [0, 64, 128]}, max_load_blocks=64)
            if tree:
                connector.launch_config['tree_f3'] = {}
            with patch.object(UCMBlendConnector, '_create_store', return_value=backend):
                connector.store = connector._create_store()
            self.assertIs(type(connector.store), TreeTemporaryStore if tree else TemporaryStore)
            self.assertEqual(connector.store.max_load_blocks, 64)
            if not tree:
                continue
            for phase in ('populate', 'read'):
                connector.worker_request_id = None
                rid = 'a' * 32 + ':p:' + phase + ':0'
                metadata = NS(request_meta={rid: NS(continuing=False,
                    request_layout=dict(request_id=rid, phase=phase))})
                with patch.object(UCMBlendConnector, 'bind_connector_metadata'):
                    connector.bind_connector_metadata(metadata)
                self.assertEqual(connector.store.namespace, 'a' * 32)
                self.assertEqual(connector.store.writable, phase == 'populate')
            connector.launch_config['persistent_setup'] = dict(cache_dir='/unused')
            with patch.object(UCMBlendConnector, '_create_store', return_value=backend):
                self.assertIs(type(connector._create_store()), SetupStore)

    def test_delayed_file_queue_reproduces_old_failure_and_prevents_new_rewrite(self):
        # Real PromptCache readiness/snapshot and real TP hash geometry; only
        # D2H and publication are simulated. No sleeping or GPU is required.
        from test_sweep import SweepTests
        from runner.sweep import PromptCache
        for policy in (TemporaryStore, TreeTemporaryStore):
            with self.subTest(policy=policy.__name__):
                fixture = SweepTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                sample = copy.deepcopy(fixture.sample)
                sample.update(token_ids=[1]*64+[2]*64+[1]*64+[3]*64+[4]*256,
                              boundaries=[0, 128, 256, 512], question_positions=[508])
                cache = PromptCache(fixture.cache, fixture.model, 2, fixture.namespace, sample)
                queue = []
                def submit(keys, shards, addresses):
                    queue.extend(keys)
                    return object()
                stores = []
                for rank in range(2):
                    store = policy(NS(dump_data=submit, wait=lambda _: None))
                    if isinstance(store, TreeTemporaryStore):
                        store.begin_namespace(fixture.namespace)
                    store.writable = True
                    stores.append(store)
                hasher = BlockHasher(str(fixture.model), 2, 0)
                for chunk in cache.chunks:
                    keys = block_keys(hasher, chunk, seed_value(fixture.namespace))
                    for rank, store in enumerate(stores):
                        rank_keys = [bytes.fromhex(shard_name(k, str(fixture.model), 2, rank)) for k in keys]
                        store.dump_data(rank_keys, [0]*len(keys), [[1]]*len(keys))
                        store.drain()
                self.assertTrue(all(not s.pending for s in stores))
                written, late = set(), []
                size = bytes_per_shard(fixture.model, 2)
                def publish(key, mtime):
                    name = key.hex()
                    path = fixture.cache/'kv'/name[:8]/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'K'*size)
                    os.utime(path, ns=(mtime, mtime))
                for key in queue:
                    if key in written:
                        late.append(key)
                    else:
                        publish(key, 1_000_000_000)
                        written.add(key)
                self.assertEqual(cache.ready()['verified_shards'], 6)
                for key in late:
                    publish(key, 2_000_000_000)
                if policy is TemporaryStore:
                    self.assertEqual(len(late), 2)
                    with self.assertRaisesRegex(RuntimeError, 'changed_shards=2') as caught:
                        cache.unchanged()
                    self.assertIn('before', str(caught.exception))
                    self.assertIn('after', str(caught.exception))
                else:
                    self.assertFalse(late)
                    cache.unchanged()
                    self.assertEqual(cache.delete()['deleted_shards'], 6)


if __name__ == '__main__':
    unittest.main()
