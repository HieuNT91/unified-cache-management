from runner.inputs.layout import request_metadata, METADATA_KEY, PROMPT_PROTOCOL
from runner.inputs.layout import stamp_sample
"""Bounded prompt-cache lifecycle and retired policy changes."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import runner.runtime.generation  # Keep mock targets stable across sys.modules fixtures.
from runner.runtime import prompt_cache
from runner.inputs.identity import BlockHasher, block_keys, shard_name
from runner.cache.setups import bytes_per_shard, atomic_json, file_hash
from runner.cache.temporary import TemporaryStore, configure_sparse, request_phase
from ucm.sparse.prophetkv.lifecycle import seed_value


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/'model'
        self.model.mkdir()
        atomic_json(self.model/'config.json',dict(model_type='qwen3',num_hidden_layers=64,
            hidden_size=5120,num_attention_heads=40,num_key_value_heads=8,head_dim=1))
        self.sample = stamp_sample(dict(token_ids=[1]*63+[99]+[2]*63+[99]+[3]*256,
            boundaries=[0,64,128,384],question_positions=[380],thinking=True,max_output_tokens=2,
            model_config_sha256=file_hash(self.model/'config.json')))
        self.cache=self.root/'cache'; self.cache.mkdir()
        self.namespace='a'*32
        self.args=NS(model=self.model,tp=2,memory=.9,output=self.root/'out')

    def write_chunk(self, namespace, chunk):
        hasher=BlockHasher(str(self.model),self.args.tp,0)
        for key in block_keys(hasher,chunk,seed_value(namespace)):
            for rank in range(self.args.tp):
                name=shard_name(key,str(self.model),self.args.tp,rank)
                path=self.cache/'kv'/name[:8]/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'K'*bytes_per_shard(self.model,self.args.tp))

    def populated(self, sample=None):
        cache=prompt_cache.PromptCache(self.cache,self.model,self.args.tp,self.namespace,sample or self.sample)
        for chunk in cache.chunks:
            self.write_chunk(self.namespace,chunk)
        return cache

    def test_prompt_cache_deduplicates_checks_every_rank_and_deletes_only_owned(self):
        sample=copy.deepcopy(self.sample)
        sample['token_ids'][64:128]=sample['token_ids'][:64]
        cache=self.populated(sample)
        self.assertEqual(len(cache.chunks),1)
        self.assertEqual(cache.ready()['verified_shards'],2)
        victim=self.cache/next(iter(cache.files))
        victim.write_bytes(b'incomplete')
        with self.assertRaisesRegex(RuntimeError,'changed'):
            cache.delete()
        self.assertTrue(victim.exists())
        self.write_chunk(self.namespace,cache.chunks[0])
        cache.ready()
        extra=self.cache/'kv'/'unknown'; extra.write_bytes(b'x')
        with self.assertRaisesRegex(RuntimeError,'Unexpected files'):
            cache.delete()
        extra.unlink()
        self.assertEqual(cache.delete()['deleted_shards'],2)
        self.assertFalse(list((self.cache/'kv').glob('*/*')))

    def test_tp1_cache_checks_one_full_sized_shard_per_block(self):
        self.args.tp = 1
        cache = self.populated()
        self.assertEqual(cache.ready()['verified_shards'], 2)
        self.assertEqual(bytes_per_shard(self.model,1), 2*bytes_per_shard(self.model,2))
        self.assertEqual(cache.delete()['deleted_shards'], 2)

    def test_incomplete_shard_fails_readiness(self):
        cache=self.populated()
        (self.cache/next(iter(cache.files))).unlink()
        cache.args.cache_ready_timeout_seconds=0
        with self.assertRaisesRegex(RuntimeError,'timed out'):
            cache.ready()

    def test_readiness_reports_extra_files_without_removing_them(self):
        cache=self.populated()
        extra=self.cache/'kv'/'unexpected';extra.write_bytes(b'evidence')
        with self.assertRaises(RuntimeError) as caught:
            cache.ready()
        message=str(caught.exception)
        self.assertIn('Unexpected files in temporary prompt cache',message)
        self.assertIn(f'cache={self.cache}',message)
        self.assertIn('expected=4, actual=5, extra=1, missing=0',message)
        self.assertIn('kv/unexpected',message)
        self.assertEqual(extra.read_bytes(),b'evidence')
        self.assertTrue(all((self.cache/name).exists() for name in cache.files))

    def test_readiness_waits_for_owned_staging_rename_before_snapshot(self):
        cache=self.populated()
        final=self.cache/next(iter(cache.files))
        staged=self.cache/'kv'/'.temp'/final.name
        staged.parent.mkdir()
        staged.write_bytes(final.read_bytes())
        def publish(_):
            self.assertIsNone(cache.before)
            staged.replace(final)
        with patch.object(prompt_cache.time,'sleep',side_effect=publish) as sleep:
            self.assertEqual(cache.ready()['verified_shards'],4)
        sleep.assert_called_once()
        cache.unchanged()
        self.assertEqual(cache.delete()['deleted_shards'],4)

    def test_readiness_retains_stuck_staging_and_rejects_unknown_staging(self):
        cache=self.populated()
        staged=self.cache/'kv'/'.temp'/Path(next(iter(cache.files))).name
        staged.parent.mkdir()
        staged.write_bytes(b'evidence')
        cache.args.cache_ready_timeout_seconds=0
        with self.assertRaisesRegex(RuntimeError,'Cache staging did not drain'):
            cache.ready()
        self.assertIsNone(cache.before)
        self.assertEqual(staged.read_bytes(),b'evidence')
        unknown=staged.with_name('unknown');staged.rename(unknown)
        with self.assertRaisesRegex(RuntimeError,'Unexpected files'):
            cache.ready()
        self.assertEqual(unknown.read_bytes(),b'evidence')

    def test_readiness_rechecks_replacement_size_and_rejects_staging_symlinks(self):
        cache=self.populated()
        final=self.cache/next(iter(cache.files))
        staged=self.cache/'kv'/'.temp'/final.name
        staged.parent.mkdir()
        staged.write_bytes(b'bad replacement')
        with patch.object(prompt_cache.time,'sleep',side_effect=lambda _: staged.replace(final)):
            with self.assertRaisesRegex(RuntimeError,'Cache incomplete'):
                cache.ready()
        self.assertIsNone(cache.before)
        self.populated()
        staged.symlink_to(final)
        with self.assertRaisesRegex(RuntimeError,'Symlink'):
            cache.ready()

    def test_prompt_cache_accepts_relocated_root_but_preserves_hdd_backup(self):
        original=self.populated();original.ready()
        backup=self.root/'hdd-backup'
        self.cache.rename(backup)
        target=self.root/'ssd-cache';target.mkdir()
        self.cache.symlink_to(target,target_is_directory=True)
        relocated=self.populated()
        self.assertEqual(relocated.ready()['verified_shards'],4)
        self.assertEqual(relocated.delete()['deleted_shards'],4)
        self.assertFalse(list((target/'kv').glob('*/*')))
        self.assertTrue(all((backup/name).is_file() for name in original.files))

    def test_store_blocks_writes_on_read_path(self):
        backend=Mock(); store=TemporaryStore(backend)
        with self.assertRaisesRegex(RuntimeError,'read-only'):
            store.dump_data([],[],[])
        backend.dump_data.assert_not_called()
        store.writable=True
        store.dump_data([],[],[])
        store.drain()
        backend.wait.assert_called_once()
        layouts={'prompt-0':[0,64,128,384]}
        self.assertEqual(request_phase('a:prompt-0:read:prime',layouts)[0],'read')
        for rid in ('a:prompt-1:read:x','a:prompt-0:bad:x','a:prompt-0:read'):
            with self.assertRaises(RuntimeError):request_phase(rid,layouts)

    def test_policy_switch_requires_quiescence_and_preserves_all_layers(self):
        sparse=NS(request_state=NS(pending=None),request=None,active=False,
            connector=NS(worker_request_id=None,store=NS(pending={})),tree_f3_config={})
        for chunk in (2048,4096):
            receipt=configure_sparse(sparse,'adatree-v1',None,adatree_chunk_size=chunk)
            self.assertEqual(receipt['scoring_layers'],list(range(64)))
            self.assertTrue(sparse.adatree_v1)
            self.assertEqual(sparse.adatree_chunk_size,chunk)
        for obj,key,value in [(sparse,'active',True),(sparse.request_state,'pending',object()),
            (sparse.connector,'worker_request_id','request'),(sparse.connector.store,'pending',{1:2})]:
            old=getattr(obj,key);setattr(obj,key,value)
            with self.assertRaisesRegex(RuntimeError,'Retire'):
                configure_sparse(sparse,'adatree-v1',None)
            setattr(obj,key,old)

    def record(self,pid):
        return dict(prompt_id=pid,subtask='task',accuracy=1.,thinking_tokens=2,answer_tokens=1,
            control_tokens=1,output_tokens=4,output_cap_reached=False,unfinished_thinking=False,
            timings={'ttft_seconds':2.})




    def test_real_temporary_connector_uses_frozen_boundaries_and_rejects_missing_kv(self):
        from ucm.integration.vllm.persistent_connector import PersistentBlendConnector
        connector=PersistentBlendConnector.__new__(PersistentBlendConnector)
        connector.persistent_setup=None
        connector.temporary_layouts={'prompt-0':self.sample['boundaries']}
        connector.block_size=64;connector.min_blend_threshold=1;connector.metrics_config=None
        connector.request_hasher=BlockHasher(str(self.model),self.args.tp,0)
        connector.requests_blend_meta={};connector.prompt_lengths={}
        connector.store=Mock()
        connector.store.lookup.side_effect=lambda keys:[True]*len(keys)
        rid=self.namespace+':prompt-0:read:prophetkv-1-measured'
        request=NS(request_id=rid,num_prompt_tokens=384,num_tokens=384,all_token_ids=self.sample['token_ids'],prompt_token_ids=self.sample['token_ids'],
            sampling_params=NS(extra_args={METADATA_KEY:request_metadata(rid,self.sample['token_ids'],'read',self.sample)}))
        self.assertEqual(connector.get_num_new_matched_tokens(request,0),(64,False))
        self.assertTrue(connector.temporary_readonly)
        cache=self.populated();cache.ready()
        actual_keys={key for call in connector.store.lookup.call_args_list for key in call.args[0]}
        files={f'kv/{key.hex()[:8]}/{key.hex()}' for key in actual_keys}
        self.assertTrue(files.issubset(cache.files))
        connector.requests_blend_meta.clear()
        connector.store.lookup.side_effect=lambda keys:[False]*len(keys)
        with self.assertRaisesRegex(RuntimeError,'Temporary context KV is incomplete'):
            connector.get_num_new_matched_tokens(request,0)






if __name__=='__main__':unittest.main()
