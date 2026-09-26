"""Bounded cache lifecycle, policy changes and concurrent collection reporting."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner import sweep
from runner.identity import BlockHasher, block_keys, shard_name
from runner.setups import bytes_per_shard, atomic_json, file_hash
from runner.temporary import TemporaryStore, configure_sparse, request_phase
from ucm.sparse.prophetkv.lifecycle import seed_value


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/'model'
        self.model.mkdir()
        atomic_json(self.model/'config.json',dict(model_type='qwen3',num_hidden_layers=64,
            hidden_size=5120,num_attention_heads=40,num_key_value_heads=8,head_dim=1))
        self.sample = dict(token_ids=[1]*63+[99]+[2]*63+[99]+[3]*256,
            boundaries=[0,64,128,384],question_positions=[380],thinking=True,max_output_tokens=2,
            model_config_sha256=file_hash(self.model/'config.json'))
        self.cache=self.root/'cache'; self.cache.mkdir()
        self.namespace='a'*32
        self.args=NS(model=self.model,tp=2,memory=.9,output=self.root/'out')

    def write_chunk(self, namespace, chunk):
        hasher=BlockHasher(str(self.model),2,0)
        for key in block_keys(hasher,chunk,seed_value(namespace)):
            for rank in range(2):
                name=shard_name(key,str(self.model),2,rank)
                path=self.cache/'kv'/name[:8]/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'K'*bytes_per_shard(self.model,2))

    def populated(self, sample=None):
        cache=sweep.PromptCache(self.cache,self.model,2,self.namespace,sample or self.sample)
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

    def test_incomplete_shard_fails_readiness(self):
        cache=self.populated()
        (self.cache/next(iter(cache.files))).unlink()
        cache.args.cache_ready_timeout_seconds=0
        with self.assertRaisesRegex(RuntimeError,'timed out'):
            cache.ready()

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
            connector=NS(worker_request_id=None,store=NS(pending={})))
        for cfg in sweep.configurations()[1:]:
            receipt=configure_sparse(sparse,cfg['method'],cfg['ratio'],cfg['layers'])
            self.assertEqual(receipt['scoring_layers'],cfg['layers'] or list(range(64)))
            self.assertEqual(sparse.ratio,cfg['ratio'])
        for obj,key,value in [(sparse,'active',True),(sparse.request_state,'pending',object()),
            (sparse.connector,'worker_request_id','request'),(sparse.connector.store,'pending',{1:2})]:
            old=getattr(obj,key);setattr(obj,key,value)
            with self.assertRaisesRegex(RuntimeError,'Retire'):
                configure_sparse(sparse,'prophetkv',.1)
            setattr(obj,key,old)

    def record(self,pid):
        return dict(prompt_id=pid,subtask='task',accuracy=1.,thinking_tokens=2,answer_tokens=1,
            control_tokens=1,output_tokens=4,output_cap_reached=False,unfinished_thinking=False,
            timings={'ttft_seconds':2.})

    def test_concurrent_reports_wait_for_both_engine_exits(self):
        output=self.root/'reports'
        expected=[dict(prompt_id=str(i),subtask='task') for i in range(10)]
        reporters=[sweep.SharedReporter(output,expected,{'method':'prophetkv'},i,2) for i in range(2)]
        def publish(i):
            for n in range(i,10,2):reporters[i].accept(self.record(str(n)))
        with ThreadPoolExecutor(2) as pool:list(pool.map(publish,range(2)))
        live=json.loads((output/'live_aggregation.json').read_text())
        self.assertEqual(live['overall']['completed'],10)
        self.assertFalse((output/'final_aggregation.json').exists())
        reporters[0].finish()
        self.assertFalse((output/'final_aggregation.json').exists())
        reporters[1].finish()
        final=json.loads((output/'final_aggregation.json').read_text())
        self.assertEqual(final['overall']['accuracy_percent'],100.)
        self.assertEqual(final['overall']['mean_thinking_tokens'],2.)
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):
            reporters[1].finish()

    def test_failed_reports_never_finalize(self):
        output=self.root/'reports'
        expected=[dict(prompt_id='a',subtask='task')]
        reporter=sweep.SharedReporter(output,expected,{'method':'baseline'},0,1)
        reporter.accept(self.record('a'))
        reporter.update(error='validation failed')
        reporter.finish()
        self.assertFalse((output/'final_aggregation.json').exists())
        self.assertEqual(json.loads((output/'live_aggregation.json').read_text())['status'],'failed')

    def test_later_phase_failure_does_not_poison_completed_baseline_shard(self):
        output=self.root/'reports'
        expected=[dict(prompt_id=str(i),subtask='task') for i in range(2)]
        reporters=[sweep.SharedReporter(output,expected,{'method':'baseline'},i,2) for i in range(2)]
        reporters[0].accept(self.record('0'));reporters[0].finish()
        reporters[0].update(error='later cached phase failed')
        reporters[1].accept(self.record('1'));reporters[1].finish()
        self.assertTrue((output/'final_aggregation.json').exists())

    def test_real_temporary_connector_uses_frozen_boundaries_and_rejects_missing_kv(self):
        from ucm.integration.vllm.persistent_connector import PersistentBlendConnector
        connector=PersistentBlendConnector.__new__(PersistentBlendConnector)
        connector.persistent_setup=None
        connector.temporary_layouts={'prompt-0':self.sample['boundaries']}
        connector.block_size=64;connector.min_blend_threshold=1;connector.metrics_config=None
        connector.request_hasher=BlockHasher(str(self.model),2,0)
        connector.requests_blend_meta={};connector.prompt_lengths={}
        connector.store=Mock()
        connector.store.lookup.side_effect=lambda keys:[True]*len(keys)
        rid=self.namespace+':prompt-0:read:prophetkv-1-measured'
        request=NS(request_id=rid,num_prompt_tokens=384,num_tokens=384,all_token_ids=self.sample['token_ids'])
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

    def test_cli_dry_run_partitions_prompts_and_keeps_baseline_connector_free(self):
        import os
        import subprocess
        import sys
        root=Path(__file__).resolve().parents[1]
        manifest=self.root/'manifest.jsonl'
        paths=[]
        for i in range(3):
            path=self.root/f'{i}.json';atomic_json(path,self.sample)
            paths.append(dict(id=str(i),prepared=str(path),subtask='task',references=['B'],scoring='choice'))
        manifest.write_text(''.join(json.dumps(row)+'\n' for row in paths))
        for shard,count in ((0,2),(1,1)):
            result=subprocess.run(['bash',str(root/'run.sh'),'sweep','--model',str(self.model),
                '--manifest',str(manifest),'--output',str(self.args.output),'--cache-root',str(self.cache),
                '--tp','2','--shard',str(shard),'--shards','2','--dry-run'],cwd=root,
                env=dict(os.environ,PYTHON_BIN=sys.executable,CUDA_VISIBLE_DEVICES=''),
                check=True,capture_output=True,text=True)
            report=json.loads(result.stdout)
            self.assertEqual(report['measurements'],count*13)
            self.assertNotIn('kv_transfer_config',report['baseline'])
            self.assertEqual(report['cached']['max_num_batched_tokens'],16384)
            self.assertEqual(report['cached']['rope_scaling']['factor'],4.)
        self.assertFalse(self.args.output.exists())
        self.assertFalse(list(self.cache.iterdir()))

    def test_failed_phase_deletes_owned_cache_only_after_successful_shutdown(self):
        group=self.root/'group';group.mkdir()
        selected=[(0,dict(id='0'),self.sample)]
        worker=NS(setup=Mock(),arm=Mock(),drain=Mock(),configure=Mock())
        for shutdown_fails in (False,True):
            with self.subTest(shutdown_fails=shutdown_fails):
                self.cache.mkdir(exist_ok=True)
                sentinel=self.cache/'owned-artifact';sentinel.write_text('owned')
                def shutdown():
                    self.assertTrue(sentinel.exists())
                    if shutdown_fails:raise RuntimeError('shutdown failed')
                llm=NS(llm_engine=NS(engine_core=NS(shutdown=shutdown)),
                       collective_rpc=Mock(side_effect=RuntimeError('worker setup failed')))
                with patch.dict('sys.modules',{'runner.worker':worker}), \
                        patch.object(sweep,'start_engine',return_value=llm), \
                        self.assertRaises(RuntimeError):
                    sweep.execute_phase(self.args,selected,sweep.configurations()[1:],{},group,
                                        self.cache,['GPU-a','GPU-b'])
                self.assertEqual(sentinel.exists(),shutdown_fails)

    def test_two_prompts_construct_once_reuse_twelve_policies_then_delete(self):
        group=self.root/'group';group.mkdir()
        evaluation=dict(subtask='task',references=['B'],scoring='choice')
        selected=[(i,dict(id=str(i),sha256='input',evaluation=evaluation),copy.deepcopy(self.sample)) for i in range(2)]
        expected=[dict(prompt_id=str(i),subtask='task') for i in range(2)]
        configs=sweep.configurations()
        reporters={c['name']:sweep.SharedReporter(self.args.output/c['name'],expected,
                   {'method':c['method']},0,1) for c in configs}
        events=[]; policy={}
        worker=NS(setup=Mock(),arm=Mock(),drain=Mock(),configure=Mock(),retire=Mock())
        class Tokenizer:
            all_special_ids=[97,98,99]
            def convert_tokens_to_ids(self,t):return {'<think>':98,'</think>':97}[t]
            def convert_ids_to_tokens(self,t):return {98:'<think>',97:'</think>'}[t]
            def decode(self,t,**kw):return 'B'
        def start(cfg):
            events.append(('start','kv_transfer_config' in cfg))
            llm=NS(llm_engine=NS(engine_core=NS(shutdown=lambda:events.append(('shutdown',)))))
            llm.get_tokenizer=lambda:Tokenizer()
            def rpc(fn,kwargs=None):
                if fn is worker.configure:
                    policy.update(kwargs)
                    from ucm.sparse.prophetkv.layers import resolve_layers
                    return [dict(rank=i,method=kwargs['method'],ratio=kwargs['ratio'],
                        scoring_layers=list(resolve_layers(kwargs['method'],kwargs['layers']))) for i in range(2)]
                if fn is worker.retire:
                    events.append(('retire',))
                    return [dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(2)]
                if fn is worker.drain:return []
                return []
            llm.collective_rpc=rpc
            return llm
        def generate(engine,tokens,budget,rid,thinking=False):
            parts=rid.split(':')
            events.append(('generate',rid))
            if len(parts)==4 and parts[2]=='populate':
                if parts[3]=='0':
                    # The previous prompt must have been deleted before this one starts.
                    self.assertFalse(list((self.cache/'kv').glob('*/*')))
                self.write_chunk(parts[0],tokens)
            if len(parts)==4 and parts[2]=='read':
                atomic_json(group/'scheduler.json',dict(request_id=rid,requests_blend_meta=0,requests_meta=0))
            return NS(outputs=[NS(token_ids=[97,10],text='B',finish_reason='stop')],num_cached_tokens=0),.2,.4
        with patch.dict('sys.modules',{'runner.worker':worker}), \
                patch.object(sweep,'start_engine',side_effect=start), \
                patch('run.generate',side_effect=generate),patch('run.verify_diagnostics') as verify:
            sweep.execute_phase(self.args,selected,configs[:1],reporters,group,self.cache,['GPU-a','GPU-b'])
            self.assertFalse(list(self.cache.iterdir()))  # baseline made no cache lookup/population
            sweep.execute_phase(self.args,selected,configs[1:],reporters,group,self.cache,['GPU-a','GPU-b'])
        self.assertEqual([e for e in events if e[0]=='start'],[('start',False),('start',True)])
        self.assertEqual(len([e for e in events if e[0]=='generate' and ':populate:' in e[1]]),4)
        self.assertEqual(verify.call_count,26)
        self.assertFalse(self.cache.exists())
        for i in range(2):
            self.assertEqual(json.loads((group/'cleanup'/f'{i:06d}.json').read_text())['remaining_shards'],0)
        for cfg in configs:
            final=json.loads((self.args.output/cfg['name']/'final_aggregation.json').read_text())
            self.assertEqual(final['overall']['completed'],2)
            self.assertEqual(final['overall']['accuracy_percent'],100.)


if __name__=='__main__':unittest.main()
