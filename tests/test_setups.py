"""Persistent setup contracts, tested with disk artifacts and CPU-only fake engines."""
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner import setups
from runner.config import validate_sample, validate_preparation, engine_config
from runner.identity import BlockHasher, block_keys, setup_seed
from runner.setup_store import SetupStore


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text(json.dumps(dict(model_type='qwen3',
            num_hidden_layers=64, hidden_size=5120, num_attention_heads=40,
            num_key_value_heads=8, head_dim=1)))
        (self.model / 'model.safetensors').write_bytes(b'fixture weights, never loaded')
        (self.model / 'tokenizer.json').write_text('{}')
        self.sample = dict(token_ids=[1]*63+[99]+[2]*63+[99]+[3]*256,
            boundaries=[0,64,128,384], question_positions=[380,381],
            max_output_tokens=1, thinking=False,
            model_config_sha256=setups.file_hash(self.model / 'config.json'))
        (self.root / 'a.json').write_text(json.dumps(self.sample))
        other = copy.deepcopy(self.sample)
        other['token_ids'][-1] = 4  # all context chunks are identical
        (self.root / 'b.json').write_text(json.dumps(other))
        self.manifest = self.root / 'prompts.jsonl'
        self.manifest.write_text('\n'.join(json.dumps(dict(id=pid, prepared=name))
            for pid, name in [('a/unsafe-path','a.json'), ('b','b.json')]))
        self.args = NS(model=self.model, manifest=self.manifest, output=self.root / 'setup',
            tp=2, memory=.9, kv_chunk_size=4096, context_length=131072, thinking=True,
            max_output_tokens=None, dry_run=False)
        self.build_calls = []

    def fake_build(self, root, descriptor, samples, missing, args, progress):
        self.build_calls.append(list(missing))
        for name in setups.required_files(descriptor['inventory'], missing):
            path = root / 'cache' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size != descriptor['inventory']['bytes_per_shard']:
                path.write_bytes(b'K'*descriptor['inventory']['bytes_per_shard'])

    def build(self):
        with patch.object(setups, 'construct_missing', side_effect=self.fake_build), \
                patch('run.tokenize_sample', side_effect=AssertionError('Imported tokens must not be tokenized')), \
                patch('sys.stdout', new=io.StringIO()):
            setups.build_setup(self.args)
        descriptor = setups.read_descriptor(self.args.output)
        inventory = json.loads((self.args.output / 'inventory.json').read_text())
        return descriptor, inventory

    def run_args(self, **kwargs):
        result = dict(setup=self.args.output, model=None, tp=None, method='prophetkv', ratio=.1,
            layers=None, num_layers=None, prompt_ids=None, max_output_tokens=None,
            memory=.9, output=self.root / 'out', dry_run=False)
        result.update(kwargs)
        return NS(**result)

    def test_manifest_evaluation_is_frozen_separately_from_kv_identity(self):
        rows=[dict(id='a/unsafe-path',prepared='a.json',subtask='retrieval',references=['answer'],scoring='exact_match'),
              dict(id='b',prepared='b.json',subtask='choices',references=['B'],scoring='choice')]
        self.manifest.write_text('\n'.join(json.dumps(row) for row in rows))
        descriptor, inventory=self.build()
        self.assertEqual(descriptor['samples'][0]['evaluation']['subtask'],'retrieval')
        self.assertEqual(descriptor['samples'][1]['evaluation']['scoring'],'choice')
        self.assertEqual(setups.load_frozen(self.args.output,descriptor)[0][1],self.sample)
        self.build()  # unchanged evaluation is reusable
        rows[0]['references']=['different']
        self.manifest.write_text('\n'.join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(ValueError,'Evaluation metadata changed'):
            self.build()
        self.args.output=self.root/'another-setup'
        other, other_inventory=self.build()
        self.assertEqual(descriptor['fingerprint'],other['fingerprint'])
        self.assertEqual(inventory,other_inventory)
        self.assertNotEqual(descriptor['evaluation_sha256'],other['evaluation_sha256'])

    def test_failed_collection_retains_live_report_without_final(self):
        self.build()
        def fail(*args):
            reporter=args[-1]
            reporter.accept(dict(prompt_id='a/unsafe-path',subtask='unlabeled',accuracy=None,
                thinking_tokens=0,answer_tokens=1,control_tokens=0,output_tokens=1,
                output_cap_reached=True,unfinished_thinking=False,timings={'ttft_seconds':.5}))
            raise RuntimeError('second prompt failed')
        with patch.object(setups,'check_environment',return_value=['GPU-a','GPU-b']), \
                patch.object(setups,'execute_collection',side_effect=fail), \
                self.assertRaisesRegex(RuntimeError,'second prompt failed'):
            setups.run_collection(self.run_args())
        live=json.loads((self.root/'out'/'live_aggregation.json').read_text())
        self.assertEqual(live['status'],'failed')
        self.assertEqual(live['overall']['completed'],1)
        self.assertFalse((self.root/'out'/'final_aggregation.json').exists())

    def test_multiple_imports_deduplicate_preserve_tokens_and_reuse_without_engine(self):
        descriptor, inventory = self.build()
        self.assertEqual(len(inventory['chunks']), 2)
        self.assertEqual(len(inventory['blocks']), 2)
        self.assertEqual(len(setups.required_files(inventory)), 4)
        self.assertEqual(setups.load_frozen(self.args.output, descriptor)[0][1], self.sample)
        self.assertEqual(setups.load_frozen(self.args.output, descriptor, ['b'])[0][0]['id'], 'b')
        for ids in (['unknown'], ['b','b'], []):
            with self.assertRaises(ValueError):
                setups.load_frozen(self.args.output, descriptor, ids)
        first = setups.snapshot(self.args.output / 'cache', inventory)
        self.args.max_output_tokens = 256
        self.build()
        self.assertEqual(len(self.build_calls), 1)
        self.assertEqual(first, setups.snapshot(self.args.output / 'cache', inventory))

    def test_interrupted_setup_and_incomplete_tp_shard_resume_only_missing(self):
        descriptor, inventory = self.build()
        state = setups.snapshot(self.args.output / 'cache', inventory)
        # Simulate missing last-rank shard, a common incomplete asynchronous dump.
        victim = setups.required_files(inventory)[-1]
        (self.args.output / 'cache' / victim).write_bytes(b'partial')
        (self.args.output / 'complete.json').unlink()
        missing = setups.incomplete_chunks(self.args.output, inventory)
        self.assertEqual(len(missing), 1)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            setups.check_receipt(self.args.output, descriptor)
        self.build()
        self.assertEqual(self.build_calls[-1], missing)
        after = setups.snapshot(self.args.output / 'cache', inventory)
        self.assertEqual({k:v for k,v in state.items() if k != victim},
                         {k:v for k,v in after.items() if k != victim})
        self.assertTrue((self.args.output / 'complete.json').exists())

    def test_failed_construction_has_no_completion_receipt_and_can_resume(self):
        def interrupted(root, descriptor, samples, missing, args, progress):
            self.fake_build(root, descriptor, samples, missing[:1], args, progress)
            raise RuntimeError('interrupted')
        with patch.object(setups, 'construct_missing', side_effect=interrupted), \
                patch('sys.stdout', new=io.StringIO()), self.assertRaisesRegex(RuntimeError, 'interrupted'):
            setups.build_setup(self.args)
        self.assertFalse((self.args.output / 'complete.json').exists())
        self.build()
        self.assertEqual(len(self.build_calls[-1]), 1)

    def test_preparation_compatibility_and_model_relocation(self):
        descriptor, _ = self.build()
        moved = self.root / 'moved-model'
        shutil.copytree(self.model, moved)
        self.args.model = moved
        self.build()
        self.assertEqual(len(self.build_calls), 1)
        for key, value in [('tp',4), ('kv_chunk_size',8192), ('context_length',65536), ('thinking',False)]:
            old = getattr(self.args, key)
            setattr(self.args, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Incompatible'):
                self.build()
            setattr(self.args, key, old)
        (moved / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            self.build()
        self.assertEqual(descriptor['fingerprint'], setups.read_descriptor(self.args.output)['fingerprint'])

    def test_changed_tokenizer_runtime_and_source_rejected(self):
        descriptor, _ = self.build()
        with patch.object(setups, 'runtime_identity', return_value={'new': 'runtime'}), \
                self.assertRaisesRegex(RuntimeError, 'runtime'):
            setups.read_descriptor(self.args.output)
        original = (self.model / 'tokenizer.json').read_text()
        (self.model / 'tokenizer.json').write_text('{"changed":1}')
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            self.build()
        (self.model / 'tokenizer.json').write_text(original)
        altered = copy.deepcopy(self.sample)
        altered['token_ids'][0] = 8
        (self.root / 'a.json').write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            self.build()

    def test_locks_multiple_readers_exclude_builder_and_builder_excludes_readers(self):
        self.build()
        with setups.setup_lock(self.args.output):
            with setups.setup_lock(self.args.output):
                pass
            with self.assertRaisesRegex(RuntimeError, 'in use'):
                with setups.setup_lock(self.args.output, build=True):
                    pass
        with setups.setup_lock(self.args.output, build=True):
            for build in (True,False):
                with self.assertRaisesRegex(RuntimeError, 'in use'):
                    with setups.setup_lock(self.args.output, build=build):
                        pass

    def test_orphan_builder_identity_blocks_reuse(self):
        self.build()
        pid = os.getpid()
        starttime = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        setups.atomic_json(self.args.output / 'owners' / 'owner.json', dict(pid=pid,starttime=starttime))
        with self.assertRaisesRegex(RuntimeError, 'still alive'):
            self.build()
        setups.atomic_json(self.args.output / 'owners' / 'owner.json', dict(pid=pid,starttime='old-pid'))
        self.build()

    def test_baseline_no_kv_lookup_and_missing_kv_cached_fails_before_engine(self):
        descriptor, inventory = self.build()
        shutil.rmtree(self.args.output / 'cache')
        with patch.object(setups, 'snapshot', side_effect=AssertionError('baseline KV lookup')), \
                patch.object(setups, 'check_environment', return_value=['GPU-a','GPU-b']), \
                patch.object(setups, 'execute_collection') as execute, \
                patch('runner.reporting.AggregationReporter.finish'):
            setups.run_collection(self.run_args(method='baseline'))
            cfg = execute.call_args.args[4]
            self.assertNotIn('kv_transfer_config', cfg)
        with patch.object(setups, 'start_engine', side_effect=AssertionError('must fail before engine')), \
                self.assertRaisesRegex(RuntimeError, 'rerun setup'):
            setups.run_collection(self.run_args(output=self.root / 'cached'))

    def test_algorithm_output_changes_reuse_identity_both_layer_interfaces(self):
        descriptor, inventory = self.build()
        for method, ratio, layers, count in [('prophetkv',.1,None,None), ('prophetkv',.8,None,None),
                ('selective_prophetkv',.2,[45,48,50,56,58],None),
                ('selective_prophetkv',.2,None,5), ('baseline',.2,None,None)]:
            args = self.run_args(method=method, ratio=ratio, layers=layers, num_layers=count,
                prompt_ids=['b'], max_output_tokens=256, dry_run=True)
            with patch('run.tokenize_sample', side_effect=AssertionError('retokenized')), \
                    patch.object(setups, 'construct_missing', side_effect=AssertionError('reconstructed')), \
                    patch('sys.stdout', new=io.StringIO()) as stdout:
                setups.run_collection(args)
                result = json.loads(stdout.getvalue())
            self.assertEqual(result['setup_fingerprint'],descriptor['fingerprint'])
            self.assertEqual(result['prompt_ids'], ['b'])
            cfg = result['engine']
            self.assertEqual(cfg['max_num_batched_tokens'], 16384)
            self.assertEqual(cfg['max_model_len'],131072)
            if method != 'baseline':
                identity = cfg['kv_transfer_config']['kv_connector_extra_config']['persistent_setup']
                self.assertEqual(identity['fingerprint'],descriptor['fingerprint'])
                self.assertTrue(identity['readonly'])
            self.assertFalse(args.output.exists())

    def test_setup_cli_dry_run_and_collection_config(self):
        env = dict(os.environ, PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES='')
        cmd = ['bash', 'run.sh', 'setup', '--model', str(self.model), '--manifest', str(self.manifest),
               '--output', str(self.args.output), '--tp', '2', '--kv-chunk-size', '8192', '--dry-run']
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['prompts'],2)
        self.assertEqual(plan['chunks'],2)
        self.assertGreater(plan['expected_kv_bytes'],0)
        self.assertFalse(self.args.output.exists())
        self.build()
        cmd = ['bash','run.sh','run','--setup',str(self.args.output),'--method','selective_prophetkv',
               '--num-layers','5','--prompt-ids','b','--output',str(self.root/'out'),'--dry-run']
        result = subprocess.run(cmd,env=env,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout)['prompt_ids'], ['b'])
        self.assertFalse((self.root / 'out').exists())

    def test_readonly_store_and_resume_preserve_completed_shards(self):
        descriptor, inventory = self.build()
        config = dict(fingerprint=descriptor['fingerprint'],tp=2,readonly=True,
            cache_dir=str(self.args.output/'cache'),bytes_per_shard=inventory['bytes_per_shard'])
        store = Mock()
        wrapped = SetupStore(store,config)
        key = bytes.fromhex(next(iter(inventory['blocks'])))
        self.assertEqual(wrapped.lookup([key]),[True])
        with self.assertRaisesRegex(RuntimeError,'read-only'):
            wrapped.dump_data([key],[0],[[1]])
        for name in ('delete','dump','remove','clear','cc_store'):
            with self.assertRaises(RuntimeError):
                getattr(wrapped,name)
        wrapped.load_data([key],[0],[[1]])
        wrapped.drain()
        self.assertEqual(wrapped.pending,{})
        store.dump_data.assert_not_called()
        config['readonly'] = False
        # A complete rank is never overwritten, even if another rank is missing.
        self.assertIsNone(wrapped.dump_data([key],[0],[[1]]))
        shard = self.args.output/'cache'/inventory['blocks'][key.hex()][1]
        shard.unlink()
        self.assertEqual(wrapped.lookup([key]),[False])
        rank_key = bytes.fromhex(shard.name)
        wrapped.dump_data([rank_key],[0],[[1]])
        store.dump_data.assert_called_once()
        with self.assertRaisesRegex(RuntimeError,'rerun setup'):
            wrapped.load_data([rank_key],[0],[[1]])

    def test_readiness_and_connector_hash_share_one_implementation(self):
        descriptor, inventory = self.build()
        from ucm.integration.vllm.ucm_connector import RequestHasher
        from runner.cache import verify_cache
        fp = descriptor['fingerprint']
        config = NS(model_config=NS(model='/different/path'), parallel_config=NS(tensor_parallel_size=2),
            kv_transfer_config=NS(kv_connector_extra_config={'persistent_setup':{'fingerprint':fp}}))
        self.assertEqual(RequestHasher(config,0)(setup_seed(fp)), BlockHasher(fp,2,0,True)(setup_seed(fp)))
        args = NS(model_path=self.model,tensor_parallel_size=2,cache_dir=self.args.output/'cache',
                  setup_fingerprint=fp,hash_seed=setup_seed(fp))
        self.assertTrue(verify_cache(args, setups.chunks_of(self.sample))['complete'])

    def test_real_connector_duplicate_chunk_layout_missing_shards_and_population_denied(self):
        descriptor, inventory = self.build()
        from ucm.integration.vllm.persistent_connector import PersistentBlendConnector
        connector = PersistentBlendConnector.__new__(PersistentBlendConnector)
        connector.persistent_setup = dict(fingerprint=descriptor['fingerprint'], tp=2,readonly=True,
            cache_dir=str(self.args.output/'cache'),bytes_per_shard=inventory['bytes_per_shard'],
            layouts={'prompt-0':self.sample['boundaries']})
        connector.block_size=64;connector.min_blend_threshold=1;connector.metrics_config=None
        connector.request_hasher=BlockHasher(descriptor['fingerprint'],2,0,True)
        connector.requests_blend_meta={};connector.prompt_lengths={}
        connector.store=SetupStore(Mock(),connector.persistent_setup)
        rid='a'*32+':prompt-0:measured'
        request=NS(request_id=rid,num_prompt_tokens=384,num_tokens=384,all_token_ids=self.sample['token_ids'])
        self.assertEqual(connector.get_num_new_matched_tokens(request,0),(64,False))
        meta=connector.requests_blend_meta[rid]
        self.assertEqual(meta.chunks_meta[0].start_token_dix,64)
        self.assertTrue(all(meta.chunks_meta[0].store_hits))
        connector.requests_blend_meta.clear()
        request.request_id='b'*32+':populate'
        with self.assertRaisesRegex(RuntimeError,'Population'):
            connector.get_num_new_matched_tokens(request,0)
        request.request_id=rid
        shard=self.args.output/'cache'/setups.required_files(inventory)[-1]
        shard.unlink()
        with self.assertRaisesRegex(RuntimeError,'rerun setup'):
            connector.get_num_new_matched_tokens(request,0)

    def test_raw_collection_tokenizes_once_with_configurable_chunks(self):
        # A tiny character tokenizer exercises real chat/question offset mapping.
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]['content'] + '\nanswer:'
            def __call__(self, text, **kwargs):
                return dict(input_ids=[ord(c) for c in text],
                            offset_mapping=[(i,i+1) for i in range(len(text))])
            def encode(self, text, **kwargs):
                return [ord(c) for c in text]
            def convert_tokens_to_ids(self, text):
                return 999
            def convert_ids_to_tokens(self, token):
                return '<|endoftext|>'
        (self.root / 'context.txt').write_text('abcdefgh'*600)
        (self.root / 'question.txt').write_text('Which item?')
        self.manifest.write_text(json.dumps(dict(id='raw',context='context.txt',question='question.txt')))
        self.args.kv_chunk_size = 1024
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()) as load, \
                patch.object(setups, 'construct_missing', side_effect=self.fake_build), \
                patch('sys.stdout', new=io.StringIO()):
            setups.build_setup(self.args)
            load.assert_called_once()
            setups.build_setup(self.args)
            load.assert_called_once()  # reusable raw inputs are not tokenized again
        descriptor=setups.read_descriptor(self.args.output)
        _, sample=setups.load_frozen(self.args.output,descriptor)[0]
        self.assertTrue(all(len(t)<=1024 for t in setups.chunks_of(sample)))
        self.assertTrue(all(p>=sample['boundaries'][-2] for p in sample['question_positions']))
        self.assertEqual(sample['max_output_tokens'],16384)

    def test_collection_executes_sequentially_with_one_engine_and_never_populates(self):
        worker=NS(setup=Mock(),arm=Mock(),drain=Mock(),retire=Mock())
        worker_patch=patch.dict(sys.modules, {'runner.worker':worker})
        worker_patch.start()
        self.addCleanup(worker_patch.stop)
        self.manifest.write_text('\n'.join(json.dumps(row) for row in [
            dict(id='a/unsafe-path',prepared='a.json',subtask='task-a',references=['answer']),
            dict(id='b',prepared='b.json',subtask='task-b',references=['wrong'])]))
        descriptor, inventory = self.build()
        before = setups.snapshot(self.args.output/'cache',inventory)
        calls=[]
        retired=[dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for i in range(2)]
        tokenizer=NS(convert_tokens_to_ids=lambda text: {'<think>':98,'</think>':97}[text],
                     convert_ids_to_tokens=lambda token: {98:'<think>',97:'</think>'}.get(token),
                     all_special_ids=[],decode=lambda tokens,**kwargs:'answer' if tokens else '')
        llm=NS(llm_engine=NS(engine_core=NS(shutdown=Mock())),get_tokenizer=lambda:tokenizer)
        def rpc(function, kwargs=None):
            if function is worker.retire:
                return retired
            if function is worker.drain:
                return []
            return [{'installed':True}]*2
        llm.collective_rpc=Mock(side_effect=rpc)
        llm.llm_engine.engine_core.shutdown.side_effect=lambda: self.assertFalse(
            (self.root/'out'/'final_aggregation.json').exists())
        def generate(engine, tokens, budget, rid, thinking=False):
            if ':prompt-1:measured' in rid:
                live=json.loads((self.root/'out'/'live_aggregation.json').read_text())
                self.assertEqual(live['overall']['completed'],1)
                self.assertFalse((self.root/'out'/'final_aggregation.json').exists())
            calls.append((list(tokens),budget,rid))
            setups.atomic_json(self.root/'out'/'scheduler.json',dict(request_id=rid,requests_blend_meta=0,requests_meta=0))
            result=NS(num_cached_tokens=64,outputs=[NS(token_ids=[7],text='answer',finish_reason='length')])
            return result,.1,.2
        with patch.object(setups,'start_engine',return_value=llm) as start, \
                patch.object(setups,'check_environment',return_value=['GPU-a','GPU-b']), \
                patch('run.generate',side_effect=generate), patch('run.verify_diagnostics') as verify, \
                patch.object(setups,'construct_missing',side_effect=AssertionError('population')), \
                patch('sys.stdout',new=io.StringIO()):
            setups.run_collection(self.run_args())
        start.assert_called_once()
        self.assertEqual(len(calls),5)  # one read-only warmup plus prime/measured per prompt
        self.assertTrue(all(len(tokens)==384 for tokens,_,_ in calls))
        self.assertTrue(all('populate' not in rid for _,_,rid in calls))
        self.assertEqual(len({rid for _,_,rid in calls}),5)
        self.assertEqual(verify.call_count,2)
        llm.llm_engine.engine_core.shutdown.assert_called_once()
        self.assertEqual(before,setups.snapshot(self.args.output/'cache',inventory))
        summary=json.loads((self.root/'out'/'summary.json').read_text())
        self.assertEqual(summary['completed'],2)
        final=json.loads((self.root/'out'/'final_aggregation.json').read_text())
        self.assertEqual(final['status'],'completed')
        self.assertEqual(final['overall']['completed'],2)
        self.assertEqual(final['overall']['mean_answer_tokens'],1)
        self.assertEqual(final['overall']['accuracy_percent'],50)
        self.assertEqual(final['subtasks']['task-a']['accuracy_percent'],100)
        self.assertEqual(final['subtasks']['task-b']['accuracy_percent'],0)
        self.assertEqual(summary['setup_fingerprint'],descriptor['fingerprint'])
        for index in range(2):
            record=json.loads((self.root/'out'/f'{index:06d}'/'result.json').read_text())
            self.assertNotIn('cache_build_and_readiness_seconds',record['timings'])
            self.assertEqual(record['construction_timings'],str(self.args.output/'progress.json'))

    def test_real_construction_loop_skips_complete_chunks_and_retires_before_completion(self):
        worker=NS(setup=Mock(),arm=Mock(),drain=Mock(),retire=Mock())
        worker_patch=patch.dict(sys.modules, {'runner.worker':worker})
        worker_patch.start()
        self.addCleanup(worker_patch.stop)
        # Prepare artifacts without a GPU, then call the actual builder loop.
        entries=setups.manifest_entries(self.manifest)
        prep=setups.preparation_identity(self.args,entries)
        descriptor,samples=setups.prepare_collection(self.args,entries,prep,write=True)
        inventory=descriptor['inventory']
        missing=list(inventory['chunks'])
        self.fake_build(self.args.output,descriptor,samples,missing[:1],self.args,{})
        before=setups.snapshot(self.args.output/'cache',inventory,missing[:1])
        llm=NS(llm_engine=NS(engine_core=NS(shutdown=Mock())),collective_rpc=Mock())
        llm.collective_rpc.side_effect=lambda fn: ([dict(rank=i,quiescent=True,transfers={'pending':0},request_bookkeeping=0)
                                                   for i in range(2)] if fn is worker.retire else [])
        def generate(engine,tokens,budget,rid):
            cid=setups.fingerprint(tokens)
            self.fake_build(self.args.output,descriptor,samples,[cid],self.args,{})
            return None,0.,0.
        progress=dict(attempts=[])
        with patch.object(setups,'check_environment'), patch.object(setups,'start_engine',return_value=llm) as start, \
                patch('run.generate',side_effect=generate) as gen:
            setups.construct_missing(self.args.output,descriptor,samples,missing,self.args,progress)
        self.assertEqual(gen.call_count,1)
        start.assert_called_once()
        self.assertEqual(before,setups.snapshot(self.args.output/'cache',inventory,missing[:1]))
        self.assertTrue(progress['attempts'][0]['engine_exited'])
        self.assertEqual(len(progress['attempts'][0]['chunks']),1)
        self.assertTrue(progress['attempts'][0]['chunks'][0]['readiness']['complete'])
        llm.llm_engine.engine_core.shutdown.assert_called_once()

    def test_limits_manifest_and_frozen_input_corruption(self):
        for chunk in (0,63,65,16385,32768,True):
            with self.assertRaises(ValueError):
                validate_preparation(chunk)
        for chunk in (64,4096,8192,16384):
            validate_preparation(chunk)
        with self.assertRaisesRegex(ValueError,'context-length'):
            validate_sample(self.sample,4096,383)
        oversized=copy.deepcopy(self.sample)
        oversized['token_ids'] += [1]*(131072-len(oversized['token_ids']))
        oversized['boundaries'][-1]=131072
        with self.assertRaisesRegex(ValueError,'plus output'):
            validate_sample(oversized)
        descriptor,_=self.build()
        path=self.args.output/descriptor['samples'][0]['file']
        path.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'Frozen sample changed'):
            setups.load_frozen(self.args.output,descriptor)
        self.manifest.write_text('{"id":"same","prepared":"a.json"}\n'*2)
        with self.assertRaisesRegex(ValueError,'unique'):
            setups.manifest_entries(self.manifest)


if __name__ == '__main__':
    unittest.main()
