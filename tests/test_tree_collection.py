"""CPU-only native TreeKV construction, readiness, retirement and warmup."""
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

import runner.cache.lifecycle  # Load before fixtures replace sys.modules temporarily.
from runner.inputs.layout import stamp_sample
from runner.cache.setups import atomic_json, file_hash
from runner.runtime.output_budget import LONGBENCH_PROTOCOL, make_policy
from runner.runtime.thinking_budget import KEY
from scripts.collection.collection_v3 import declaration
from scripts.collection import tree_prophetkv_data as collection
from scripts.collection import longbench_a800_control as control
from test_ruler_thinking import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, dataset='ruler'):
    from test_run_controller import fixture as run_fixture
    return run_fixture(root, dataset, 4096)


class TreeCollectionTests(unittest.TestCase):
    def test_local_generation_progress_reports_first_and_final_token_counts(self):
        from runner.runtime.generation import generate
        class Engine:
            active = False
            count = 0
            def has_unfinished_requests(self): return self.active
            def add_request(self, rid, prompt, params): self.active = True; self.rid = rid
            def step(self):
                self.count += 1; self.active = self.count < 3
                return [NS(request_id=self.rid, outputs=[NS(token_ids=[65]*self.count)], finished=not self.active)]
        progress = Mock()
        with patch.dict(sys.modules, {'vllm': NS(SamplingParams=lambda **kw:NS(**kw)),
                                     'vllm.inputs': NS(TokensPrompt=lambda **kw:kw)}), \
                patch('runner.runtime.generation.time.perf_counter', return_value=1.):
            generate(Engine(), [65], 3, 'progress-test', progress=progress)
        values = [c.args[0] for c in progress.call_args_list]
        self.assertEqual([v['output_tokens'] for v in values], [1,3])
        self.assertEqual([v['finished'] for v in values], [False,True])


    def adopt(self, args):
        with patch('runner.operations.code_hashes', return_value={'source': 'original'}):
            collection.adopt(args)







    def test_f3_cache_readiness_immutability_retirement_and_warmup(self):
        from runner.tree.collection import TreePromptCache, retire_warmup
        from runner.runtime.sweep import PromptCache
        from runner.tree.prophetkv import write_f3, f3_path
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            sample = dict(token_ids=list(range(576)), boundaries=[0,64,320,576])
            pc = object.__new__(TreePromptCache); pc.cache = cache; pc.sample = sample; pc.f3_before = None
            warm = [100,200,300,400]*32
            write_f3(cache, str(cache), warm, [0.]*len(warm)); retire_warmup(cache, warm)
            self.assertEqual(list((cache/'tree-f3').iterdir()), [])
            with patch.object(PromptCache, 'ready', return_value={'verified_shards':10}), \
                    patch.object(PromptCache, 'unchanged'), patch.object(PromptCache, 'delete', return_value={'deleted_shards':10}):
                with self.assertRaises(RuntimeError): pc.ready()
                for a,z in zip(sample['boundaries'][:-2], sample['boundaries'][1:-1]):
                    tokens = sample['token_ids'][a:z]
                    write_f3(cache, str(cache), tokens, [0.]*len(tokens))
                self.assertEqual(len(pc.ready()['tree_f3']), 2); pc.unchanged()
                path = f3_path(cache, sample['token_ids'][:64]); original = path.read_bytes()
                path.write_text('{}')
                with self.assertRaises(RuntimeError): pc.unchanged()
                path.write_bytes(original); pc.unchanged()
                extra = cache/'tree-f3/unknown.json'; extra.write_text('{}')
                with self.assertRaises(RuntimeError): pc.unchanged()
                extra.unlink()
                result = pc.delete()
                self.assertEqual(result['deleted_f3_records'], 2)
                self.assertEqual(list((cache/'tree-f3').iterdir()), [])

    def test_engine_begin_uses_tree_cache_and_only_population_before_answers(self):
        from test_collection_no_priming import NoPrimingTests
        with tempfile.TemporaryDirectory() as temp:
            engine, row, sample = NoPrimingTests().fixture(Path(temp))
            engine.tree_cached = True; engine.protocol['percentages'] = [5,20,40,60]
            cache = Mock(chunks=[sample['token_ids'][:64],sample['token_ids'][64:320]])
            with patch('runner.corpus.runtime.validate'), patch('runner.tree.collection.TreePromptCache', return_value=cache) as ctor, \
                    patch('runner.runtime.generation.generate') as generate, patch('runner.corpus.runtime.retirement'), \
                    patch('runner.corpus.runtime.clean_capture'):
                engine.begin(row)
            ctor.assert_called_once(); cache.ready.assert_called_once(); cache.unchanged.assert_called_once()
            self.assertEqual(generate.call_count, 2)
            self.assertTrue(all(c.kwargs == {'phase':'populate'} for c in generate.call_args_list))

    def test_real_engine_constructor_enables_f3_and_retires_warmup(self):
        from runner.corpus.runtime import Engine
        from runner.tree.prophetkv import write_f3
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp)); self.adopt(args)
            protocol, rows = collection.stage(args.root, 'tree')
            llm = Mock(); llm.collective_rpc.return_value = []; configs = []
            def start(cfg):
                configs.append(cfg); return llm
            def warmup(_engine, tokens, cap, rid, **kwargs):
                self.assertEqual(kwargs, {'phase':'populate'})
                f3 = configs[0]['kv_transfer_config']['kv_connector_extra_config']['tree_f3']
                write_f3(Path(f3['cache_dir']), f3['namespace'], tokens, [0.]*len(tokens))
            with patch.dict(sys.modules, {'runner.runtime.worker':NS(setup=Mock())}), \
                    patch('runner.corpus.runtime.start_engine', side_effect=start), \
                    patch('runner.corpus.runtime.validate_initialization'), \
                    patch('runner.operations.code_hashes', return_value={}), \
                    patch('runner.reporting.aggregate.OutputAnalyzer'), \
                    patch('runner.runtime.generation.generate', side_effect=warmup) as generate, \
                    patch('runner.cache.lifecycle.wait_for_cache', return_value={'verified_shards':2*args.tp}), \
                    patch('runner.corpus.runtime.retirement') as retired:
                engine = Engine(args.root/'tree', protocol, 0, 'cached', 'test', rows[:1])
            self.assertEqual(list((engine.cache/'tree-f3').iterdir()), [])
            generate.assert_called_once(); retired.assert_called_once_with(llm, 4)
            self.assertTrue(collection.read(engine.session/'initialization.json')['validated'])
            engine.close(); llm.llm_engine.engine_core.shutdown.assert_called_once()




if __name__ == '__main__': unittest.main()
