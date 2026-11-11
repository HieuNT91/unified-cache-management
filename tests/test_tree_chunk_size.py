"""Chunk-size metadata, immutable preparation, scheduler/worker agreement."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import runner.runtime.generation  # Keep patch targets stable across sys.modules fixtures.
from runner.inputs.layout import stamp_sample, request_metadata
from runner.cache.setups import atomic_json, file_hash
from runner.tree.collection import runtime_sample, TreePromptCache
from runner.corpus.runtime import Engine
from scripts.collection import tree_prophetkv_data as collection
import test_collection_no_priming as no_priming


def enlarged(sample):
    return stamp_sample(dict(sample, token_ids=list(range(10048)),
        boundaries=[0,4096,8192,9728,10048], question_positions=[9728,9729]))


class ChunkSizeTests(unittest.TestCase):
    def test_only_boundaries_change_and_default_preserves_exact_original(self):
        source = enlarged({'thinking':True}); before = copy.deepcopy(source)
        for size in (2048, 4096):
            with self.subTest(size=size):
                result = runtime_sample(source, {'kv_chunk_size':size})
                self.assertEqual(result['boundaries'], list(range(0,9728,size))+[9728,10048])
                self.assertEqual(source, before)
                self.assertEqual({k:v for k,v in result.items() if k!='boundaries'},
                                 {k:v for k,v in source.items() if k!='boundaries'})
        self.assertIs(runtime_sample(source, {}), source)
        self.assertEqual(runtime_sample(source, {'kv_chunk_size':4096}), source)
        for size in (True, 256, 1000, 8192, '1024'):
            with self.assertRaises(ValueError): runtime_sample(source, {'kv_chunk_size':size})
        source['token_ids'][1] = 123
        with self.assertRaisesRegex(ValueError, 'identity'): runtime_sample(source, {'kv_chunk_size':2048})

    def test_worker_populates_requested_chunks_and_reads_original_full_tokens(self):
        for size in (2048, 4096):
            with self.subTest(size=size):
                with tempfile.TemporaryDirectory() as temp:
                    root=Path(temp); engine,row,sample=no_priming.NoPrimingTests().fixture(root)
                    sample=enlarged(sample); atomic_json(root/'sample.json',sample)
                    row['sha256']=file_hash(root/'sample.json')
                    pins=(file_hash(root/'sample.json'),(root/'sample.json').stat().st_mtime_ns)
                    engine.tree_cached=True; engine.protocol.update(kv_chunk_size=size,percentages=[5,20,40,60])
                    with patch('runner.corpus.runtime.validate'), patch('runner.runtime.generation.generate') as generate, \
                            patch('runner.corpus.runtime.retirement'), patch('runner.corpus.runtime.clean_capture'), \
                            patch.object(TreePromptCache,'ready',return_value={}), patch.object(TreePromptCache,'unchanged'):
                        engine.begin(row)
                    chunks=[call.args[1] for call in generate.call_args_list]
                    self.assertEqual([len(c) for c in chunks], [size]*(9728//size) + ([9728%size] if 9728%size else []))
                    self.assertEqual(sum(chunks,[]),sample['token_ids'][:9728])
                    self.assertEqual(engine.sample['token_ids'],sample['token_ids'])
                    metadata=request_metadata('read-request',sample['token_ids'],'read',engine.sample)
                    self.assertEqual(metadata['boundaries'],engine.sample['boundaries'])
                    self.assertEqual(engine.construction['layout']['prepared_boundaries'],sample['boundaries'])
                    self.assertEqual(engine.construction['layout']['runtime_boundaries'],metadata['boundaries'])
                    self.assertEqual(pins,(file_hash(root/'sample.json'),(root/'sample.json').stat().st_mtime_ns))
