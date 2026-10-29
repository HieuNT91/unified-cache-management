"""Chunk-1024 metadata, immutable preparation, scheduler/worker agreement."""
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

import runner.generation  # Keep patch targets stable across sys.modules fixtures.
from runner.layout import stamp_sample, request_metadata
from runner.setups import atomic_json, file_hash
from runner.tree_collection import runtime_sample, TreePromptCache
from runner.corpus_runtime import Engine
from scripts import tree_prophetkv_data as collection
from test_tree_collection import fixture, ROOT
from test_collection_no_priming import NoPrimingTests


def enlarged(sample):
    return stamp_sample(dict(sample, token_ids=list(range(10048)),
        boundaries=[0,4096,8192,9728,10048], question_positions=[9728,9729]))


class ChunkSizeTests(unittest.TestCase):
    def test_only_boundaries_change_and_default_preserves_exact_original(self):
        source = enlarged({'thinking':True}); before = copy.deepcopy(source)
        result = runtime_sample(source, {'kv_chunk_size':1024})
        self.assertEqual(result['boundaries'], list(range(0,9728,1024))+[9728,10048])
        self.assertEqual(source, before)
        self.assertEqual({k:v for k,v in result.items() if k!='boundaries'},
                         {k:v for k,v in source.items() if k!='boundaries'})
        self.assertIs(runtime_sample(source, {}), source)
        self.assertEqual(runtime_sample(source, {'kv_chunk_size':4096}), source)
        for size in (True, 1000, '1024'):
            with self.assertRaises(ValueError): runtime_sample(source, {'kv_chunk_size':size})
        source['token_ids'][1] = 123
        with self.assertRaisesRegex(ValueError, 'identity'): runtime_sample(source, {'kv_chunk_size':1024})

    def test_worker_populates_1024_chunks_and_reads_original_full_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); engine,row,sample=NoPrimingTests().fixture(root)
            sample=enlarged(sample); atomic_json(root/'sample.json',sample)
            row['sha256']=file_hash(root/'sample.json')
            pins=(file_hash(root/'sample.json'),(root/'sample.json').stat().st_mtime_ns)
            engine.tree_cached=True; engine.protocol.update(kv_chunk_size=1024,percentages=[5,20,40,60])
            with patch('runner.corpus_runtime.validate'), patch('runner.generation.generate') as generate, \
                    patch('runner.corpus_runtime.retirement'), patch('runner.corpus_runtime.clean_capture'), \
                    patch.object(TreePromptCache,'ready',return_value={}), patch.object(TreePromptCache,'unchanged'):
                engine.begin(row)
            chunks=[call.args[1] for call in generate.call_args_list]
            self.assertEqual([len(c) for c in chunks], [1024]*9+[512])
            self.assertEqual(sum(chunks,[]),sample['token_ids'][:9728])
            self.assertEqual(engine.sample['token_ids'],sample['token_ids'])
            metadata=request_metadata('read-request',sample['token_ids'],'read',engine.sample)
            self.assertEqual(metadata['boundaries'],engine.sample['boundaries'])
            self.assertEqual(engine.construction['layout']['prepared_boundaries'],sample['boundaries'])
            self.assertEqual(engine.construction['layout']['runtime_boundaries'],metadata['boundaries'])
            self.assertEqual(pins,(file_hash(root/'sample.json'),(root/'sample.json').stat().st_mtime_ns))

    def test_both_cohorts_adopt_old_inputs_scheduler_layout_and_resume(self):
        for dataset in ('ruler','longbench-v2'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                args=fixture(Path(temp),dataset)
                receipt=json.loads((args.prepared/'preparation.json').read_text())
                for path in (args.prepared/'samples').glob('*.json'):
                    atomic_json(path,enlarged(json.loads(path.read_text())))
                    receipt['files'][str(path.relative_to(args.prepared))]=file_hash(path)
                atomic_json(args.prepared/'preparation.json',receipt)
                pins={p:(file_hash(p),p.stat().st_mtime_ns) for p in args.prepared.rglob('*') if p.is_file()}
                args.root=args.root.with_name('chunk1024');args.kv_chunk_size=1024
                args.percentages=[5,20,40,60]
                with patch('scripts.router_control.code_hashes',return_value={}): collection.adopt(args)
                protocol,rows=collection.stage(args.root,'tree')
                self.assertEqual(protocol['answer_seeds'],[0] if dataset=='ruler' else [0,1])
                self.assertEqual(protocol['kv_chunk_size'],1024)
                self.assertFalse((args.root/'prepared').exists())
                # Real engine config path, stopped before any model/GPU startup.
                def capture(cfg):
                    bounds=cfg['kv_transfer_config']['kv_connector_extra_config']['temporary_layouts'][rows[0]['id']]
                    self.assertEqual(bounds,list(range(0,9728,1024))+[9728,10048])
                    raise RuntimeError('captured before GPU')
                with patch.dict(sys.modules,{'runner.worker':NS(setup=None)}), \
                        patch('runner.corpus_runtime.start_engine',side_effect=capture), \
                        self.assertRaisesRegex(RuntimeError,'captured before GPU'):
                    Engine(args.root/'tree',protocol,0,'cached','fixture',rows[:1])
                collection.adopt(args)
                args.kv_chunk_size=None
                with self.assertRaisesRegex(ValueError,'parameters differ'): collection.adopt(args)
                args.kv_chunk_size=4096
                with self.assertRaisesRegex(ValueError,'parameters differ'): collection.adopt(args)
                self.assertEqual(pins,{p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})

    def test_launcher_templates_forward_chunk_size_and_keep_old_prepared(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'scripts/launcher').mkdir(parents=True);(root/'config').mkdir()
            (root/'scripts/launcher/server_env.sh').write_bytes((ROOT/'scripts/launcher/server_env.sh').read_bytes())
            fake=root/'python';fake.write_text(f'#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n');fake.chmod(0o755)
            for server,name in (('l20','l20_ruler_thinking_tree_prophetkv.sh'),('a800','a800_longbench_tree_prophetkv.sh')):
                launcher=root/'scripts/launcher'/name;launcher.write_bytes((ROOT/'scripts/launcher'/name).read_bytes())
                envfile=root/'config'/f'.env.{server}.tree-prophetkv.chunk1024'
                envfile.write_bytes((ROOT/'config'/f'.env.{server}.tree-prophetkv.chunk1024.example').read_bytes())
                for command in ('configure','start','resume'):
                    result=subprocess.run(['bash',str(launcher),command],cwd='/tmp',
                        env=dict(PATH=os.environ['PATH'],PYTHON_BIN=str(fake),UCM_ENV_FILE=str(envfile)),capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr);argv=json.loads(result.stdout)
                    self.assertEqual(argv[argv.index('--kv-chunk-size')+1],'1024')
                    self.assertNotIn('chunk1024',argv[argv.index('--prepared')+1])
                    self.assertIn('chunk1024',argv[argv.index('--root')+1])
                    self.assertIn('chunk1024',argv[argv.index('--cache-root')+1])
                    self.assertEqual(argv[argv.index('--answers-per-question')+1],'1' if server=='l20' else '2')
                    if server=='a800': self.assertEqual(argv[argv.index('--answer-seeds')+1:argv.index('--answer-seeds')+3],['0','1'])
