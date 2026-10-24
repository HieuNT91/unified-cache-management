"""CPU-only adoption, scheduling and launcher coverage for tree-prophetkv."""
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

import runner.cache  # Load before fixtures replace sys.modules temporarily.
from runner.layout import stamp_sample
from runner.setups import atomic_json, file_hash
from runner.output_budget import LONGBENCH_PROTOCOL, make_policy
from runner.thinking_budget import KEY
from scripts.collection_v3 import declaration
from scripts import tree_prophetkv_data as collection
from scripts import longbench_a800_control as control
from test_l20_traj4f import fixture as ruler_fixture, GPUS
from test_ruler_thinking import Tokenizer

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, dataset='ruler'):
    args = ruler_fixture(root, n=5)
    args.dataset = dataset
    args.percentages = [5, 20]  # The tiny synthetic chunks cannot fit S4 at 1%.
    args.answers_per_question = 1 if dataset == 'ruler' else 2
    args.root.mkdir()
    args.prepared.rename(args.root/'prepared'); args.prepared = args.root/'prepared'
    if dataset == 'longbench-v2':
        args.devices = ','.join(GPUS[:4])
        hashes = {n: file_hash(args.model/n) for n in ('config.json', 'tokenizer.json')}
        spec = dict(model_config_sha256=hashes['config.json'], tokenizer_files=hashes,
                    **declaration(dataset, 4, thinking=True))
        original = json.loads(next((args.prepared/'samples').glob('*.json')).read_text())
        manifest = []; files = {}
        for i in range(503):
            sample = copy.deepcopy(original)
            sample.pop('generator_config_sha256'); sample.pop('task'); sample.pop('execution_profile')
            sample.update(source_row=i, source_id=f'source-{i}', evaluation_protocol=LONGBENCH_PROTOCOL,
                          source_metadata=dict(domain='d', sub_domain='s', difficulty='easy', length='short'),
                          **{KEY: make_policy(Tokenizer())})
            stamp_sample(sample, 'native-template')
            name = f'samples/lb-{i}.json'; atomic_json(args.prepared/name, sample)
            files[name] = file_hash(args.prepared/name)
            manifest.append(dict(id=f'longbench-v2-{i}', prepared=name, subtask='d / s', references=['A'], scoring='longbench_v2'))
        (args.prepared/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
        files['manifest.jsonl'] = file_hash(args.prepared/'manifest.jsonl')
        atomic_json(args.prepared/'preparation.json', dict(spec=spec, files=files))
    return args


class TreeCollectionTests(unittest.TestCase):
    def test_local_generation_progress_reports_first_and_final_token_counts(self):
        from runner.generation import generate
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
                patch('runner.generation.time.perf_counter', return_value=1.):
            generate(Engine(), [65], 3, 'progress-test', progress=progress)
        values = [c.args[0] for c in progress.call_args_list]
        self.assertEqual([v['output_tokens'] for v in values], [1,3])
        self.assertEqual([v['finished'] for v in values], [False,True])

    def test_local_validation_is_explicit_tp4_subset_with_full_windows(self):
        from runner.tree_profiles import config
        for dataset in ('ruler','longbench-v2'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp),dataset)
                manifest=[json.loads(s) for s in (args.prepared/'manifest.jsonl').read_text().splitlines()]
                args.validation_prompt_ids=[r['id'] for r in manifest[:2]]
                with self.assertRaises(ValueError): self.adopt(args)
                args.local_validation=True
                with self.assertRaises(ValueError): self.adopt(args)
                args.tp=4; args.devices=','.join(GPUS[:4]); self.adopt(args)
                settings, plan, rows=collection.load(args.root)
                self.assertEqual(len(rows),2); self.assertEqual(plan['answers'],4*args.answers_per_question)
                self.assertEqual(settings['groups'],[GPUS[:4]])
                protocol,_=collection.stage(args.root,'tree')
                cfg=config(args.model,settings['execution_profile'],True,args.cache_root,
                    hardware=settings['hardware_profile'],tp=4)
                self.assertEqual(cfg['cpu_offload_gb'],4.)
                self.assertEqual(cfg['max_model_len'],82304 if dataset=='ruler' else 131072)
                with patch.object(control,'check_hardware',return_value={}) as check:
                    control.hardware(protocol); check.assert_called_once_with(protocol)
                args.validation_prompt_ids=['unknown']
                with self.assertRaises(ValueError): self.adopt(args)

    def adopt(self, args):
        with patch('scripts.router_control.code_hashes', return_value={'source': 'original'}):
            collection.adopt(args)

    def test_both_cohorts_readonly_adoption_shards_seeds_and_fast_resume(self):
        for dataset in ('ruler', 'longbench-v2'):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp), dataset)
                pins = {p: (file_hash(p), p.stat().st_mtime_ns) for p in args.prepared.rglob('*') if p.is_file()}
                self.adopt(args)
                settings, plan, rows = collection.load(args.root)
                self.assertEqual(plan['phase_order'], ['cached'])
                self.assertEqual((plan['probes'], plan['baseline_answers']), (0, 0))
                self.assertEqual(plan['answers'], len(rows)*2*args.answers_per_question)
                self.assertEqual(settings['answer_seeds'], list(range(args.answers_per_question)))
                self.assertEqual(len(settings['groups']), 5 if dataset == 'ruler' else 2)
                self.assertEqual([sum(r['ordinal'] % len(settings['groups']) == g for r in rows)
                                  for g in range(len(settings['groups']))], [13]*5 if dataset == 'ruler' else [252,251])
                protocol, _ = collection.stage(args.root, 'tree')
                from runner.tree_collection import inventory
                self.assertEqual(list(inventory(protocol)), ['tree-prophetkv-5', 'tree-prophetkv-20'])
                for stage in ('features', 'ruler', 'primary'):
                    with self.assertRaises(ValueError): collection.stage(args.root, stage)
                real_read = collection.read
                def metadata_only(path):
                    if Path(path).parent.name == 'samples': raise AssertionError('Resume rescanned inputs')
                    return real_read(path)
                with patch.object(collection, 'read', side_effect=metadata_only), \
                        patch('scripts.router_control.code_hashes', side_effect=AssertionError('Source guard')):
                    collection.adopt(args)
                self.assertEqual(pins, {p:(file_hash(p),p.stat().st_mtime_ns) for p in pins})
                result = collection.report(args.root)
                self.assertEqual(result['available_answers'], 0)
                with self.assertRaises(ValueError): collection.report(args.root, final=True)
                self.assertFalse((args.root/'final.json').exists())

    def test_rejects_old_outputs_modified_inputs_model_seeds_and_small_budget(self):
        for kind in ('result', 'input', 'model', 'seeds', 's4', 'cache', 'nested', 'missing'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp), 'longbench-v2' if kind == 'seeds' else 'ruler')
                if kind == 'result': atomic_json(args.root/'old-results.json', {})
                elif kind == 'input': next((args.prepared/'samples').glob('*.json')).write_text('{}')
                elif kind == 'model': (args.model/'tokenizer.json').write_text('{}')
                elif kind == 'seeds': args.answers_per_question = 5
                elif kind == 's4': args.percentages = [1]
                elif kind == 'cache': atomic_json(args.cache_root/'old.json', {})
                elif kind == 'nested': args.root = args.prepared/'result'
                else: (args.prepared/'preparation.json').unlink()
                with self.assertRaises(ValueError): self.adopt(args)
                self.assertFalse((args.root/'plan.json').exists())

    def test_worker_only_tree_answers_all_seeds_and_skips_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), 'longbench-v2'); self.adopt(args)
            protocol, rows = collection.stage(args.root, 'tree'); rows = rows[:1]
            engine = Mock(); engine.answer.return_value = ({}, []); engine.end.return_value = {}
            for finished in (False, True):
                engine.reset_mock()
                with patch('runner.corpus_runtime.Engine', return_value=engine) as ctor, \
                        patch('runner.setups.check_environment', return_value=GPUS[:2]), \
                        patch.object(control, 'accepted', return_value={} if finished else None), \
                        patch.object(control, 'publish') as publish:
                    control.worker(args.root, 'tree', 'cached', 'attempt', 0, stage_loader=lambda *_:(protocol, rows))
                if finished:
                    ctor.assert_not_called(); continue
                self.assertEqual(engine.answer.call_count, 4)
                self.assertEqual([c.kwargs['seed'] for c in engine.answer.call_args_list], [0,1]*2)
                self.assertTrue(all(c.args[0]['method'] == 'tree-prophetkv' for c in engine.answer.call_args_list))
                engine.probe.assert_not_called(); engine.begin.assert_called_once(); engine.end.assert_called_once()
                engine.close.assert_called_once(); self.assertEqual(publish.call_count, 4)

    def test_supervisor_only_cached_and_failure_never_completes(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp)); self.adopt(args)
            with patch.object(control, 'execute_phase', side_effect=RuntimeError('failed')), patch.object(collection, 'report') as report:
                with self.assertRaisesRegex(RuntimeError, 'failed'): collection.supervise(args.root)
                report.assert_not_called(); self.assertFalse((args.root/'tree/complete.json').exists())
            with patch.object(control, 'execute_phase') as execute, patch.object(collection, 'report'):
                collection.supervise(args.root)
                self.assertEqual([c.args[2] for c in execute.call_args_list], ['cached'])
                self.assertEqual(Path(execute.call_args.kwargs['entrypoint']).resolve(), Path(collection.__file__).resolve())

    def test_launchers_load_defaults_and_forward_existing_inputs_without_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'checkout with spaces'; (root/'scripts/launcher').mkdir(parents=True); (root/'config').mkdir()
            loader = ROOT/'scripts/launcher/server_env.sh'
            (root/'scripts/launcher/server_env.sh').write_text(loader.read_text())
            fake = root/'python'
            fake.write_text(f'#!{sys.executable}\nimport json,sys,os\nprint(json.dumps([sys.argv[1:],os.environ["CUDA_VISIBLE_DEVICES"]]))\n'); fake.chmod(0o755)
            for server, name, dataset, key, count in (
                ('l20', 'l20_ruler_thinking_tree_prophetkv.sh', 'ruler', 'GPU_DEVICES', 10),
                ('a800', 'a800_longbench_tree_prophetkv.sh', 'longbench-v2', 'GPU_PRIMARY', 4)):
                script = root/'scripts/launcher'/name; script.write_text((ROOT/'scripts/launcher'/name).read_text())
                envfile = root/'config'/f'.env.{server}.tree-prophetkv'
                envfile.write_text(f'PYTHON_BIN="{fake}"\nEXPERIMENT_DIR="/fresh result"\nMODEL_PATH="/model"\nCACHE_ROOT="/fresh cache"\n{key}="{",".join(GPUS[:count])}"\n')
                for command in ('configure', 'start', 'resume', 'status', 'stop', 'report'):
                    out = subprocess.run(['bash', str(script), command], cwd='/tmp', env=dict(PATH=os.environ['PATH']), capture_output=True, text=True)
                    self.assertEqual(out.returncode, 0, out.stderr)
                    argv, devices = json.loads(out.stdout)
                    self.assertEqual(devices, ''); self.assertEqual(argv[1], command)
                    self.assertEqual(argv[argv.index('--dataset')+1], dataset)
                    self.assertEqual(argv[argv.index('--prepared')+1], '/fresh result/prepared')
                    self.assertEqual(argv[argv.index('--devices')+1], ','.join(GPUS[:count]))
                    self.assertEqual(argv[argv.index('--answers-per-question')+1], '1' if server == 'l20' else '2')
                self.assertFalse(Path('/fresh result').exists())

    def test_tree_engine_config_keeps_tp2_memory_and_worker_method(self):
        from runner.tree_collection import configure_engine, actions, SCHEMA
        from runner.tree_prophetkv import PROTOCOL
        from runner.tree_profiles import config
        from runner.corpus_worker import tree_mode
        definitions = actions([5, 20, 40, 60])
        protocol = dict(schema=SCHEMA, tree_protocol=PROTOCOL, percentages=[5,20,40,60],
            actions=definitions, scheduled_actions=[a['id'] for a in definitions],
            phase_order=['cached'], answer_validation='native-answer-diagnostics-v1')
        from runner.naive_reuse import inventory
        self.assertEqual(list(inventory(protocol)), protocol['scheduled_actions'])
        for profile, hardware, window in (('ruler-thinking', 'l20-tp2', 82304), ('longbench-v2', 'server', 131072)):
            cfg = config('/model', profile, True, '/cache', hardware=hardware, tp=2)
            configure_engine(cfg, Path('/cache'), protocol)
            self.assertEqual((cfg['tensor_parallel_size'], cfg['gpu_memory_utilization'], cfg['max_model_len']), (2,.95,window))
            self.assertNotIn('num_gpu_blocks_override', cfg)
            extra = cfg['kv_transfer_config']['kv_connector_extra_config']
            self.assertEqual(extra['tree_f3'], dict(protocol=PROTOCOL,cache_dir='/cache',namespace='/cache',readonly=False))
            self.assertTrue(all(s['method'] == 'tree-prophetkv' for s in extra['ucm_sparse_config'].values()))
        sparse = NS(router_arrays=None, router_capture=False, model=NS(layers=[]), connector=NS(store=NS(operations={})))
        configure = Mock()
        modules = {'runner.worker': NS(configure=configure),
            'ucm.sparse.state': NS(get_ucm_sparse=lambda:sparse),
            'vllm.distributed': NS(get_tensor_model_parallel_rank=lambda:0,get_tp_group=lambda:NS(barrier=Mock())),
            'ucm.sparse.prophetkv.attention': NS(install=Mock())}
        with patch.dict(sys.modules, modules):
            tree_mode(None, 'request:read', definitions[0])
            configure.assert_called_once_with(None, 'tree-prophetkv', .05)
            with self.assertRaises(ValueError): tree_mode(None, 'request:read', definitions[0], capture=True)

    def test_f3_cache_readiness_immutability_retirement_and_warmup(self):
        from runner.tree_collection import TreePromptCache, retire_warmup
        from runner.sweep import PromptCache
        from runner.tree_prophetkv import write_f3, f3_path
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
            with patch('runner.corpus_runtime.validate'), patch('runner.tree_collection.TreePromptCache', return_value=cache) as ctor, \
                    patch('runner.generation.generate') as generate, patch('runner.corpus_runtime.retirement'), \
                    patch('runner.corpus_runtime.clean_capture'):
                engine.begin(row)
            ctor.assert_called_once(); cache.ready.assert_called_once(); cache.unchanged.assert_called_once()
            self.assertEqual(generate.call_count, 2)
            self.assertTrue(all(c.kwargs == {'phase':'populate'} for c in generate.call_args_list))

    def test_real_engine_constructor_enables_f3_and_retires_warmup(self):
        from runner.corpus_runtime import Engine
        from runner.tree_prophetkv import write_f3
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
            with patch.dict(sys.modules, {'runner.worker':NS(setup=Mock())}), \
                    patch('runner.corpus_runtime.start_engine', side_effect=start), \
                    patch('runner.corpus_runtime.validate_initialization'), \
                    patch('scripts.router_control.code_hashes', return_value={}), \
                    patch('runner.reporting.OutputAnalyzer'), \
                    patch('runner.generation.generate', side_effect=warmup) as generate, \
                    patch('runner.cache.wait_for_cache', return_value={'verified_shards':4}), \
                    patch('runner.corpus_runtime.retirement') as retired:
                engine = Engine(args.root/'tree', protocol, 0, 'cached', 'test', rows[:1])
            self.assertEqual(list((engine.cache/'tree-f3').iterdir()), [])
            generate.assert_called_once(); retired.assert_called_once_with(llm, 2)
            self.assertTrue(collection.read(engine.session/'initialization.json')['validated'])
            engine.close(); llm.llm_engine.engine_core.shutdown.assert_called_once()

    def test_custom_controller_detaches_and_spawns_exact_tp_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), 'longbench-v2'); self.adopt(args)
            def helper(*args, **kwargs):
                payload = json.loads(kwargs['input'])
                self.assertEqual(Path(payload['command'][3]).resolve(), Path(collection.__file__).resolve())
                self.assertEqual(payload['env']['CUDA_VISIBLE_DEVICES'], '')
                raise RuntimeError('inspected')
            with patch.object(control, 'idle'), patch.object(control.subprocess, 'run', side_effect=helper):
                with self.assertRaisesRegex(RuntimeError, 'inspected'):
                    control.detach(args.root, 'tree', stage_loader=collection.stage, entrypoint=collection.__file__)
            children = []
            def spawn(argv, **kwargs):
                self.assertEqual(Path(argv[2]).resolve(), Path(collection.__file__).resolve())
                self.assertEqual(argv[3], 'worker'); self.assertIn('cached', argv)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], ','.join(GPUS[len(children)*2:len(children)*2+2]))
                child = Mock(pid=10000+len(children),returncode=0); child.poll.return_value=0
                children.append(child); return child
            with patch.object(control, 'pending', side_effect=[['pending']]*3+[[]]), \
                    patch.object(control, 'engines_idle'), patch.object(control, 'hardware', return_value={}), \
                    patch.object(control, 'cleanup_caches'), patch.object(control, 'identity', return_value='synthetic'), \
                    patch.object(control, 'group_alive', return_value=False), patch.object(control.subprocess, 'Popen', side_effect=spawn):
                control.execute_phase(args.root, 'tree', 'cached', {}, run_loader=collection.load,
                    stage_loader=collection.stage, entrypoint=collection.__file__)
            self.assertEqual(len(children), 2)

    def test_seeded_records_are_accepted_and_reported_with_distinct_denominators(self):
        from runner.corpus_records import publish, accepted
        from runner.reporting import OutputAnalyzer
        from runner.layout import sample_provenance
        from runner.thinking_budget import BudgetState
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), 'longbench-v2'); self.adopt(args)
            protocol, rows = collection.stage(args.root, 'tree'); row = rows[0]
            root = args.root/'tree'; case = protocol['scheduled_actions'][0]
            atomic_json(root/'init.json', dict(validated=True,engine_config={}))
            policy = row[KEY]; tokens = [65,policy['close_id']]+policy['transition_ids']+[65,3]
            sample = collection.read(args.prepared/row['prepared'])
            for seed in (0,1):
                record = dict(**sample_provenance(sample), prompt_id=row['id'],sequence_id=row['id'],seed=seed,
                    method=case,executed_action=case,input_sha256=row['sha256'],group=0,gpu_uuids=GPUS[:2],
                    cache_immutable=True,initialization='init.json',initialization_sha256=file_hash(root/'init.json'),
                    answer_validation=protocol['answer_validation'],
                    retirement=[dict(rank=r,quiescent=True,transfers={'pending':0},request_bookkeeping=0) for r in range(2)],
                    accuracy=1.,subtask=row['subtask'],scoring='longbench_v2',prediction='The correct answer is (A)',
                    output_cap_reached=False,output_token_ids=tokens,max_output_tokens=16384,
                    timings=dict(ttft_seconds=1.,first_answer_content_seconds=1.1))
                record.update(OutputAnalyzer(Tokenizer()).analyze(tokens,[],True),
                    **BudgetState(policy).consume(tokens).metrics(Tokenizer()))
                publish(root,case,row,protocol,record,[],seed=seed)
                self.assertEqual(accepted(root,case,row,protocol,seed=seed), record)
            result = collection.report(args.root)
            self.assertEqual(result['available_answers'],2)
            self.assertEqual(result['methods'][case]['overall']['expected'],1006)
            self.assertEqual(result['methods'][case]['overall']['accuracy_percent'],100.)
            with self.assertRaises(ValueError): accepted(root,case,row,protocol,seed=2)


if __name__ == '__main__': unittest.main()
