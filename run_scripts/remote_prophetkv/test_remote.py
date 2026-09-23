"""CPU safety/protocol tests. No GPU work or historical scheduler imports."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import common
import suite
from cacheblend_ruler import verify_cache
from common import dump, load
from ruler_assets import materialize_hotpot


class ProtocolTests(unittest.TestCase):
    def test_private_connector_imports_on_python310_without_changing_install(self):
        from typing_extensions import Self
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / 'installed/ucm'
            relative = Path('integration/vllm/blend_connector.py')
            connector = installed / relative
            connector.parent.mkdir(parents=True)
            original = ('from typing import TYPE_CHECKING, List, Self, Tuple\n'
                        'class Chunk:\n'
                        '    def merge(self, other: Self) -> None: pass\n')
            connector.write_text(original)
            private = Path(directory) / 'private/ucm'
            suite.copy_private_ucm(installed, private)
            # Simulate Python 3.10 even when the test interpreter is newer.
            typing310 = SimpleNamespace(TYPE_CHECKING=False, List=list, Tuple=tuple)
            namespace = {}
            with patch.dict(sys.modules, {'typing': typing310}):
                exec(compile((private / relative).read_text(), str(relative), 'exec'), namespace)
            self.assertIs(namespace['Chunk'].merge.__annotations__['other'], Self)
            self.assertEqual(connector.read_text(), original)
            # An already-compatible installed copy is preserved byte-for-byte.
            fixed = (private / relative).read_bytes()
            connector.write_bytes(fixed)
            suite.copy_private_ucm(installed, private)
            self.assertEqual((private / relative).read_bytes(), fixed)

        # Exercise the checkout's actual typing imports without importing CUDA/vLLM.
        tree = ast.parse((common.REPO / 'ucm' / relative).read_text())
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module in ('typing', 'typing_extensions')]
        with patch.dict(sys.modules, {'typing': typing310}):
            namespace = {}
            exec(compile(ast.Module(body=imports, type_ignores=[]), str(relative), 'exec'), namespace)
        self.assertIs(namespace['Self'], Self)

    def test_vllm092_patches_resolve_uuid_and_release_nvml_on_failure(self):
        original = '''import os
class Platform:
    device_control_env_var = "CUDA_VISIBLE_DEVICES"
    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        if cls.device_control_env_var in os.environ and os.environ[
                cls.device_control_env_var] != "":
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_id = device_ids[device_id]
            return int(physical_device_id)
        else:
            return device_id

'''
        for filename in ('vllm-adapt.patch', 'vllm-adapt-sparse.patch'):
            with self.subTest(patch=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / 'vllm/platforms/interface.py'
                path.parent.mkdir(parents=True)
                path.write_text(original)
                patch_file = common.REPO / 'ucm/integration/vllm/patch/0.9.2' / filename
                subprocess.run(['git', 'apply', '--include=vllm/platforms/interface.py', str(patch_file)],
                               cwd=root, check=True, capture_output=True)
                namespace = runpy.run_path(str(path))
                resolve = namespace['Platform'].device_id_to_physical_device_id
                nvml = Mock()
                nvml.nvmlDeviceGetIndex.return_value = 7
                modules = {'vllm': SimpleNamespace(),
                           'vllm.utils': SimpleNamespace(import_pynvml=lambda: nvml)}
                with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(resolve(1), 1)
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    self.assertEqual(resolve(1), 1)
                    os.environ['CUDA_VISIBLE_DEVICES'] = '5,2'
                    self.assertEqual(resolve(1), 2)
                    nvml.nvmlInit.assert_not_called()
                    os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-first,GPU-second'
                    self.assertEqual(resolve(1), 7)
                    nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with('GPU-second')
                    nvml.nvmlInit.assert_called_once_with()
                    nvml.nvmlShutdown.assert_called_once_with()
                    nvml.reset_mock()
                    nvml.nvmlDeviceGetHandleByUUID.side_effect = RuntimeError('Unknown UUID')
                    with self.assertRaisesRegex(RuntimeError, 'Unknown UUID'):
                        resolve(0)
                    nvml.nvmlShutdown.assert_called_once_with()
                    self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-first,GPU-second')

    def test_packaged_hotpot_materializes_offline_atomically(self):
        rows = [dict(_id='x', question='Q?', answer='A', context=[['Doc', ['Text']]])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'hotpotqa.json.zip'
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr('hotpotqa.json', json.dumps(rows))
            destination = materialize_hotpot(root)
            self.assertEqual(load(destination), rows)
            self.assertFalse(list(root.glob('*.tmp')))
            # An existing validated JSON is accepted without rewriting it.
            before = destination.stat().st_mtime_ns
            self.assertEqual(materialize_hotpot(root), destination)
            self.assertEqual(destination.stat().st_mtime_ns, before)

    def test_packaged_hotpot_rejects_extra_or_nested_members(self):
        rows = [dict(_id='x', question='Q?', answer='A', context=[])]
        for members in ({'nested/hotpotqa.json': json.dumps(rows)},
                        {'hotpotqa.json': json.dumps(rows), 'extra': 'x'}):
            with self.subTest(members=list(members)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with zipfile.ZipFile(root / 'hotpotqa.json.zip', 'w') as bundle:
                    for name, value in members.items():
                        bundle.writestr(name, value)
                with self.assertRaisesRegex(ValueError, 'exactly one top-level'):
                    materialize_hotpot(root)
                self.assertFalse((root / 'hotpotqa.json').exists())

    def test_four_job_partition_has_no_missing_or_duplicate_units(self):
        units = common.scope_for_job('all')
        self.assertEqual(len(units), 16)
        self.assertEqual(len(set(units)), 16)
        self.assertEqual({task for length, task in units if length == 65536}, set(common.TASKS))
        self.assertEqual({(length, task) for length, task in units if length != 65536},
                         {(n, 'niah_multivalue') for n in (8192, 16384, 32768)})
        self.assertTrue(all(len(scope) == 4 for scope in common.JOBS.values()))

    def test_job_shell_commands_assign_disjoint_pairs_and_paths(self):
        scripts = Path(__file__).resolve().parents[1]
        roots, caches, devices = set(), set(), set()
        with tempfile.TemporaryDirectory(prefix='remote jobs ') as temporary:
            env = os.environ.copy()
            env.update(PYTHON_BIN=sys.executable, MODEL_PATH='/model with spaces',
                       RESULT_ROOT=temporary + '/results', CACHE_ROOT=temporary + '/cache', NUM_SAMPLES='100')
            for job in range(4):
                plan = json.loads(subprocess.check_output(['bash', str(scripts / f'job_{job}.sh'), 'plan'], env=env))
                cfg = plan['settings']
                self.assertEqual(plan['tensor_parallel_size'], 2)
                self.assertEqual(plan['requests'], 2800)
                self.assertEqual(cfg['indices'], [job * 2, job * 2 + 1])
                self.assertEqual(cfg['model'], '/model with spaces')
                self.assertFalse(devices.intersection(cfg['indices']))
                devices.update(cfg['indices'])
                roots.add(cfg['root'])
                caches.add(cfg['cache'])
            self.assertEqual(len(roots), 4)
            self.assertEqual(len(caches), 4)
            self.assertEqual(devices, set(range(8)))
            self.assertEqual(list(Path(temporary).iterdir()), [])  # plan is read-only

    def test_exactly_two_gpus_required(self):
        for value in ('0', '0,0', '0,1,2,3', '0,1,2,3,4,5,6,7'):
            with patch.dict(os.environ, {'GPU_INDICES': value}):
                with self.assertRaisesRegex(ValueError, 'exactly two'):
                    common.settings()

    def test_all_official_task_generators_and_query_spans(self):
        import yaml
        ruler = common.REPO / 'benchmarks/vendor/RULER'
        definitions = yaml.safe_load((ruler / 'scripts/synthetic.yaml').read_text())
        constants = runpy.run_path(str(ruler / 'scripts/data/synthetic/constants.py'))['TASKS']
        cfg = dict(ruler=str(ruler), model='/model with spaces', samples=100)
        for task in common.TASKS:
            with self.subTest(task=task):
                source = Path('/results') / '65536' / task / 'validation.jsonl'
                cmd = suite.generator_command(cfg, 65536, task, source, definitions, constants)
                self.assertTrue(Path(cmd[1]).is_file())
                self.assertEqual(cmd[cmd.index('--tokens_to_generate') + 1], '128')
                self.assertEqual(cmd[cmd.index('--num_samples') + 1], '100')
                self.assertEqual(cmd[cmd.index('--save_name') + 1], task)
                for key, value in definitions[task]['args'].items():
                    self.assertEqual(cmd[cmd.index('--' + key) + 1], str(value))
                base = constants[definitions[task]['task']]
                content = base['template'].format(context='haystack', query='my needle', type_needle_v='numbers')
                # NIAH generators place the answer prefix in input; others save it separately.
                if task.startswith('niah_'):
                    content += base['answer_prefix'].format(query='my needle', type_needle_v='numbers')
                begin, question = suite.query_span(content, task)
                self.assertEqual(content[begin:begin + len(question)], question)
                self.assertNotIn('haystack', question)
                self.assertNotIn(' The special magic', question)
                if task == 'fwe':
                    self.assertTrue(question.startswith('What are the three'))
                # Few-shot examples must not displace the final query.
                prefix = '\nQuestion: example\nWhat example?\n'
                b2, q2 = suite.query_span(prefix + content, task)
                self.assertEqual(q2, question)
                self.assertEqual(b2, begin + len(prefix))

    def test_official_qa_scoring_accepts_any_reference(self):
        from cacheblend_ruler import score_prediction
        self.assertEqual(score_prediction('qa_1', 'The answer is Paris.', ['Paris', 'City of Paris']), 1)
        self.assertEqual(score_prediction('qa_2', 'The answer is Paris.', ['Paris', 'City of Paris']), 1)
        self.assertEqual(score_prediction('niah_multivalue', '12\x0013', ['12', '13', '14', '15']), .5)

    def test_requested_scope_and_rope(self):
        self.assertEqual(common.LENGTHS, (8192, 16384, 32768, 65536))
        self.assertEqual(common.CASES, ('baseline', 'prophetkv-5', 'prophetkv-10',
                         'prophetkv-20', 'prophetkv-30', 'prophetkv-40', 'prophetkv-50'))
        self.assertEqual(100 * len(common.scope_for_job('all')) * len(common.CASES), 11200)
        self.assertIsNone(common.rope_for_length(32768))
        self.assertEqual(common.rope_for_length(65536)['factor'], 4)

    def test_worker_build_uses_tp_and_requested_cache_modes(self):
        import worker
        sample = dict(tokens=66048, context_target=65536, token_ids=[7] * 128,
                      boundaries=[0, 64, 128])
        p = dict(model='/model', tensor_parallel_size=2, gpu_memory_utilization=.90,
                 rope_scaling_64k=common.rope_for_length(65536))
        modules = {'vllm': SimpleNamespace(LLM=lambda **kwargs: kwargs),
                   'vllm.config': SimpleNamespace(KVTransferConfig=lambda **kwargs: kwargs)}
        with patch.dict(sys.modules, modules):
            for case in common.CASES:
                cfg = worker.build(SimpleNamespace(case=case, cache_dir=Path('/cache'), smoke=False, persistent=True), sample, p)
                self.assertEqual(cfg['tensor_parallel_size'], 2)
                self.assertFalse(cfg['enable_prefix_caching'])
                self.assertEqual(cfg['rope_scaling'], p['rope_scaling_64k'])
                self.assertGreaterEqual(cfg['max_model_len'], sample['tokens'] + 128)
                if case == 'baseline':
                    self.assertNotIn('kv_transfer_config', cfg)
                else:
                    self.assertEqual(cfg['kv_transfer_config']['kv_connector'], 'PersistentBlendConnector')
                    sparse = cfg['kv_transfer_config']['kv_connector_extra_config']['ucm_sparse_config']['ProphetKV']
                    self.assertEqual(sparse['ratio'], int(case.split('-')[-1]) / 100)
                    self.assertEqual(sparse['method'], 'prophetkv')

    def test_explicit_uuid_visibility_and_non_a800_rejected(self):
        listing = 'GPU 0: NVIDIA A800 80GB PCIe (UUID: GPU-aaa)\nGPU 1: NVIDIA A800 80GB PCIe (UUID: GPU-bbb)\n'
        raw = '0, GPU-aaa, NVIDIA A800 80GB PCIe, 81920\n1, GPU-bbb, NVIDIA A800 80GB PCIe, 81920\n'
        with patch('common.subprocess.check_output', side_effect=[listing, raw]):
            _, devices = common.selected_devices([1, 0])
        self.assertEqual([d['uuid'] for d in devices], ['GPU-bbb', 'GPU-aaa'])
        with patch('common.selected_devices', return_value=(listing, devices)), patch.dict(os.environ, {
                'REMOTE_GPU_DEVICES': json.dumps(devices), 'CUDA_VISIBLE_DEVICES': 'GPU-bbb,GPU-aaa'}):
            self.assertEqual(common.verify_gpu_visibility(), devices)
            os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
            with self.assertRaises(RuntimeError):
                common.verify_gpu_visibility()
        with patch('common.inventory', return_value=('', {0: dict(name='RTX 5090')})):
            with self.assertRaises(RuntimeError):
                common.selected_devices([0])

    def test_encode_non_thinking_and_query_excludes_answer_prefix(self):
        class Tokenizer:
            pad_token_id = 0
            is_fast = True

            def apply_chat_template(self, messages, **kwargs):
                assert kwargs['enable_thinking'] is False
                return '<user>' + messages[0]['content'] + '</user><think>\n\n</think>\n'

            def __call__(self, text, **kwargs):
                return dict(input_ids=[ord(c) for c in text], offset_mapping=[(i, i+1) for i in range(len(text))])

            def encode(self, text, **kwargs):
                return [ord(c) for c in text]

        row = dict(input='abc ' * 2000 + '\nWhat are the magic numbers? The special magic numbers are',
                   outputs=['1234', '4567'], index=99)
        sample = suite.encode(Tokenizer(), row, dict(chunk=4096), 8192, 0)
        self.assertEqual(sample['source_row'], 0)
        self.assertEqual(sample['source_index'], 99)
        self.assertEqual(sample['query']['text'], 'What are the magic numbers?')
        actual_query = ''.join(chr(sample['token_ids'][i]) for i in sample['query']['positions'])
        self.assertEqual(actual_query, sample['query']['text'])
        self.assertGreaterEqual(sample['query']['positions'][0], sample['boundaries'][-2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.json'
            dump(path, sample)
            self.assertEqual(suite.load_sample(path), sample)

    def test_cache_requires_every_tp_shard_and_correct_size(self):
        import hashlib
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model = base / 'model'
            model.mkdir()
            dump(model / 'config.json', dict(num_key_value_heads=8, num_attention_heads=64,
                 hidden_size=5120, head_dim=128, num_hidden_layers=64))
            cache = base / 'cache'
            tokens = list(range(64))
            def hashed(meta, value):
                value = value if isinstance(value, bytes) else pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                return hashlib.md5(meta + value).digest()
            metas = [f'{model}:8:torch.bfloat16:{rank}'.encode() for rank in range(8)]
            key = hashed(metas[0], (hashed(metas[0], 'UCM_HASH_SEED'), tuple(tokens)))
            paths = []
            for rank, meta in enumerate(metas):
                name = (key if rank == 0 else hashed(meta, key)).hex()
                path = cache / 'kv' / name[:8] / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('wb') as stream:
                    stream.truncate(64 * 128 * 4 * 64)
                paths.append(path)
            args = SimpleNamespace(model_path=model, tensor_parallel_size=8, cache_dir=cache)
            self.assertEqual(verify_cache(args, [tokens])['verified_shards'], 8)
            paths[-1].write_bytes(b'bad')
            with self.assertRaisesRegex(RuntimeError, 'Cache incomplete'):
                verify_cache(args, [tokens])

    def test_refuse_cleanup_of_live_or_unowned_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache/owned-sample-cache'
            cache.mkdir(parents=True)
            cfg = dict(cache=str(cache.parent))
            with self.assertRaisesRegex(RuntimeError, 'unrecognized cache'):
                suite.cleanup(cfg, root)
            dump(cache / 'owner.json', dict(result_root=str(root)))
            dump(root / 'active.json', dict(pgid=123))
            with patch('suite.group_alive', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'still alive'):
                    suite.cleanup(cfg, root)
            self.assertTrue(cache.exists())
            with patch('suite.group_alive', return_value=False):
                suite.cleanup(cfg, root)
            self.assertFalse(cache.exists())

    def test_lock_rejects_duplicate_supervisors(self):
        with tempfile.TemporaryDirectory() as directory:
            with suite.locked(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, 'holds run.lock'):
                    with suite.locked(Path(directory)):
                        self.fail('Second lock acquired')

    def test_pid_reuse_does_not_authorize_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dump(root / 'supervisor.json', dict(pid=123, identity={'start': 'old'}))
            with patch('suite.identity', return_value={'start': 'new'}):
                self.assertIsNone(suite.live_supervisor(root))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.devices = [dict(uuid=f'GPU-{i}', index=i) for i in range(2)]
        self.p = dict(num_layers=2, tensor_parallel_size=2, gpu_devices=self.devices)
        dump(self.root / 'protocol.json', self.p)
        ids = [8] * 63 + [0] + [9] * 63 + [0] + [10] * 256
        self.sample = dict(id='sample', dataset='ruler', label='niah_multivalue', token_ids=ids,
                           boundaries=[0, 64, 128, 384], tokens=len(ids),
                           fresh_suffix_tokens=256, context_target=8192, query=dict(positions=[370]), source_metadata=dict(references=['123']))
        self.input_path = self.root / 'input.json'
        dump(self.input_path, self.sample)
        self.meta = self.sample | dict(input_path=str(self.input_path))
        self.path = self.root / 'prophetkv-50.json'
        self.imports = [dict(rank=i, visible_uuid=d['uuid'], ucm_path=str(self.root / 'private_ucm/ucm/__init__.py'))
                        for i, d in enumerate(self.devices)]
        selection = dict(kind='prophetkv_selection', request_id='measured-1-0', eligible_count=64,
                         selected_count=32, probe_layers=2, alignment_count=2, question_positions=[370],
                         scores=[1.] * 64, selected_positions=list(range(64, 96)))
        layers = [dict(kind='layer_counts', layer=f'model.layers.{i}.self_attn.attn',
                       projection_tokens=288, attention_tokens=288, ffn_tokens=288) for i in range(2)]
        self.diag = [w | dict(diagnostics=copy.deepcopy([selection, *layers])) for w in self.imports]
        self.record = dict(sample_id='sample', case='prophetkv-50', label='niah_multivalue', context_target=8192,
                           prompt_tokens=384, max_output_tokens=128, smoke=False, timing_source=common.TIMING,
                           cache_unchanged=True, online_mask_reused=False, gpu_devices=self.devices,
                           tensor_parallel_size=2, ttft_seconds=1., generation_seconds=2., output_tokens=1,
                           output_token_ids=[100], references=['123'], score=1., prediction='123', finish_reason='stop',
                           worker_imports=self.imports, cache_verification=dict(complete=True, verified_shards=4,
                                                                            expected_unique_blocks=2))
        self.path.with_suffix('.log').write_text('MEASURE_BEGIN\nrequest_id: measured-1-0, '
            'req_stage: BlendStage.CACHE_BLEND, first chunk prefix hit: 1, chunks cache total hit: 1\n'
            'MEASURE_END\nWORKER_COMPLETE\n')
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        dp = self.path.with_suffix('.diagnostics.json')
        dump(dp, self.diag)
        dump(self.path, self.record)

    def test_resume_revalidates_results_without_checksums(self):
        suite.accept(self.root, self.p, self.meta, 'prophetkv-50', self.path)
        self.assertTrue(suite.valid(self.root, self.p, self.meta, 'prophetkv-50', self.path))
        self.path.with_suffix('.log').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Worker failed'):
            suite.valid(self.root, self.p, self.meta, 'prophetkv-50', self.path)

    def test_legacy_checksums_ignored_without_rewriting_artifacts(self):
        self.record.update(protocol_sha256='old', prompt_sha256='old', input_sha256='old', diagnostics_sha256='old')
        self.sample['prompt_sha256'] = 'old'
        dump(self.input_path, self.sample)
        self.save()
        dump(self.path.with_suffix('.validated.json'), dict(record_sha256='old', log_sha256='old', diagnostics_sha256='old'))
        before = self.path.read_bytes()
        with patch('hashlib.sha256', side_effect=AssertionError('Checksum must not run')):
            self.assertTrue(suite.valid(self.root, self.p, self.meta, 'prophetkv-50', self.path))
        self.assertEqual(self.path.read_bytes(), before)

    def test_prepare_and_resume_do_not_read_checkpoint_or_hash_files(self):
        (self.root / 'protocol.json').unlink()
        installed = self.root / 'installed/ucm'
        connector = installed / 'integration/vllm/blend_connector.py'
        connector.parent.mkdir(parents=True)
        connector.write_text('from typing import TYPE_CHECKING, List, Self, Tuple\n')
        (installed / 'sparse/blend').mkdir(parents=True)
        (installed / 'sparse/factory.py').write_text('# installed factory\n')
        cfg = dict(scope=[dict(length=8192, task='niah_multivalue')], samples=1,
                   ruler=str(common.REPO / 'benchmarks/vendor/RULER'),
                   model='/checkpoint-must-not-be-read', indices=[0, 1], memory=.9)
        source = self.root / 'datasets/8192/niah_multivalue/validation.jsonl'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps(dict(input='already generated', outputs=['123'])) + '\n')
        runtime = {'uc-manager': '0.3.0'}
        with patch('suite.preflight', return_value=(runtime, installed, Path('/runtime-not-scanned'), self.devices, None)), \
                patch('suite.encode', return_value=self.sample), \
                patch('hashlib.sha256', side_effect=AssertionError('Checksum must not run')):
            p = suite.prepare(cfg, self.root)
        self.assertFalse(p['artifact_checksums'])
        self.assertNotIn('model_hashes', p)
        self.assertNotIn('source_hashes', p)
        # Existing protocols can contain stale hashes, even inaccessible paths.
        p.update(model_hashes={'/missing/weights.safetensors': 'old'},
                 source_hashes={'old.py': 'old'}, private_ucm_hashes={'old.py': 'old'},
                 runtime_hashes={'/missing/runtime.py': 'old'}, dataset_hashes={'/missing/data': 'old'})
        dump(self.root / 'protocol.json', p)
        before = (self.root / 'protocol.json').read_bytes()
        with patch('suite.version', return_value='0.3.0'), \
                patch('suite.selected_devices', return_value=('', self.devices)), \
                patch('hashlib.sha256', side_effect=AssertionError('Checksum must not run')):
            self.assertEqual(suite.verify(cfg, self.root), p)
            with self.assertRaisesRegex(ValueError, 'Configuration changed'):
                suite.verify(cfg | dict(samples=100), self.root)
        self.assertEqual((self.root / 'protocol.json').read_bytes(), before)

    def test_refresh_repairs_prepared_job_and_archives_smoke(self):
        cfg = dict(samples=100)
        p = self.p | dict(settings=cfg, samples=[self.meta], measured_requests=7,
                         scope=[dict(length=8192, task='niah_multivalue')])
        dump(self.root / 'protocol.json', p)
        connector = self.root / 'private_ucm/ucm/integration/vllm/blend_connector.py'
        connector.parent.mkdir(parents=True)
        connector.write_text('from typing import TYPE_CHECKING, List, Self, Tuple\n')
        runtime = self.root / 'private_ucm/ucm/sparse/prophetkv/runtime.py'
        runtime.parent.mkdir(parents=True)
        runtime.write_text('# old runtime\n')
        dump(self.root / 'smoke/validation.json', dict(complete=False))
        before = (self.root / 'protocol.json').read_bytes()
        input_before = self.input_path.read_bytes()
        with suite.locked(self.root):
            suite.refresh_runtime(cfg, self.root)
        self.assertIn('from typing_extensions import Self', connector.read_text())
        self.assertEqual(runtime.read_text(), (common.HERE / 'method/prophetkv/runtime.py').read_text())
        self.assertEqual((self.root / 'protocol.json').read_bytes(), before)
        self.assertEqual(self.input_path.read_bytes(), input_before)
        self.assertFalse((self.root / 'smoke').exists())
        archives = list((self.root / 'runtime-refresh').iterdir())
        self.assertEqual(len(archives), 1)
        self.assertTrue((archives[0] / 'smoke/validation.json').exists())
        suite.refresh_runtime(cfg, self.root)
        self.assertEqual(list((self.root / 'runtime-refresh').iterdir()), archives)

    def test_refresh_refuses_live_engines_or_validated_measurements(self):
        with patch('suite.check_orphan', side_effect=RuntimeError('live engine')):
            with self.assertRaisesRegex(RuntimeError, 'live engine'):
                suite.refresh_runtime({}, self.root)
        cfg = dict(samples=100)
        dump(self.root / 'protocol.json', self.p | dict(settings=cfg))
        private = self.root / 'private_ucm/ucm'
        connector = private / 'integration/vllm/blend_connector.py'
        connector.parent.mkdir(parents=True)
        connector.write_text('from typing import TYPE_CHECKING, List, Self, Tuple\n')
        runtime = private / 'sparse/prophetkv/runtime.py'
        runtime.parent.mkdir(parents=True)
        runtime.write_text('# old runtime\n')
        dump(self.root / 'records/sample/baseline.validated.json', dict(complete=True))
        with self.assertRaisesRegex(RuntimeError, 'validated measurements'):
            suite.refresh_runtime(cfg, self.root)
        self.assertEqual(runtime.read_text(), '# old runtime\n')

    def test_legacy_smoke_gate_requires_persistent_requalification(self):
        p = self.p | dict(samples=[self.meta])
        gate = self.root / 'smoke/8192/niah_multivalue'
        artifact = gate / 'reference.json'
        dump(artifact, dict(control='native_dense_exact_prefix'))
        dump(gate / 'validation.json', dict(complete=True, sample_id=self.meta['id'],
             protocol_sha256='old', artifacts={str(artifact): 'old'}))
        with patch('hashlib.sha256', side_effect=AssertionError('Checksum must not run')):
            self.assertFalse(suite.valid_gate(self.root, p, 8192, 'niah_multivalue'))
            receipt = load(gate / 'validation.json')
            dump(gate / 'validation.json', receipt | dict(engine_policy=common.ENGINE_POLICY))
            self.assertTrue(suite.valid_gate(self.root, p, 8192, 'niah_multivalue'))
            artifact.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing gate artifact'):
                suite.valid_gate(self.root, p, 8192, 'niah_multivalue')

    def test_reject_tp_disagreement(self):
        self.diag[1]['diagnostics'][0]['scores'][0] = 2.
        self.save()
        with self.assertRaisesRegex(ValueError, 'TP ranks disagree'):
            suite.validate(self.root, self.p, self.meta, 'prophetkv-50', self.path)

    def test_reject_incomplete_cache_shards(self):
        self.record['cache_verification']['verified_shards'] = 3
        self.save()
        with self.assertRaisesRegex(ValueError, 'Incomplete cache shards'):
            suite.validate(self.root, self.p, self.meta, 'prophetkv-50', self.path)

    def test_reject_missing_measured_cache_hits(self):
        self.path.with_suffix('.log').write_text('MEASURE_BEGIN\nMEASURE_END\nWORKER_COMPLETE\n')
        with self.assertRaisesRegex(ValueError, 'cache-hit evidence'):
            suite.validate(self.root, self.p, self.meta, 'prophetkv-50', self.path)

    def test_reject_total_generation_as_ttft(self):
        self.record['ttft_seconds'] = 3.
        self.save()
        with self.assertRaisesRegex(ValueError, 'timing'):
            suite.validate(self.root, self.p, self.meta, 'prophetkv-50', self.path)

    def test_partial_report_uses_paired_mean_speedup(self):
        import csv
        p = self.p | dict(samples=[dict(id='a', context_target=8192, label='niah_multivalue'), dict(id='b', context_target=8192, label='niah_multivalue')],
                         measured_requests=14, samples_per_task_length=2, scope=[dict(length=8192, task='niah_multivalue')])
        for name, baseline, method in [('a', 2., 1.), ('b', 8., 2.)]:
            for case, ttft in [('baseline', baseline), ('prophetkv-50', method)]:
                dump(self.root / 'records' / name / f'{case}.json', dict(sample_id=name, case=case,
                     context_target=8192, label='niah_multivalue', ttft_seconds=ttft, generation_seconds=ttft+1, prompt_tokens=8000,
                     output_tokens=128, score=.5, finish_reason='length'))
        with patch('suite.valid', side_effect=lambda root, p, meta, case, path: path.exists()):
            suite.report(self.root, p)
        with (self.root / 'partial/comparison.csv').open() as stream:
            table = list(csv.DictReader(stream))
        row = next(r for r in table if r['method'] == 'prophetkv-50' and r['context_target'] == '8192')
        self.assertAlmostEqual(float(row['ttft_speedup']), 10 / 3)
        self.assertEqual(int(row['paired_samples']), 2)
        self.assertEqual(float(row['accuracy_percent']), 50.)
        self.assertEqual(int(row['length_limited']), 2)
        self.assertFalse((self.root / 'final').exists())


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        sys.path.insert(0, str(Path(__file__).parent / 'method'))

    def test_setup_registers_missing_rope_and_normalizes_only_once(self):
        from prophetkv.runtime import setup
        torch = self.torch
        table = torch.tensor([[1.25, 1.25, 0., 0.], [.75, 1., 1., .75]])
        original = table.clone()
        model = type('Qwen3Model', (), {})()
        model.layers = [SimpleNamespace(self_attn=SimpleNamespace(
            rotary_emb=SimpleNamespace(cos_sin_cache=table)))] * 64
        wrapper = SimpleNamespace(model=model)
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=wrapper))
        for preinitialized in (False, True):
            for wrapped in (False, True):
                with self.subTest(preinitialized=preinitialized, wrapped=wrapped):
                    connector = SimpleNamespace(cos_sin_cache=table if preinitialized else None,
                        bind_connector_metadata=Mock(), wait_for_layer_load=Mock())
                    connector.setup_model = Mock(side_effect=lambda m: setattr(
                        connector, 'cos_sin_cache', m.model.layers[0].self_attn.rotary_emb.cos_sin_cache))
                    sparse = SimpleNamespace()
                    group = SimpleNamespace(connector=connector) if wrapped else connector
                    modules = {'ucm.sparse.state': SimpleNamespace(
                        has_ucm_sparse=lambda: True, get_ucm_sparse=lambda: sparse),
                        'vllm.distributed.kv_transfer': SimpleNamespace(get_kv_transfer_group=lambda: group)}
                    with patch.dict(sys.modules, modules):
                        self.assertEqual(setup(worker)['layers'], 64)
                        normalized = connector.cos_sin_cache
                        bound = connector.bind_connector_metadata
                        setup(worker)
                    self.assertIs(connector.cos_sin_cache, normalized)
                    self.assertIs(connector.bind_connector_metadata, bound)
                    torch.testing.assert_close(normalized, table / 1.25)
                    torch.testing.assert_close(table, original)
                    if preinitialized:
                        connector.setup_model.assert_not_called()
                    else:
                        connector.setup_model.assert_called_once_with(wrapper)

    def test_delta_rope_rejects_uninitialized_table(self):
        from prophetkv.runtime import delta_rotation_table
        for table in (None, self.torch.empty(0, 4), self.torch.ones(3), self.torch.ones(2, 3)):
            with self.subTest(table=table), self.assertRaisesRegex(RuntimeError, 'initialized'):
                delta_rotation_table(table)

    def test_tp_head_means_equal_global_head_mean(self):
        from prophetkv.selection import context_importance
        torch = self.torch
        torch.manual_seed(4)
        q, k = torch.randn(7, 64, 16), torch.randn(131, 8, 16)
        global_scores = context_importance(q, k)
        shards = [context_importance(q[:, i*8:(i+1)*8], k[:, i:i+1]) for i in range(8)]
        torch.testing.assert_close(torch.stack(shards).mean(0), global_scores, atol=1e-7, rtol=1e-5)

    def test_global_selection_reduces_before_ranking(self):
        from prophetkv.runtime import global_selection
        torch = self.torch
        group = SimpleNamespace(world_size=2, broadcast=lambda x, src: x)
        module = SimpleNamespace(get_tp_group=lambda: group,
                tensor_model_parallel_all_reduce=lambda x: x + torch.tensor([0., 0., 10., 0.]))
        with patch.dict(sys.modules, {'vllm.distributed': module}):
            scores, chosen = global_selection(torch.tensor([99., 4., 0., 2.]), 1, .5)
        torch.testing.assert_close(scores, torch.tensor([2., 5., 1.]))
        self.assertEqual(chosen.tolist(), [2])

    def test_yarn_delta_preserves_single_amplitude(self):
        from prophetkv.runtime import delta_rotation_table
        torch = self.torch
        amplitude = 1.138629436
        angles = torch.arange(30).float()[:, None] * torch.tensor([.2, .3])[None]
        scaled = torch.cat((angles.cos(), angles.sin()), -1) * amplitude
        table = delta_rotation_table(scaled)
        torch.testing.assert_close(table, scaled / amplitude)
        def rotate(x, row):
            a, b = x.chunk(2)
            c, s = row.chunk(2)
            return torch.cat((a*c-b*s, b*c+a*s))
        original = torch.tensor([1., 2., 3., 4.])
        # Key already has model Q/K scaling; applying unnormalized delta would square it.
        shifted = rotate(rotate(original, scaled[4]), table[7])
        torch.testing.assert_close(shifted, rotate(original, scaled[11]), atol=1e-6, rtol=1e-6)

    def test_causal_positions_and_future_kv_invariance(self):
        from prophetkv.attention import dense_reference
        torch = self.torch
        torch.manual_seed(7)
        q, k = torch.randn(3, 8, 16), torch.randn(40, 1, 16)
        v = torch.randn_like(k)
        pos = torch.tensor([0, 17, 39])
        a = dense_reference(q, k, v, pos)
        k[18:] *= 100
        v[18:] += 100
        b = dense_reference(q, k, v, pos)
        torch.testing.assert_close(a[:2], b[:2], atol=0, rtol=0)

    def test_all_six_budgets_and_stable_ties(self):
        from prophetkv.selection import select, request_mask, RequestState, RequestMetadata
        torch = self.torch
        for ratio in (.05, .1, .2, .3, .4, .5):
            self.assertEqual(select(torch.ones(64), 64, ratio).tolist(), list(range(64, 64+int(64*ratio))))
        positions = torch.arange(64, 256)
        mask = request_mask(positions, torch.tensor([64]), 64, 192, torch.tensor([1, 0]))
        self.assertEqual(positions[mask].tolist(), [64] + list(range(128, 256)))
        state = RequestState()
        meta = RequestMetadata('prime', (0, 64, 128, 384), (370,))
        state.arm(meta)
        state.take('prime')
        with self.assertRaises(RuntimeError):
            state.take('measured')



class PersistentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ValidationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root

    def test_schedule_groups_methods_and_skips_only_validated_prompts(self):
        samples = [dict(id=str(i), context_target=length) for i, length in enumerate((8192, 32768, 65536, 65536))]
        with patch('suite.valid', side_effect=lambda root, p, m, case, path: m['id']=='0' and case=='baseline'):
            plan = list(suite.session_plan(self.root, {}, samples))
        self.assertEqual(len(plan), 14)
        self.assertEqual([e['id'] for e in plan[0][2]], ['1'])
        self.assertEqual([e['id'] for e in plan[1][2]], ['0', '1'])
        self.assertEqual([e['id'] for e in plan[7][2]], ['2', '3'])
        with patch('suite.valid', return_value=False):
            self.assertEqual(len(list(suite.session_plan(self.root, {}, samples, 'prophetkv-10'))), 2)
        p = dict(samples=[m | dict(tokens=n) for m, n in zip(samples, (8000,33000,66000,67000))])
        self.assertEqual(common.model_limit(p, samples[0]), common.model_limit(p, samples[1]))
        self.assertGreaterEqual(common.model_limit(p, samples[0]), 33128)
        self.assertGreaterEqual(common.model_limit(p, samples[2]), 67128)
        self.assertNotEqual(common.model_limit(p, samples[0]), common.model_limit(p, samples[2]))

    def test_namespaced_cache_verifier_requires_both_tp_shards(self):
        import hashlib
        import pickle
        from lifecycle import seed_value
        model = self.root / 'model'
        dump(model / 'config.json', dict(num_key_value_heads=8, num_attention_heads=64,
             hidden_size=5120, head_dim=128, num_hidden_layers=64))
        cache = self.root / 'cache'
        tokens = list(range(64))
        seed = seed_value('a'*32)
        def hashed(meta, value):
            value = value if isinstance(value, bytes) else pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(meta + value).digest()
        metas = [f'{model}:2:torch.bfloat16:{rank}'.encode() for rank in range(2)]
        key = hashed(metas[0], (hashed(metas[0], seed), tuple(tokens)))
        for rank in range(2):
            name = (key if rank == 0 else hashed(metas[rank], key)).hex()
            path = cache / 'kv' / name[:8] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as stream:
                stream.truncate(64 * 4 * 128 * 4 * 64)
        args = SimpleNamespace(model_path=model, tensor_parallel_size=2, cache_dir=cache, hash_seed=seed)
        self.assertEqual(verify_cache(args, [tokens])['verified_shards'], 2)
        args.hash_seed = seed_value('b'*32)
        with self.assertRaisesRegex(RuntimeError, 'Cache incomplete'):
            verify_cache(args, [tokens])
        args.hash_seed = seed
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Cache incomplete'):
            verify_cache(args, [tokens])

    def test_transfer_failure_retains_pending_handle(self):
        from lifecycle import TrackedStore
        store = Mock()
        store.dump_data.return_value = object()
        tracked = TrackedStore(store)
        tracked.dump_data('data')
        store.wait.side_effect = RuntimeError('not complete')
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            tracked.drain()
        self.assertEqual(len(tracked.pending), 1)
        store.wait.side_effect = None
        self.assertEqual(tracked.drain(), dict(pending=0, completed=1))

    def test_retirement_deletes_only_known_block_files(self):
        from lifecycle import delete_retired_files
        from prophetkv_common import cache_snapshot
        cache = self.root / 'cache'
        block = cache / 'kv' / ('a'*8) / ('a'*32)
        block.parent.mkdir(parents=True)
        block.write_bytes(b'kv')
        dump(cache / 'owner.json', dict(result_root=str(self.root)))
        snapshot = cache_snapshot(cache)
        unexpected = block.parent / 'unexpected'
        unexpected.write_bytes(b'do not delete')
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            delete_retired_files(cache, snapshot)
        self.assertTrue(block.exists())
        unexpected.unlink()
        delete_retired_files(cache, snapshot)
        self.assertFalse(block.exists())
        self.assertTrue((cache / 'owner.json').exists())
        block.parent.mkdir(parents=True)
        block.symlink_to(cache / 'owner.json')
        with self.assertRaisesRegex(ValueError, 'symlinked'):
            delete_retired_files(cache, cache_snapshot(cache))

    def test_tp_barrier_rejects_live_missing_rank_pending_and_stale_scheduler(self):
        import persistent_worker as worker
        llm = Mock()
        llm.llm_engine.has_unfinished_requests.return_value = True
        with self.assertRaisesRegex(RuntimeError, 'live'):
            worker.barrier(llm, 'request', False, 2)
        llm.llm_engine.has_unfinished_requests.return_value = False
        ranks = [dict(rank=i, quiescent=True, transfers=dict(pending=0), request_bookkeeping=0) for i in range(2)]
        for bad in (ranks[:1], [ranks[0], ranks[0]], [ranks[0], ranks[1] | dict(transfers=dict(pending=1))]):
            llm.collective_rpc.return_value = bad
            with self.assertRaisesRegex(RuntimeError, 'Every TP rank'):
                worker.barrier(llm, 'request', False, 2)
        llm.collective_rpc.return_value = ranks
        receipt = self.root / 'scheduler.json'
        dump(receipt, dict(request_id='old', requests_blend_meta=0, requests_meta=0))
        with patch.dict(os.environ, PROPHETKV_SCHEDULER_RECEIPT=str(receipt)):
            with self.assertRaisesRegex(RuntimeError, 'Scheduler'):
                worker.barrier(llm, 'new', True, 2)
            self.assertEqual(worker.barrier(llm, 'old', True, 2)['workers'], ranks)

    def persistent_record(self):
        f = self.fixture
        ns = 'a'*32
        rid = ns + ':measured'
        r = f.record
        r.update(engine_policy=common.ENGINE_POLICY, request_id=rid, namespace=ns, session_id='session',
                 max_model_len=common.model_limit(f.p, f.meta), rope_group='native',
                 cache_files_after_retirement=0,
                 retirement=dict(workers=[dict(rank=i, quiescent=True, transfers=dict(pending=0), request_bookkeeping=0)
                                          for i in range(2)],
                                 scheduler=dict(request_id=rid, requests_blend_meta=0, requests_meta=0)))
        r['cache_verification']['namespace'] = ns
        for w in f.diag:
            w['diagnostics'][0]['request_id'] = rid
        text = f.path.with_suffix('.log').read_text().replace('measured-1-0', rid)
        f.path.with_suffix('.log').write_text(text)
        f.save()
        return f, rid, text

    def test_validation_uses_real_namespaced_request_and_both_tp_retirements(self):
        f, rid, text = self.persistent_record()
        suite.validate(f.root, f.p, f.meta, 'prophetkv-50', f.path)
        f.record['retirement']['workers'][1]['transfers']['pending'] = 1
        f.save()
        with self.assertRaisesRegex(ValueError, 'retirement'):
            suite.validate(f.root, f.p, f.meta, 'prophetkv-50', f.path)

    def test_completed_request_survives_later_session_failure(self):
        f, rid, text = self.persistent_record()
        pending = [f.meta | dict(output=str(f.path)), dict(id='next', output=str(f.root/'missing.json'))]
        log = 'REQUEST_BEGIN '+rid+'\n'+text.replace('WORKER_COMPLETE\n','')
        log += 'REQUEST_COMPLETE '+rid+' '+str(f.path)+'\nTraceback: later request failed\n'
        with patch('suite.progress'):
            remainder = suite.collect_completed(f.root, f.p, 'prophetkv-50', pending, log, False)
        self.assertEqual(len(pending), 1)
        self.assertIn('Traceback', remainder)
        self.assertNotIn('Traceback', f.path.with_suffix('.log').read_text())
        self.assertTrue(suite.valid(f.root, f.p, f.meta, 'prophetkv-50', f.path))
        pending = [f.meta | dict(output=str(f.path))]
        with self.assertRaisesRegex(RuntimeError, 'Invalid request'):
            suite.collect_completed(f.root, f.p, 'prophetkv-50', pending, 'Traceback\n'+log, False)

    def test_cached_worker_reuses_engine_and_retires_before_next_prompt(self):
        import persistent_worker as worker
        from contextlib import ExitStack, redirect_stdout
        import io
        f = self.fixture
        samples = {}
        entries = []
        for i in range(3):
            sample = f.sample | dict(id=f'sample-{i}', chunk_size=64, source_row=i)
            input_path = f.root / f'input-{i}.json'
            dump(input_path, sample)
            samples[str(input_path)] = sample
            entries.append(sample | dict(input_path=str(input_path), output=str(f.root/f'out-{i}.json')))
        p = f.p | dict(samples=entries, model='/model')
        args = SimpleNamespace(case='prophetkv-10', smoke=False, cache_dir=f.root/'cache', session=f.root/'session.json')
        events = []
        llm = Mock()
        llm.collective_rpc.return_value = f.imports
        result = SimpleNamespace(num_cached_tokens=64, outputs=[SimpleNamespace(text='123', token_ids=[1], finish_reason='stop')])
        def generate(engine, ids, budget, rid):
            events.append(('generate', rid, budget))
            return result, .1, .2
        def barrier(*args):
            events.append(('barrier', args[1]))
            return dict(workers=[], scheduler={})
        with ExitStack() as stack:
            build = stack.enter_context(patch.object(worker, 'build', return_value=llm))
            stack.enter_context(patch.object(worker, 'generate', side_effect=generate))
            stack.enter_context(patch.object(worker, 'barrier', side_effect=barrier))
            verify = stack.enter_context(patch.object(worker, 'wait_for_cache', side_effect=lambda *a: dict(complete=True)))
            stack.enter_context(patch.object(worker, 'delete_retired_files', side_effect=lambda *a: events.append(('delete',))))
            stack.enter_context(redirect_stdout(io.StringIO()))
            worker.run_session(args, p, entries)
        build.assert_called_once()
        llm.llm_engine.engine_core.shutdown.assert_called_once()
        records = [load(e['output']) for e in entries]
        self.assertEqual(len({r['namespace'] for r in records}), 3)
        self.assertEqual([r['session_request_index'] for r in records], [0,1,2])
        self.assertEqual(load(args.session)['engine_initializations'], 1)
        self.assertEqual(verify.call_count, 3)
        for call, record in zip(verify.call_args_list, records):
            self.assertEqual(call.args[0].tensor_parallel_size, 2)
            self.assertEqual(call.args[0].hash_seed[-1], record['namespace'])
        for i, event in enumerate(events):
            if event[0]=='delete':
                self.assertEqual(events[i-1][0], 'barrier')
        for record in records:
            generated = [e for e in events if e[0]=='generate' and e[1].startswith(record['namespace'])]
            self.assertEqual([e[2] for e in generated], [1,1,1,128])
        metadata = [call.kwargs['kwargs']['request_id'] for call in llm.collective_rpc.call_args_list
                    if call.args[0] is worker.set_request]
        self.assertEqual(len(metadata), 6)
        self.assertEqual(len(set(metadata)), 6)

    def test_session_failure_retries_only_unaccepted_prompts(self):
        f = self.fixture
        entries = [dict(id=str(i), input_path='unused', output=str(self.root/f'{i}.json')) for i in range(2)]
        cfg = dict(cache=str(self.root/'cache'), indices=[0,1])
        processes = [Mock(pid=900001), Mock(pid=900002)]
        for process in processes:
            process.poll.return_value=1
        seen = []
        def collect(root, p, case, pending, text, smoke):
            seen.append([e['id'] for e in pending])
            pending.pop(0)
            return 'SESSION_COMPLETE' if len(seen)==2 else 'Traceback: injected failure'
        processes[1].poll.return_value=0
        with patch('suite.subprocess.Popen', side_effect=processes), patch('suite.busy_devices', return_value=[]), \
             patch('suite.selected_devices', return_value=('', f.devices)), patch('suite.identity', return_value={}), \
             patch('suite.terminate'), patch('suite.group_alive', return_value=False), \
             patch('suite.collect_completed', side_effect=collect):
            suite.launch_session(cfg, self.root, f.p, 'baseline', entries)
        self.assertEqual(seen, [['0','1'], ['1']])
        manifests = [load(path) for path in (self.root/'sessions').glob('*.manifest.json')]
        self.assertEqual(sorted(len(m) for m in manifests), [1,2])

    def test_install_persistence_preserves_existing_measurements_and_protocol(self):
        f = self.fixture
        cfg = dict(samples=100)
        dump(f.root/'protocol.json', f.p | dict(settings=cfg, samples=[f.meta], measured_requests=7,
                                               scope=[dict(length=8192, task='niah_multivalue')]))
        private = f.root/'private_ucm/ucm'
        connector = private/'integration/vllm/blend_connector.py'
        connector.parent.mkdir(parents=True)
        connector.write_text('from typing_extensions import Self\n')
        runtime = private/'sparse/prophetkv/runtime.py'
        runtime.parent.mkdir(parents=True)
        runtime.write_text((common.HERE/'method/prophetkv/runtime.py').read_text())
        record = f.root/'records/old/baseline.validated.json'
        dump(record, dict(complete=True))
        original = record.read_bytes(), (f.root/'protocol.json').read_bytes()
        suite.refresh_runtime(cfg, f.root)
        self.assertEqual(original, (record.read_bytes(), (f.root/'protocol.json').read_bytes()))
        self.assertTrue((private/'integration/vllm/persistent_connector.py').is_file())


if __name__ == '__main__':
    unittest.main()
