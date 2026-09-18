"""CPU safety/protocol tests. No GPU work or historical scheduler imports."""
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
from unittest.mock import patch

import common
import suite
from cacheblend_ruler import verify_cache, prompt_digest
from common import dump, load, sha


class ProtocolTests(unittest.TestCase):
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
                cfg = worker.build(SimpleNamespace(case=case, cache_dir=Path('/cache'), smoke=False), sample, p)
                self.assertEqual(cfg['tensor_parallel_size'], 2)
                self.assertFalse(cfg['enable_prefix_caching'])
                self.assertEqual(cfg['rope_scaling'], p['rope_scaling_64k'])
                self.assertGreaterEqual(cfg['max_model_len'], sample['tokens'] + 128)
                if case == 'baseline':
                    self.assertNotIn('kv_transfer_config', cfg)
                else:
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
                           prompt_sha256=prompt_digest(ids), boundaries=[0, 64, 128, 384], tokens=len(ids),
                           fresh_suffix_tokens=256, context_target=8192, query=dict(positions=[370]), source_metadata=dict(references=['123']))
        self.input_path = self.root / 'input.json'
        dump(self.input_path, self.sample)
        self.meta = self.sample | dict(input_path=str(self.input_path), input_sha256=sha(self.input_path))
        self.path = self.root / 'prophetkv-50.json'
        self.imports = [dict(rank=i, visible_uuid=d['uuid'], ucm_path=str(self.root / 'private_ucm/ucm/__init__.py'))
                        for i, d in enumerate(self.devices)]
        selection = dict(kind='prophetkv_selection', request_id='measured-1-0', eligible_count=64,
                         selected_count=32, probe_layers=2, alignment_count=2, question_positions=[370],
                         scores=[1.] * 64, selected_positions=list(range(64, 96)))
        layers = [dict(kind='layer_counts', layer=f'model.layers.{i}.self_attn.attn',
                       projection_tokens=288, attention_tokens=288, ffn_tokens=288) for i in range(2)]
        self.diag = [w | dict(diagnostics=copy.deepcopy([selection, *layers])) for w in self.imports]
        self.record = dict(sample_id='sample', case='prophetkv-50', label='niah_multivalue', context_target=8192, input_sha256=sha(self.input_path),
                           prompt_sha256=self.sample['prompt_sha256'], protocol_sha256=sha(self.root / 'protocol.json'),
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
        dump(self.path, self.record | dict(diagnostics_sha256=sha(dp)))

    def test_accept_and_reject_tampered_resume(self):
        suite.accept(self.root, self.p, self.meta, 'prophetkv-50', self.path)
        self.assertTrue(suite.valid(self.root, self.p, self.meta, 'prophetkv-50', self.path))
        self.path.with_suffix('.log').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Accepted artifact changed'):
            suite.valid(self.root, self.p, self.meta, 'prophetkv-50', self.path)

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


if __name__ == '__main__':
    unittest.main()
