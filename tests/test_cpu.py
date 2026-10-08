from runner.layout import stamp_sample
import copy
import sys
import types
import unittest
from unittest.mock import patch

from runner.config import engine_config, validate_sample
from ucm.sparse.prophetkv.layers import resolve_layers


class ConfigurationTests(unittest.TestCase):
    def test_shared_runtime_and_cli_help_without_gpu_imports(self):
        import os
        import subprocess
        script = '''
import sys
sys.modules.update({name: None for name in ('run', 'torch', 'vllm', 'transformers')})
from runner import generation, preparation, single
from runner import corpus_runtime, router_measure, setups, sweep
import runpy
sys.argv = ['run.py', '--help']
runpy.run_path('run.py', run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-c', script],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
            capture_output=True, text=True, check=True)
        self.assertIn('prepare', result.stdout)
        self.assertIn('setup', result.stdout)

    def test_shared_model_settings_and_baseline_isolation(self):
        for method in ('baseline', 'prophetkv', 'selective_prophetkv'):
            cfg = engine_config('/model', method, count=5 if method.startswith('selective') else None,
                                cache_dir='/cache', end_token=99)
            self.assertEqual(cfg['rope_scaling'], dict(rope_type='yarn', factor=4., original_max_position_embeddings=32768))
            self.assertEqual(cfg['max_model_len'], 131072)
            self.assertTrue(cfg['enable_chunked_prefill'])
            self.assertEqual(cfg['max_num_batched_tokens'], 16384)
            self.assertFalse(cfg['enable_prefix_caching'])
            self.assertEqual('kv_transfer_config' in cfg, method != 'baseline')

    def test_configuration_cli_dry_runs(self):
        import hashlib
        import json
        import os
        from pathlib import Path
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            config = model / 'config.json'
            config.write_text(json.dumps(dict(model_type='qwen3',num_hidden_layers=64,hidden_size=5120)))
            ids = [1] * 384
            ids[63] = ids[127] = 99
            sample = root / 'input.json'
            sample.write_text(json.dumps(stamp_sample(dict(token_ids=ids,boundaries=[0,64,128,384],
                question_positions=[380],max_output_tokens=1,thinking=False,
                model_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))))
            cases = [('baseline', []), ('prophetkv', []),
                     ('selective_prophetkv', ['--num-layers','5']),
                     ('selective_prophetkv', ['--layers','45','48','50','56','58'])]
            for method, extra in cases:
                result = subprocess.run(['bash','run.sh','run','--model',str(model),
                    '--input',str(sample),'--output',str(root/'unused'),'--method',method,
                    '--dry-run',*extra],env=dict(os.environ,PYTHON_BIN=sys.executable,CUDA_VISIBLE_DEVICES=''),
                    capture_output=True,text=True,check=True)
                cfg = json.loads(result.stdout)
                self.assertTrue(cfg['enable_chunked_prefill'])
                self.assertEqual(cfg['max_num_batched_tokens'],16384)
                self.assertEqual(cfg['max_model_len'],131072)
                self.assertFalse(cfg['enable_prefix_caching'])
                self.assertEqual('kv_transfer_config' in cfg,method!='baseline')
                self.assertFalse((root/'unused').exists())

    def test_layer_policies(self):
        self.assertEqual(resolve_layers('prophetkv'), tuple(range(64)))
        self.assertEqual(resolve_layers('selective_prophetkv', count=3), (61, 62, 63))
        self.assertEqual(resolve_layers('selective_prophetkv', [58, 45, 48]), (45, 48, 58))
        self.assertEqual(resolve_layers('selective_prophetkv', count=64), resolve_layers('prophetkv'))

    def test_reject_invalid_policies(self):
        for layers, count in [(None, None), ([], None), ([1, 1], None), ([64], None),
                               ([-1], None), ([True], None), ([1], 1), (None, 0), (None, 65)]:
            with self.subTest(layers=layers, count=count), self.assertRaises(ValueError):
                resolve_layers('selective_prophetkv', layers, count)
        with self.assertRaises(ValueError):
            resolve_layers('prophetkv', count=5)
        with self.assertRaises(ValueError):
            engine_config('/model', 'baseline', count=5)

    def test_prompt_boundaries_and_capacity(self):
        ids = [1] * 384
        ids[63] = ids[127] = 99
        sample = stamp_sample(dict(token_ids=ids, boundaries=[0, 64, 128, 384], question_positions=[380, 381],
                      max_output_tokens=256, thinking=False))
        self.assertIs(validate_sample(sample), sample)
        for key, value in [('question_positions', [1]), ('boundaries', [0, 65, 128, 384]),
                           ('max_output_tokens', 131072), ('thinking', 'false')]:
            altered = copy.deepcopy(sample)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_sample(altered)


class AlgorithmTests(unittest.TestCase):
    def test_saved_score_replay_rejects_wrong_masks_and_missing_layers(self):
        from runner.generation import verify_diagnostics
        scores = [1.] * 64
        event = dict(kind='prophetkv_selection', scores=scores,
            selected_positions=list(range(64, 96)), scoring_layers=[63],
            fusion='mean_layers_fp32', alignment_count=64)
        positions = list(range(64, 96)) + list(range(128, 384))
        step = dict(kind='prefill_step', start=64, end=384, scheduled_tokens=320,
                    recomputed_tokens=288, no_forward=False, prefill_complete=True)
        layers = [dict(start=64, end=384, selected_positions=positions,
                       projection_tokens=288, attention_tokens=288, ffn_tokens=288, kind='layer_counts', layer=f'model.layers.{i}.self_attn.attn',
                       selected_set_verified=True) for i in range(64)]
        workers = [dict(rank=i, diagnostics=[copy.deepcopy(event), copy.deepcopy(step), *copy.deepcopy(layers)]) for i in range(2)]
        sample = dict(boundaries=[0, 64, 128, 384])
        verify_diagnostics(workers, sample, 'selective_prophetkv', .5, 2, [63])
        broken = copy.deepcopy(workers)
        broken[1]['diagnostics'][0]['selected_positions'][-1] = 96
        with self.assertRaises(RuntimeError):
            verify_diagnostics(broken, sample, 'selective_prophetkv', .5, 2, [63])
        broken = copy.deepcopy(workers)
        broken[0]['diagnostics'].pop()
        with self.assertRaises(RuntimeError):
            verify_diagnostics(broken, sample, 'selective_prophetkv', .5, 2, [63])

    def test_attention_matches_dense_gqa_reference(self):
        import torch
        from ucm.sparse.prophetkv.selection import context_importance
        torch.manual_seed(7)
        q, k = torch.randn(5, 4, 8), torch.randn(19, 2, 8)
        expanded = k.repeat_interleave(2, dim=1)
        expected = torch.einsum('qhd,khd->qhk', q, expanded).div(8**.5).softmax(-1).mean((0, 1))
        torch.testing.assert_close(context_importance(q, k, key_tile=7, query_tile=2), expected)

    def test_selection_floor_and_ties(self):
        import torch
        from ucm.sparse.prophetkv.selection import select
        self.assertEqual(select(torch.ones(7), 64, .5).tolist(), [64, 65, 66])
        self.assertEqual(select(torch.ones(7), 64, 0).tolist(), [])

    def test_probe_averages_configured_layers_and_aligns_all_caches(self):
        import torch
        from ucm.sparse.prophetkv import runtime
        from ucm.sparse.prophetkv.selection import RequestMetadata, select
        for method, chosen in [('prophetkv', tuple(range(64))),
                               ('selective_prophetkv', (1, 3)),
                               ('selective_prophetkv', tuple(range(64))),
                               ('selective_prophetkv', (0,))]:
            with self.subTest(method=method, chosen=chosen):
                layers = [types.SimpleNamespace(index=i,
                    self_attn=types.SimpleNamespace(o_proj=lambda x: (x, None)),
                    post_attention_layernorm=lambda x, r: (x, r), mlp=lambda x: x) for i in range(64)]
                aligned = set()
                sparse = types.SimpleNamespace(method=method, scoring_layers=chosen, ratio=.2,
                    model=types.SimpleNamespace(layers=layers), selection_diagnostics=[],router_capture=method=='prophetkv',
                    request=RequestMetadata('test', (0, 64, 128, 384), (380, 381)),
                    attn_metadata=types.SimpleNamespace(block_table=torch.arange(6)[None]),
                    connector=types.SimpleNamespace(wait_for_layer_load=aligned.add, prophet_aligned=aligned))
                cache = torch.zeros(2, 6, 64, 1, 2)
                context = types.SimpleNamespace(virtual_engine=0, no_compile_layers={
                    f'model.layers.{i}.self_attn.attn': types.SimpleNamespace(kv_cache=[cache]) for i in range(64)})
                module = types.ModuleType('vllm.forward_context')
                module.get_forward_context = lambda: context
                projected = []
                def project(layer, hidden, residual, positions):
                    projected.append(layer.index)
                    value = torch.full((256, 1, 2), float(layer.index + 1))
                    return value, value, value, hidden
                def importance(q, k):
                    return torch.full((len(k),), float(q[0, 0, 0]))
                def selection(scores, prefix, ratio):
                    eligible = scores[prefix:]
                    return eligible, select(eligible, prefix, ratio)
                with patch.dict(sys.modules, {'vllm.forward_context': module}), \
                        patch.object(runtime, 'project', project), \
                        patch.object(runtime, 'context_importance', importance), \
                        patch.object(runtime, 'global_selection', selection), \
                        patch('torch.cuda.synchronize'):
                    runtime.probe(sparse, torch.arange(64, 384), torch.zeros(320, 2))
                self.assertEqual(projected, list(range(chosen[-1] + 1)))
                self.assertEqual(len(aligned), 64)
                event = sparse.selection_diagnostics[0]
                torch.testing.assert_close(event['scores'], torch.full((64,), sum(i + 1 for i in chosen) / len(chosen)))
                self.assertEqual(event['fusion'], 'mean_layers_fp32')
                if method == 'prophetkv':
                    captured, global_scores, local_mean = sparse.router_arrays
                    self.assertEqual(tuple(captured.shape),(64,128))
                    replay=torch.zeros(128,dtype=torch.float32)
                    for values in captured:replay.add_(values)
                    replay.div_(64)
                    self.assertTrue(torch.equal(replay,local_mean))
                    self.assertTrue(torch.equal(global_scores,event['scores']))

    def test_delta_rotation_does_not_double_yarn_amplitude(self):
        import torch
        from ucm.sparse.prophetkv.runtime import delta_rotation_table
        table = torch.tensor([[1.2, 1.2, 0., 0.], [.6, .6, .6, .6]])
        torch.testing.assert_close(delta_rotation_table(table), table / 1.2)


if __name__ == '__main__':
    unittest.main()
