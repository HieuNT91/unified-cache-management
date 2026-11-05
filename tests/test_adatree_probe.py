"""Exercise routing inside the native TreeKV probe, without a GPU request."""
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch._dynamo

from ucm.sparse.prophetkv import runtime, attention
from ucm.sparse.prophetkv.selection import RequestMetadata
from runner.tree.prophetkv import write_f3


class IntegratedProbeTests(unittest.TestCase):
    def test_one_post_s4_probe_routes_before_the_only_tree_selection(self):
        layers = [NS(self_attn=NS(o_proj=lambda x: (x, None), head_dim=2),
                     post_attention_layernorm=lambda x, r: (x, r), mlp=lambda x: x)
                  for _ in range(64)]
        aligned, scored, sinks, reductions = set(), [], [], []
        cache = torch.zeros(2, 10, 64, 1, 2)
        before = cache.clone()
        context = NS(virtual_engine=0, no_compile_layers={
            f'model.layers.{i}.self_attn.attn': NS(kv_cache=[cache]) for i in range(64)})

        def project(layer, hidden, residual, positions):
            q = positions.float()[:, None, None].expand(-1, 1, 2)
            return q, q, q, hidden

        def importance(q, k):
            scored.append(q[:, 0, 0].tolist())
            for start in (128, 256):
                torch.testing.assert_close(k[start:start+4, 0, 0], torch.arange(start, start+4).float())
            return torch.ones(len(k))

        def kernel(q, k, v, out, qs, qn, ks, kn, positions, maxq, **kwargs):
            sinks.append(positions.tolist())
            out.zero_()

        def reduce(value):
            reductions.append(tuple(value.shape))
            return 2 * value

        group = NS(world_size=2, broadcast=lambda value, src: value)
        modules = {'vllm.forward_context': NS(get_forward_context=lambda: context),
                   'vllm.distributed': NS(get_tp_group=lambda: group, tensor_model_parallel_all_reduce=reduce)}
        with tempfile.TemporaryDirectory() as root:
            for start in (128, 256):
                write_f3(root, 'fixture', list(range(start, start+128)), [.5]*128)
            sparse = NS(method='tree-prophetkv', adatree_v1=True,
                        scoring_layers=tuple(range(64)), ratio=.05,
                        model=NS(layers=layers, get_input_embeddings=lambda ids: ids.float()[:, None].expand(-1, 2)),
                        prompt_token_ids=tuple(range(640)),
                        request=RequestMetadata('request', (0, 128, 256, 384, 640), (400, 401)),
                        selection_diagnostics=[], attn_metadata=NS(block_table=torch.arange(10)[None]),
                        connector=NS(wait_for_layer_load=aligned.add, prophet_aligned=aligned),
                        tree_f3_config=dict(cache_dir=root, namespace='fixture', readonly=True))
            with patch.dict('sys.modules', modules), patch.object(runtime, 'project', project), \
                    patch.object(runtime, 'context_importance', importance), \
                    patch.object(attention, 'positional_kernel', return_value=kernel), \
                    patch('torch.cuda.synchronize') as synchronize:
                selected = runtime.probe(sparse, torch.arange(384, 640), torch.zeros(256, 2))
                # Keep only the original probe barrier; routing adds none for
                # timing or profiling.
                synchronize.assert_called_once_with()
        self.assertEqual(scored, [list(map(float, range(400, 640)))] * 64)
        self.assertEqual(sinks, [list(range(128, 132)) + list(range(256, 260))] * 64)
        self.assertEqual(reductions, [(384,), (64, 256)])
        self.assertEqual(len(aligned), 64)
        self.assertEqual(sparse.ratio, .05)
        self.assertEqual(len(selected), 12)
        self.assertTrue(set(range(128, 132)) | set(range(256, 260)) <= set(selected.tolist()))
        route, selection = sparse.selection_diagnostics
        self.assertEqual(route['kind'], 'adatree_v1')
        self.assertEqual(route['action'], 'tree-prophetkv-5')
        self.assertAlmostEqual(route['features']['S60_p90'], .5)
        self.assertTrue(route['feature_mask_only'])
        self.assertEqual(route['independent_probes'], 0)
        self.assertEqual(selection['kind'], 'prophetkv_selection')
        self.assertEqual(selection['ratio'], .05)
        self.assertFalse(any('seconds' in key or 'warning' in key or 'overhead' in key for key in route))
        self.assertNotIn('native_probe_seconds', selection)
        self.assertGreaterEqual(selection['probe_seconds'], 0)
        self.assertGreaterEqual(selection['tree_seconds'], 0)
        torch.testing.assert_close(cache, before)


if __name__ == '__main__':
    unittest.main()
