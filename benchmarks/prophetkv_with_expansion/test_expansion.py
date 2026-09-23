"""CPU checks; CUDA parity tests are added with the selector implementation."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'ucm/sparse'))
from prophetkv.selection import (ExpansionConfig, expansion_lookup, select,
                                select_expansion, select_expansion_reference)

spec = importlib.util.spec_from_file_location('expansion_data', HERE / 'data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class Configuration(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(asdict(ExpansionConfig()), dict(
            total_ratio=.20, anchor_ratio=.15, max_gap=2, score_exponent=.5,
            window_scale=8., window_exponent=.5, min_window=8, max_window=64))

    def test_explicit_budgets(self):
        for ratio in (0., .1, .3, 1.):
            with self.assertRaises(ValueError):
                ExpansionConfig(total_ratio=ratio)
            self.assertEqual(ExpansionConfig(total_ratio=ratio, anchor_ratio=ratio).anchor_ratio, ratio)

    def test_invalid(self):
        for name in asdict(ExpansionConfig()):
            for value in (float('nan'), float('inf'), -float('inf'), True, '1'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ExpansionConfig(**{name: value})
        for kwargs in (dict(total_ratio=1.1, anchor_ratio=.1), dict(anchor_ratio=.3),
                       dict(anchor_ratio=-.1), dict(max_gap=-1), dict(max_gap=2.),
                       dict(min_window=1.5), dict(max_window=-1), dict(window_scale=-1),
                       dict(min_window=65, max_window=64)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExpansionConfig(**kwargs)


class InputSelection(unittest.TestCase):
    def test_inclusive_limit_and_first_qualifying_order(self):
        rows = [65537, 65536, 65535, 70000, 61120, 62000]
        selected, census = data.qualifying(rows, lambda n: dict(tokens=n), count=3)
        self.assertEqual([i for i, _, _ in selected], [1, 2, 4])
        self.assertEqual([encoded['tokens'] for _, _, encoded in selected], [65536, 65535, 61120])
        self.assertEqual(len(census), 5)

    def test_insufficient_inputs_fail(self):
        with self.assertRaises(ValueError):
            data.qualifying([65537, 65536], lambda n: dict(tokens=n), count=2)


class Selector(unittest.TestCase):
    def check_case(self, values, config, expected, prefix=4096):
        scores = torch.tensor(values, dtype=torch.float64)
        selected, d = select_expansion(scores, prefix, config, diagnostics=True)
        self.assertEqual(selected.tolist(), [x + prefix for x in expected])
        self.assertEqual(selected.dtype, torch.long)
        self.assertEqual(d['final_count'], len(expected))
        self.assertEqual(d['anchor_count'] + d['unique_expansion_count'] + d['fallback_count'], len(expected))
        return d

    def test_zero_empty_full_and_floor(self):
        for n in (0, 1, 11):
            for total, anchor in ((0., 0.), (.2, .15), (1., 0.), (1., 1.)):
                c = ExpansionConfig(total_ratio=total, anchor_ratio=anchor)
                self.check_case([1.] * n, c, list(range(int(n * total))))

    def test_reject_nonfinite_even_zero_and_full(self):
        for total in (0., 1.):
            c = ExpansionConfig(total_ratio=total, anchor_ratio=total)
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.assertRaises(ValueError):
                    select_expansion(torch.tensor([value]), 0, c)

    def test_transitive_groups_anchor_only_sum_internal_gaps(self):
        c = ExpansionConfig(total_ratio=.6, anchor_ratio=.3, max_gap=2, min_window=1, max_window=1)
        scores = torch.tensor([10., 0., 0., 9., 0., 0., 8., 0., 0., 0.])
        selected, d = select_expansion_reference(scores, 0, c, diagnostics=True, trace=True)
        self.assertEqual(d['segment_count'], 1)
        self.assertEqual(d['segments'][0]['anchors'], 3)
        self.assertAlmostEqual(d['segments'][0]['score'], 27 / 3**.5)
        self.assertEqual(selected.tolist(), [0, 1, 2, 3, 6, 7])
        self.assertEqual(d['unique_expansion_count'], 1)
        self.assertEqual(d['fallback_count'], 2)

    def test_segment_ties_leftmost_and_partial_window(self):
        c = ExpansionConfig(total_ratio=.5, anchor_ratio=.2, max_gap=0, min_window=8, max_window=8)
        self.check_case([0., 10., 0., 0., 0., 0., 10., 0., 0., 0.], c, [1, 2, 3, 4, 6])

    def test_overlap_first_visit_and_anchor_retention(self):
        c = ExpansionConfig(total_ratio=.8, anchor_ratio=.2, max_gap=0, min_window=5, max_window=5)
        d = self.check_case([0., 10., 0., 9., 0., 0., 0., 0., 0., 0.], c, list(range(1, 9)))
        self.assertEqual(d['unique_expansion_count'], 6)

    def test_suffix_clipping_fallback_and_zero_anchors(self):
        c = ExpansionConfig(total_ratio=.5, anchor_ratio=.2)
        d = self.check_case([1., 2., 3., 4., 5., 6., 7., 8., 20., 19.], c, [5, 6, 7, 8, 9])
        self.assertEqual(d['fallback_count'], 3)
        d = self.check_case([1., 4., 2., 3.], ExpansionConfig(total_ratio=.5, anchor_ratio=0), [1, 3])
        self.assertEqual(d['segment_count'], 0)

    def test_anchor_count_normalization_not_span(self):
        c = ExpansionConfig(total_ratio=.6, anchor_ratio=.4, max_gap=2, min_window=1, max_window=1)
        scores = torch.tensor([10., 0., 0., 10., 0., 0., 0., 11., 11., 0.])
        _, d = select_expansion_reference(scores, 0, c, diagnostics=True, trace=True)
        self.assertEqual([s['start'] for s in d['segments']], [7, 0])
        self.assertEqual([s['window'] for s in d['segments']], [1, 1])
        self.assertAlmostEqual(d['segments'][1]['score'], 20 / 2**.5)

    def test_rounding_clamping_and_zero_windows(self):
        for scale, expected in ((2.5, 2), (3.5, 4), (0., 0)):
            c = ExpansionConfig(window_scale=scale, window_exponent=0., min_window=0)
            self.assertEqual(expansion_lookup(c, 2)[1], (0, expected, expected))
        c = ExpansionConfig(window_scale=100., min_window=8, max_window=64)
        self.assertEqual(expansion_lookup(c, 2)[1], (0, 64, 64))

    def test_cross_chunk_expansion(self):
        values = torch.zeros(8192)
        values[4094] = 10
        c = ExpansionConfig(total_ratio=5/8192, anchor_ratio=1/8192, min_window=8)
        self.assertEqual(select_expansion(values, 4096, c).tolist(), list(range(8190, 8195)))

    def test_original_topk_unchanged(self):
        generator = torch.Generator().manual_seed(123)
        for n in (0, 11, 1000):
            scores = torch.randint(0, 9, (n,), generator=generator).float()
            for ratio in (0., .15, .2, 1.):
                expected = sorted(sorted(range(n), key=lambda i: (-scores[i].item(), i))[:int(n*ratio)])
                self.assertEqual(select(scores, 100, ratio).tolist(), [i+100 for i in expected])


class Integration(unittest.TestCase):
    def test_both_methods_share_once_only_layer_zero_probe(self):
        import importlib
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch, Mock
        stub = ModuleType('ucm.sparse.blend.blend')

        class Blend:
            def __init__(self, config, role):
                self.blend_config=config.kv_transfer_config.kv_connector_extra_config['ucm_sparse_config']['Blend']
                self.compute_meta={'legacy': True}

            def layer_begin(self, positions, hidden, residual):
                return positions, hidden, residual

        stub.Blend=Blend
        with patch.dict(sys.modules, {'ucm.sparse.blend.blend': stub}):
            sys.modules.pop('prophetkv.prophetkv', None)
            module=importlib.import_module('prophetkv.prophetkv')
            try:
                for method in ('prophetkv', 'prophetkv_with_expansion'):
                    config=SimpleNamespace(kv_transfer_config=SimpleNamespace(
                        kv_connector_extra_config={'ucm_sparse_config': {'ProphetKV': {'method': method}}}))
                    sparse=module.ProphetKV(config, None)
                    self.assertEqual(sparse.compute_meta, {})
                    sparse.active=True
                    sparse.req=SimpleNamespace(prefix_len=64,chunk_hit_mask=[True]*3)
                    sparse.request=SimpleNamespace(boundaries=(0,64,256,512))
                    sparse.blend_req_metas=SimpleNamespace(compute_mask=torch.zeros(448,dtype=torch.bool),
                                                         update_query_lens=Mock(),update_need_re_index=Mock())
                    sparse._update_attn_metadata=Mock()
                    with patch.object(module, 'probe', return_value=torch.tensor([66,100])) as probe:
                        positions, hidden, residual=sparse.layer_begin(torch.arange(64,512),torch.ones(448,2),None)
                        self.assertEqual(positions.tolist(),[66,100]+list(range(256,512)))
                        for _ in range(35):
                            positions,hidden,residual=sparse.layer_begin(positions,hidden,residual)
                        probe.assert_called_once()
                alias=SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config={
                    'ucm_sparse_config': {'prophetkv_with_expansion': {}}}))
                self.assertEqual(module.ProphetKV(alias,None).method,'prophetkv_with_expansion')
            finally:
                sys.modules.pop('prophetkv.prophetkv', None)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA selector checks run after owned GPU release')
class CudaParity(unittest.TestCase):
    def compare(self, scores, config):
        reference, rd = select_expansion_reference(scores, 4096, config, diagnostics=True)
        selected, gd = select_expansion(scores.cuda(), 4096, config, diagnostics=True)
        self.assertTrue(torch.equal(reference, selected.cpu()))
        self.assertEqual(rd, {k: v.item() if torch.is_tensor(v) else v for k, v in gd.items()})

    def test_randomized_and_crafted(self):
        generator = torch.Generator().manual_seed(20260923)
        configs = [ExpansionConfig(), ExpansionConfig(total_ratio=.5, anchor_ratio=.2, max_gap=0),
                   ExpansionConfig(total_ratio=.8, anchor_ratio=.1, max_gap=20, min_window=0, max_window=0),
                   ExpansionConfig(total_ratio=1., anchor_ratio=0.), ExpansionConfig(total_ratio=1., anchor_ratio=1.),
                   ExpansionConfig(total_ratio=0., anchor_ratio=0.),
                   ExpansionConfig(total_ratio=.6, anchor_ratio=.3, max_gap=2, window_scale=2.5, window_exponent=0., min_window=0)]
        for n in (0, 1, 11, 100, 1024, 61440):
            for c in configs:
                for scores in (torch.randn(n, generator=generator), torch.ones(n),
                               torch.randint(0, 9, (n,), generator=generator).double()):
                    with self.subTest(n=n, config=c):
                        self.compare(scores, c)

    def test_near_ties_and_sequential_float64(self):
        c = ExpansionConfig(total_ratio=.8, anchor_ratio=.4, max_gap=0, min_window=1, max_window=1)
        for epsilon in (0., 2**-50, 2**-30):
            self.compare(torch.tensor([1., 1.+epsilon, 0., 0., 1., 1., 0., 0., 0., 0.], dtype=torch.float64), c)

    def test_saved_probe_scores(self):
        # Existing probe artifacts are read-only inputs, never inference inputs.
        import json
        root = HERE.parents[1] / '.results/qwen3-32b-prophetkv-yarn2-ruler10x50-lb50-20260922'
        paths = sorted(root.glob('records/**/*.diagnostics.json'))[:8]
        self.assertTrue(paths, 'Expected preserved probe-score artifacts')
        for path in paths:
            for worker in json.loads(path.read_text()):
                for event in worker['diagnostics']:
                    if event.get('kind') == 'prophetkv_selection':
                        self.compare(torch.tensor(event['scores']), ExpansionConfig())


if __name__ == '__main__':
    unittest.main()
