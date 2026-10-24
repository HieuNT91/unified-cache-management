"""Opt-in real CUDA parity; set UCM_TREE_CUDA_TEST=1 on assigned GPU UUIDs."""
import os
import unittest


@unittest.skipUnless(os.environ.get('UCM_TREE_CUDA_TEST') == '1', 'Explicit CUDA test opt-in required')
class TreeAttentionCudaTests(unittest.TestCase):
    def test_bf16_gqa_context_and_chunked_suffix_match_dense(self):
        import torch
        from ucm.sparse.prophetkv import tree_attention

        self.assertTrue(os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'),
                        'Assign GPU UUIDs explicitly before running this test')
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(7)
        device = 'cuda'
        small = [64, 65, 90, 127, 128, 140, 190]
        large = list(range(64, 512, 2)) + list(range(512, 1024, 3))
        cases = [
            ([0, 64, 128, 192, 448], small, small[:4]),
            ([0, 64, 128, 192, 448], small, small + list(range(192, 240))),
            ([0, 64, 128, 192, 448], small, list(range(240, 300))),
            ([0, 64, 512, 1024, 1088], large, large + list(range(1024, 1088))),
        ]
        for bounds, selected, rows in cases:
            with self.subTest(query_rows=len(rows), suffix_start=bounds[-2]):
                positions = torch.tensor(rows, device=device)
                recomputed = torch.tensor(selected, device=device)
                key = torch.randn(max(rows)+1, 2, 128, device=device, dtype=torch.bfloat16)
                value = torch.randn_like(key)
                query = torch.randn(len(rows), 16, 128, device=device, dtype=torch.bfloat16)
                output = torch.full_like(query, float('nan'))
                allowed = torch.cat((torch.arange(bounds[1], device=device), recomputed))
                tree_attention.rmask_attention(query, key, value, positions, bounds,
                                               recomputed, 128**-.5, output)
                tree_attention.masked_suffix_attention(query, key, value, positions,
                                                       bounds, allowed, 128**-.5, output)
                reference = tree_attention.dense_reference(query, key, value, positions,
                                                           bounds, recomputed)
                torch.testing.assert_close(output.float(), reference, atol=.008, rtol=.02)
                print(f'CUDA BF16 tree mask: rows={len(rows)} '
                      f'max_abs_error={(output.float()-reference).abs().max().item():.6g}')


if __name__ == '__main__':
    unittest.main()
