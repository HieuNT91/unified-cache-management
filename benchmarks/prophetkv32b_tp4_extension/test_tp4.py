"""CPU checks for the local four-GPU scope and activation-memory adaptation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import common
import extension


class LocalTP4Tests(unittest.TestCase):
    def test_inference_and_tiling_match_the_preserved_cohort(self):
        self.assertTrue(extension.verify_unchanged_inference())

    def test_extension_never_schedules_retained_methods(self):
        self.assertFalse(set(common.CASES) & {'baseline', 'prophetkv-5', 'prophetkv-20'})
        self.assertEqual(len(extension.ALL_CASES), 6)

    def test_scope_is_exactly_the_requested_sixty_requests(self):
        cfg = common.settings()
        self.assertEqual(cfg['indices'], [1, 2, 3, 4])
        self.assertEqual(cfg['samples'], 10)
        self.assertEqual(cfg['chunk'], 4096)
        self.assertEqual(cfg['scope'], [dict(length=65536, task=t) for t in ('vt', 'cwe')])
        self.assertEqual(common.CASES, ('prophetkv-30', 'prophetkv-40', 'prophetkv-50'))
        self.assertEqual(cfg['samples'] * len(cfg['scope']) * len(common.CASES), 60)

    def test_physical_zero_cannot_be_selected(self):
        devices = {i: dict(index=i, name='NVIDIA RTX 4500 Ada Generation') for i in range(5)}
        with patch.object(common, 'inventory', return_value=('', devices)):
            self.assertEqual(len(common.selected_devices([1, 2, 3, 4])[1]), 4)
            with self.assertRaises(RuntimeError):
                common.selected_devices([0, 1, 2, 3])

    def test_tiling_preserves_every_token_including_remainder(self):
        # Extract only the pure function, avoiding import of any GPU worker.
        tree = ast.parse(Path(__file__).with_name('memory_worker.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tiled_mlp')
        namespace = dict(torch=torch, TILE=7)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<tiled_mlp>', 'exec'), namespace)
        for count in (1, 7, 8, 21, 24):
            calls = []
            def forward(x):
                calls.append(len(x))
                return x.square() + 3 * x
            x = torch.arange(count * 5, dtype=torch.float32).reshape(count, 5)
            actual = namespace['tiled_mlp'](SimpleNamespace(_untiled_forward=forward), x)
            self.assertTrue(torch.equal(actual, x.square() + 3 * x))
            self.assertEqual(sum(calls), count)
            self.assertLessEqual(max(calls), 7)

    def test_full_mask_preserves_tensor_identity_and_partial_mask_prunes(self):
        path = Path(__file__).parent / 'method/prophetkv/prophetkv.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'layer_begin')
        for removed in (0, 64):
            mask = torch.ones(320, dtype=torch.bool)
            mask[:removed] = False
            namespace = dict(torch=torch, probe=lambda *a: torch.empty(0), request_mask=lambda *a: mask)
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<layer_begin>', 'exec'), namespace)
            updates = []
            metadata = SimpleNamespace(compute_mask=torch.zeros_like(mask),
                update_query_lens=lambda i, n: updates.append(('removed', n)),
                update_need_re_index=lambda flag: updates.append(('reindex', flag)))
            obj = SimpleNamespace(layer_index=0, active=True, method='prophetkv',
                req=SimpleNamespace(prefix_len=64, chunk_hit_mask=[True]),
                request=SimpleNamespace(boundaries=(0, 64, 128, 384)),
                blend_req_metas=metadata, _update_attn_metadata=lambda: updates.append(('metadata', True)))
            positions = torch.arange(64, 384)
            hidden = torch.randn(320, 8)
            pos, out, residual = namespace['layer_begin'](obj, positions, hidden, None)
            self.assertTrue(torch.equal(pos, positions[mask]))
            self.assertTrue(torch.equal(out, hidden[mask]))
            self.assertIsNone(residual)
            self.assertIn(('removed', removed), updates)
            self.assertIn(('reindex', removed != 0), updates)
            if not removed:
                self.assertIs(pos, positions)
                self.assertIs(out, hidden)
                self.assertNotIn(('metadata', True), updates)


if __name__ == '__main__':
    unittest.main()
