from runner.inputs.layout import stamp_sample
import copy
import unittest

from runner.inputs.config import engine_config, validate_sample


class ConfigurationTests(unittest.TestCase):

    def test_shared_model_settings_and_baseline_isolation(self):
        for method in ('baseline', 'adatree-v1'):
            cfg = engine_config('/model', method, cache_dir='/cache', end_token=99)
            self.assertEqual(cfg['rope_scaling'], dict(rope_type='yarn', factor=4., original_max_position_embeddings=32768))
            self.assertEqual(cfg['max_model_len'], 131072)
            self.assertTrue(cfg['enable_chunked_prefill'])
            self.assertEqual(cfg['max_num_batched_tokens'], 16384)
            self.assertFalse(cfg['enable_prefix_caching'])
            self.assertEqual('kv_transfer_config' in cfg, method != 'baseline')




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


    def test_delta_rotation_does_not_double_yarn_amplitude(self):
        import torch
        from ucm.sparse.prophetkv.runtime import delta_rotation_table
        table = torch.tensor([[1.2, 1.2, 0., 0.], [.6, .6, .6, .6]])
        torch.testing.assert_close(delta_rotation_table(table), table / 1.2)


if __name__ == '__main__':
    unittest.main()
