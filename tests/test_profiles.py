import unittest

from runner.config import engine_config
from runner.profiles import PROFILES, profile_of
from ucm.sparse.prophetkv.layers import resolve_layers

QWEN32 = dict(model_type='qwen3', num_hidden_layers=64, hidden_size=5120, max_position_embeddings=40960)
QWEN4 = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=2560, max_position_embeddings=262144)


class ProfileTests(unittest.TestCase):
    def test_official_default_is_unchanged(self):
        cfg = engine_config('/model', 'prophetkv', .2, cache_dir='/c', end_token=1)
        self.assertEqual(cfg['max_model_len'], 131072)
        self.assertEqual(cfg['rope_scaling'], dict(rope_type='yarn', factor=4.0,
                                                   original_max_position_embeddings=32768))
        self.assertEqual(engine_config('/model', 'prophetkv', .2, cache_dir='/c', end_token=1,
                                       profile='qwen3-32b'), cfg)
        self.assertEqual(resolve_layers('prophetkv'), tuple(range(64)))
        store = cfg['kv_transfer_config']['kv_connector_extra_config']['ucm_connectors'][0]['ucm_connector_config']
        self.assertNotIn('buffer_number', store)

    def test_qwen3_4b_2507_is_native_rope_36_layers(self):
        self.assertEqual(profile_of(QWEN32)[0], 'qwen3-32b')
        self.assertEqual(profile_of(QWEN4)[0], 'qwen3-4b-instruct-2507')
        cfg = engine_config('/model', 'selective_prophetkv', .1, count=5, cache_dir='/c', end_token=1,
                            profile='qwen3-4b-instruct-2507')
        self.assertNotIn('rope_scaling', cfg)
        self.assertEqual(cfg['max_model_len'], PROFILES['qwen3-4b-instruct-2507']['window'])
        sparse = cfg['kv_transfer_config']['kv_connector_extra_config']['ucm_sparse_config']['ProphetKV']
        self.assertEqual(sparse['scoring_layers'], [31, 32, 33, 34, 35])
        store = cfg['kv_transfer_config']['kv_connector_extra_config']['ucm_connectors'][0]['ucm_connector_config']
        self.assertEqual(store['buffer_number'], 1152)
        self.assertEqual(resolve_layers('prophetkv', num_layers=36), tuple(range(36)))
        with self.assertRaises(ValueError):
            resolve_layers('selective_prophetkv', [36], num_layers=36)

    def test_rejects_other_checkpoints(self):
        for cfg in (dict(QWEN4, max_position_embeddings=40960), dict(QWEN32, model_type='llama'),
                    dict(QWEN32, quantization_config={'bits': 4})):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                profile_of(cfg)


if __name__ == '__main__':
    unittest.main()
