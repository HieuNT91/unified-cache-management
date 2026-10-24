"""CPU coverage for the L20 tree collection's native pinned transfer pool."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runner.setups import bytes_per_shard
from runner.tree_collection import SCHEMA, actions, configure_engine
from runner.tree_prophetkv import PROTOCOL
from runner.tree_profiles import config


def configured(hardware='l20-tp2', dataset='ruler-thinking', tp=2):
    definitions = actions([5, 20, 40, 60])
    protocol = dict(schema=SCHEMA, tree_protocol=PROTOCOL, percentages=[5,20,40,60],
                    actions=definitions, scheduled_actions=[a['id'] for a in definitions],
                    phase_order=['cached'], answer_validation='native-answer-diagnostics-v1',
                    hardware_profile=hardware, tp=tp)
    cfg = config('/model', dataset, True, '/cache', hardware=hardware, tp=tp)
    before = copy.deepcopy(protocol)
    configure_engine(cfg, Path('/cache'), protocol)
    if protocol != before:
        raise AssertionError('Runtime buffer tuning changed frozen protocol')
    return cfg


class TreeTransferBufferTests(unittest.TestCase):
    def test_l20_pool_geometry_and_unchanged_transfer_policy(self):
        cfg = configured()
        extra = cfg['kv_transfer_config']['kv_connector_extra_config']
        store = extra['ucm_connectors'][0]['ucm_connector_config']
        self.assertEqual(store, dict(storage_backends='/cache', use_direct=False,
                                     timeout_ms=120000, buffer_number=1152))
        self.assertNotIn('max_load_blocks', extra)
        self.assertEqual(cfg['gpu_memory_utilization'], .95)
        self.assertEqual(cfg['max_model_len'], 82304)
        self.assertEqual(cfg['cpu_offload_gb'], 0)
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model/'config.json').write_text(json.dumps(dict(num_key_value_heads=8,
                num_attention_heads=64, hidden_size=5120, head_dim=128, num_hidden_layers=64)))
            size = bytes_per_shard(model, 2)
        self.assertEqual(size, 8 * 1024**2)
        self.assertEqual(size * store['buffer_number'] * 10, 90 * 1024**3)
        self.assertGreater(store['buffer_number'], 65536 // cfg['block_size'])

    def test_other_profiles_and_non_tree_l20_keep_existing_pool(self):
        for hardware, dataset, tp in (
                ('server', 'longbench-v2', 2), ('l40-tp2', 'ruler-thinking', 2),
                ('rtx4500ada-tree-validation', 'ruler-thinking', 4),
                ('l20-tp2', 'ruler-thinking', 4)):
            with self.subTest(hardware=hardware, tp=tp):
                cfg = configured(hardware, dataset, tp)
                extra = cfg['kv_transfer_config']['kv_connector_extra_config']
                self.assertNotIn('buffer_number', extra['ucm_connectors'][0]['ucm_connector_config'])
        for cached in (True, False):
            cfg = config('/model', 'ruler-thinking', cached, '/cache', hardware='l20-tp2', tp=2)
            if cached:
                self.assertNotIn('buffer_number', cfg['kv_transfer_config']['kv_connector_extra_config']
                                 ['ucm_connectors'][0]['ucm_connector_config'])
            else:
                self.assertNotIn('kv_transfer_config', cfg)

    def test_python_backend_forwards_buffer_count_to_native_config(self):
        from ucm.store.pcstore import pcstore_connector_v1 as backend
        native_config = SimpleNamespace(transferBufferNumber=4096, transferTimeoutMs=30000,
                                       transferDeviceId=-1)
        received = []
        class NativeStore:
            @staticmethod
            def Config(*args):
                return native_config
            def Setup(self, value):
                received.append(value)
                return 0
        cfg = configured()['kv_transfer_config']['kv_connector_extra_config']
        options = dict(cfg['ucm_connectors'][0]['ucm_connector_config'], device_id=0)
        # No CUDA allocation or native Setup: exercise the real Python mapping only.
        with patch.object(backend.ucmpcstore, 'PcStore', NativeStore):
            backend.UcmPcStoreV1(options)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].transferBufferNumber, 1152)
        self.assertEqual(received[0].transferTimeoutMs, 120000)


if __name__ == '__main__':
    unittest.main()
