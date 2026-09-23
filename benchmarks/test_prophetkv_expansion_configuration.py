"""Constructor regression without vLLM initialization or GPU allocations."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ucm/sparse'))


class ConfigurationDispatch(unittest.TestCase):
    def test_explicit_expansion_budget_and_legacy_alias(self):
        stub = ModuleType('ucm.sparse.blend.blend')

        class Blend:
            def __init__(self, config, role):
                self.blend_config = config.kv_transfer_config.kv_connector_extra_config['ucm_sparse_config']['Blend']

        stub.Blend = Blend

        def config(parameters):
            return SimpleNamespace(kv_transfer_config=SimpleNamespace(
                kv_connector_extra_config={'ucm_sparse_config': {'ProphetKV': parameters}}))

        previous = sys.modules.pop('prophetkv.prophetkv', None)
        try:
            with patch.dict(sys.modules, {'ucm.sparse.blend.blend': stub}):
                ProphetKV = importlib.import_module('prophetkv.prophetkv').ProphetKV
                for total, anchor in ((.2, .15), (.3, .1), (0., 0.), (1., 1.)):
                    parameters = dict(method='prophetkv_with_expansion',
                                      expansion=dict(total_ratio=total, anchor_ratio=anchor))
                    sparse = ProphetKV(config(parameters), None)
                    self.assertEqual(sparse.ratio, total)
                    self.assertEqual(sparse.expansion_config.anchor_ratio, anchor)
                    self.assertEqual(sparse.compute_meta, {})
                with self.assertRaises(ValueError):
                    ProphetKV(config(dict(method='prophetkv_with_expansion', ratio=.3)), None)
                with self.assertRaises(ValueError):
                    ProphetKV(config(dict(method='prophetkv_with_expansion',
                                          expansion=dict(total_ratio=.3))), None)
                self.assertEqual(ProphetKV(config(dict(method='prophetkv', ratio=.3)), None).ratio, .3)
        finally:
            sys.modules.pop('prophetkv.prophetkv', None)
            if previous is not None:
                sys.modules['prophetkv.prophetkv'] = previous


if __name__ == '__main__':
    unittest.main()
