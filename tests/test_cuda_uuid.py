"""UUID startup regression, without CUDA or a locally patched vLLM install."""
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from ucm.integration.vllm.patch.patch_funcs.v092.cuda_uuid import patch_cuda_uuid_mapping


class CudaUuidTests(unittest.TestCase):
    def setUp(self):
        class Platform:
            device_control_env_var = 'CUDA_VISIBLE_DEVICES'

            @classmethod
            def device_id_to_physical_device_id(cls, device_id):
                visible = os.environ.get(cls.device_control_env_var, '')
                return int(visible.split(',')[device_id]) if visible else device_id

        class CudaPlatform(Platform):
            pass

        self.platform = Platform
        self.cuda = CudaPlatform
        self.nvml = Mock()
        self.nvml.nvmlDeviceGetHandleByUUID.side_effect = lambda value: {'GPU-a': 7, 'GPU-b': 2}[value]
        self.nvml.nvmlDeviceGetIndex.side_effect = lambda handle: handle
        interface = ModuleType('vllm.platforms.interface')
        interface.Platform = Platform
        utils = ModuleType('vllm.utils')
        utils.import_pynvml = Mock(return_value=self.nvml)
        modules = {'vllm.platforms.interface': interface, 'vllm.utils': utils}
        for context in (patch.dict(sys.modules, modules),
                        patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-b,GPU-a'})):
            context.start()
            self.addCleanup(context.stop)

    def test_original_failure_then_uuid_order_and_repeated_install(self):
        with self.assertRaises(ValueError):
            self.cuda.device_id_to_physical_device_id(0)
        patch_cuda_uuid_mapping()
        mapper = self.platform.device_id_to_physical_device_id.__func__
        patch_cuda_uuid_mapping()
        self.assertIs(mapper, self.platform.device_id_to_physical_device_id.__func__)
        self.assertEqual([self.cuda.device_id_to_physical_device_id(i) for i in (0, 1)], [2, 7])
        self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-b,GPU-a')
        os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-a,GPU-b'
        self.assertEqual(self.cuda.device_id_to_physical_device_id(0), 7)
        self.assertEqual(self.nvml.nvmlInit.call_count, 3)
        self.assertEqual(self.nvml.nvmlShutdown.call_count, 3)

    def test_numeric_empty_and_unset_visibility_keep_original_behavior(self):
        patch_cuda_uuid_mapping()
        for visible in ('5,3', '', None):
            if visible is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible
            self.assertEqual(self.cuda.device_id_to_physical_device_id(1), 3 if visible else 1)
        self.nvml.nvmlInit.assert_not_called()

    def test_unknown_uuid_fails_and_releases_nvml(self):
        patch_cuda_uuid_mapping()
        os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-unknown'
        with self.assertRaises(KeyError):
            self.cuda.device_id_to_physical_device_id(0)
        self.nvml.nvmlShutdown.assert_called_once()
        with self.assertRaises(IndexError):
            self.cuda.device_id_to_physical_device_id(1)

    def test_other_platform_uses_its_original_mapping(self):
        patch_cuda_uuid_mapping()
        class OtherPlatform(self.platform):
            device_control_env_var = 'ASCEND_RT_VISIBLE_DEVICES'
        with patch.dict(os.environ, {'ASCEND_RT_VISIBLE_DEVICES': '6,4'}):
            self.assertEqual(OtherPlatform.device_id_to_physical_device_id(0), 6)
        self.nvml.nvmlInit.assert_not_called()

    def test_mapping_installed_before_backend_patches(self):
        from ucm.integration.vllm.patch import apply_patch as module
        for rerope in (False, True):
            with self.subTest(rerope=rerope), \
                    patch.object(module, '_patches_applied', False), \
                    patch.object(module, 'vllm_use_rerope', rerope), \
                    patch.object(module, 'PLATFORM', 'cuda'), \
                    patch.object(module, 'get_vllm_version', return_value='0.9.2'), \
                    patch.object(module, '_apply_patches_v092') as normal, \
                    patch.object(module, '_apply_patches_rerope') as other:
                backend = other if rerope else normal
                backend.side_effect = lambda: self.assertEqual(
                    self.cuda.device_id_to_physical_device_id(0), 2)
                module.apply_all_patches()
                backend.assert_called_once()


if __name__ == '__main__':
    unittest.main()
