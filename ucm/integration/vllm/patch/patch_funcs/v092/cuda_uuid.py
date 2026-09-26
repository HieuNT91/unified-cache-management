"""Keep CUDA UUID visibility usable by vLLM 0.9.2's NVML queries."""
import os


def patch_cuda_uuid_mapping():
    # Install before attention backend imports probe device capabilities, in
    # both the driver and spawned workers. Do not rewrite CUDA_VISIBLE_DEVICES:
    # CUDA ordinal ordering need not match NVML's physical indices.
    from vllm.platforms.interface import Platform

    original = Platform.device_id_to_physical_device_id.__func__
    if getattr(original, "_ucm_cuda_uuid_mapping", False):
        return

    def device_id_to_physical_device_id(cls, device_id):
        visible = os.environ.get(cls.device_control_env_var, "")
        if cls.device_control_env_var == "CUDA_VISIBLE_DEVICES" and visible:
            device = visible.split(",")[device_id]
            if device.startswith("GPU-"):
                from vllm.utils import import_pynvml

                nvml = import_pynvml()
                nvml.nvmlInit()
                try:
                    handle = nvml.nvmlDeviceGetHandleByUUID(device)
                    return nvml.nvmlDeviceGetIndex(handle)
                finally:
                    nvml.nvmlShutdown()
        return original(cls, device_id)

    device_id_to_physical_device_id._ucm_cuda_uuid_mapping = True
    Platform.device_id_to_physical_device_id = classmethod(device_id_to_physical_device_id)
