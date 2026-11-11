# Create the `ucm` environment

Requires Linux x86_64, glibc ≥2.28, Conda, Git, Make, and a working NVIDIA driver
(≥560.35.05 recommended for [CUDA 12.6.3](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/)).
Run from the repository root. Internet access is required to download packages and native sources.

```bash
conda create -n ucm -c conda-forge python=3.10.22 pip -y
conda install -n ucm -c nvidia -c conda-forge \
  cuda-nvcc=12.6.85 cuda-cudart-dev=12.6.77 cuda-nvml-dev=12.6.77 \
  gcc_linux-64=12.4 gxx_linux-64=12.4 -y
conda activate ucm
python -m pip install -r requirements.lock.txt
python scripts/install_native.py
python -m pip check
```

The lock file pins tested Python packages, including PyTorch 2.7.0/cu126 and vLLM 0.9.2.
The installer builds `uc-manager==0.3.0` from source using Conda's CUDA 12.6,
without relying on system CUDA or `.so` files from another environment.
The native build targets architectures 80/89/90 for A100/L20/H20.
A separate FlashAttention installation is not required.

Run the CPU checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests python -m unittest discover -s tests
```

Run a smoke test on **one available GPU assigned to you**. Replace `GPU-...` with its UUID from `nvidia-smi -L`:

```bash
CUDA_VISIBLE_DEVICES=GPU-... python - <<'PY'
import torch, vllm._C
from ucm.store.pcstore import ucmpcstore
from ucm.shared.trans import ucmtrans
device = ucmtrans.Device()
device.Setup(0)
stream = device.MakeStream()
src = torch.arange(256, device='cuda', dtype=torch.float32)
dst = torch.empty(256, pin_memory=True)
torch.cuda.synchronize()
stream.DeviceToHost(src.data_ptr(), dst.data_ptr(), src.numel() * src.element_size())
torch.testing.assert_close(dst, src.cpu())
print(torch.cuda.get_device_name(0), 'OK')
PY
```

In `config/.env`, set `PYTHON_BIN` to the output of `which python` and set `GPU_UUIDS`.
Adjust `TP` (1/2/4), `NUMBER_GPUS`, and `GPU_MEMORY_UTILIZATION` in `run.sh` for your machine.
GPU models are unrestricted. CPU offload is disabled; vLLM checks actual KV capacity.

On the current host, the kernel and userspace NVIDIA driver versions differ. Keep
`PYTHON_WRAPPER=/home/thnguyen/.local/bin/nvidia-driver-run` in `.env` and prepend
this wrapper to `python` when running checks. Leave it empty on hosts with matching drivers.
The environment has been smoke-tested on RTX 4500 Ada; Qwen3-32B TP1 on H20 has not been tested.
