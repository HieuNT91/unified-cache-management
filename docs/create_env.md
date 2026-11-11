# Tạo môi trường `ucm`

Linux x86_64, glibc ≥2.28, Conda, Git, Make và driver NVIDIA hoạt động
(khuyến nghị ≥560.35.05 cho [CUDA 12.6.3](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/)).
Chạy tại thư mục gốc repo; cần Internet để tải package và source native.

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

Lock giữ các phiên bản Python đã kiểm tra, gồm PyTorch 2.7.0/cu126 và vLLM 0.9.2.
Script build mới `uc-manager==0.3.0` từ source, dùng CUDA 12.6 trong Conda;
không dùng CUDA hệ thống hay `.so` từ env cũ. Native có targets 80/89/90 cho
A100/L20/H20. Không cần cài riêng FlashAttention.

Kiểm tra CPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests python -m unittest discover -s tests
```

Kiểm tra nhẹ trên **một GPU rảnh được cấp**, thay `GPU-...` bằng UUID từ `nvidia-smi -L`:

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

Trong `config/.env`, đặt `PYTHON_BIN` bằng kết quả `which python`, cùng `GPU_UUIDS`.
Trong `run.sh`, chỉnh `TP` (1/2/4), `NUMBER_GPUS`, `GPU_MEMORY_UTILIZATION` theo máy.
Không giới hạn tên GPU, không CPU offload; vLLM kiểm tra dung lượng KV thực tế.

Riêng máy hiện tại có driver kernel/userspace lệch phiên bản: giữ
`PYTHON_WRAPPER=/home/thnguyen/.local/bin/nvidia-driver-run` trong `.env`, và thêm
wrapper này trước `python` khi kiểm tra. Máy có driver đồng bộ để wrapper trống.
Môi trường đã được smoke-test trên RTX 4500 Ada; chưa chạy Qwen3-32B TP1 trên H20.
