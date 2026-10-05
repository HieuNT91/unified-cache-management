# Cài env server mới

Linux x86_64, Python 3.10; dùng lại CUDA toolkit đã cài tại
`/home/zhufangzhou/jh/envs/cuda-12.8`. Chạy lần lượt trong cùng Bash session.
Server `noah` dùng GCC/G++ hệ thống 12.3.0. Người dùng đã xác nhận build/cài
native UCM thành công với cấu hình dưới; chưa xác nhận GPU inference.

## 1. Env Python và dependency

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y --prefix /home/zhufangzhou/jh/envs/ucm python=3.10 pip
conda activate /home/zhufangzhou/jh/envs/ucm

export CUDA_HOME=/home/zhufangzhou/jh/envs/cuda-12.8
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export NVCC_PREPEND_FLAGS="-Xcompiler=-B/usr/bin/"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
hash -r
command -v ld  # /usr/bin/ld
nvidia-smi
nvcc --version

python -m pip install --upgrade pip
python -m pip install 'setuptools==75.8.2' wheel 'cmake==3.31.6' ninja
python -m pip install \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' 'torchaudio==2.7.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'vllm==0.9.2' 'transformers==4.53.2' 'wrapt==1.17.2' \
  'numpy==2.2.6' 'scipy==1.15.3' 'nltk==3.10.3' \
  'wonderwords==3.0.1' 'matplotlib==3.10.6' PyYAML
```

Bỏ qua `conda create` nếu env đã tồn tại. Giữ `NVCC_PREPEND_FLAGS` khi build:
`nvcc` tự đưa thư mục Conda lên đầu PATH; `-Xcompiler=-B/usr/bin/` ép GCC dùng
linker hệ thống, tránh lỗi `__nptl_change_stack_perm@GLIBC_PRIVATE`.

Giữ nguyên các phiên bản trên; không cần cài riêng `flash-attn`.
Wheel CUDA theo [PyTorch](https://pytorch.org/get-started/previous-versions/) và
[vLLM 0.9.2](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/gpu.html).

## 2. Native UCM 0.3.0

Worktree này cần thư viện native từ UCM 0.3.0. Build ở source riêng:

```bash
mkdir -p /home/zhufangzhou/jh/src
git clone --depth 1 --branch v0.3.0 \
  https://github.com/ModelEngine-Group/unified-cache-management.git \
  /home/zhufangzhou/jh/src/ucm-native-0.3.0
cd /home/zhufangzhou/jh/src/ucm-native-0.3.0
test "$(git rev-parse HEAD)" = 8dd98d1eb42c60d3f22547441e948006e1c31bf3
```

Sửa đường dẫn CUDA hard-code của release cho layout Conda và chỉ rõ overload
`cudaMemcpyAsync`. Chỉ chạy một lần trong source vừa clone:

```bash
python - <<'PY'
import os
from pathlib import Path
c = Path(os.environ['CUDA_HOME'])
inc = next(p for p in (c/'include', c/'targets/x86_64-linux/include')
           if (p/'cuda_runtime.h').is_file())
lib = next(p for p in (c/'lib64', c/'targets/x86_64-linux/lib', c/'lib')
           if (p/'libcudart.so').is_file())
for name in ('ucm/shared/trans/cuda/CMakeLists.txt',
             'ucm/store/nfsstore/device/cuda/CMakeLists.txt'):
    p = Path(name)
    s = p.read_text()
    assert '"/usr/local/cuda/"' in s
    s = s.replace('"/usr/local/cuda/"', '"$ENV{CUDA_HOME}"')
    s = s.replace('${CUDA_ROOT}/include', str(inc))
    p.write_text(s.replace('${CUDA_ROOT}/lib64', str(lib)))
p = Path('ucm/store/nfsstore/device/cuda/cuda_device.cu')
s = p.read_text()
assert s.count('CUDA_API(cudaMemcpyAsync,') == 2
s = s.replace('CUDA_API(cudaMemcpyAsync,',
    'CUDA_API((static_cast<cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t)>(cudaMemcpyAsync)),')
p.write_text(s)
PY

# Stubs chỉ dùng để link, không thêm vào LD_LIBRARY_PATH.
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES='' PLATFORM=cuda ENABLE_SPARSE=FALSE \
  python -m pip install --no-build-isolation --no-deps .
```

`ENABLE_SPARSE=FALSE` chỉ áp dụng khi build native, bỏ extension của thuật toán
khác. Launcher tự bật sparse khi chạy ProphetKV. CMake cần mạng để tải dependency.

Nếu đã build lỗi trước khi đặt cấu hình bước 1, đổi tên thư mục build rồi chạy
lại lệnh `pip install` ở trên; không chạy lại patch source:

```bash
if [ -d build ]; then mv build "build.failed-$(date +%Y%m%d-%H%M%S)"; fi
```

## 3. Kiểm tra và cấu hình TP2

Tại checkout branch `prophetkv/tp2-router-data` trên server:

```bash
export PYTHONPATH="$PWD"
export PLATFORM=cuda
CUDA_VISIBLE_DEVICES='' python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import torch, ucm
for name, want in {'torch':'2.7.0', 'vllm':'0.9.2',
                   'transformers':'4.53.2', 'uc-manager':'0.3.0'}.items():
    got = version(name)
    print(name, got)
    assert got.split('+')[0] == want
assert torch.version.cuda == '12.8'
assert Path(ucm.__file__).resolve().parent == Path.cwd()/'ucm'
from ucm.store.pcstore import ucmpcstore
from ucm.shared.metrics import ucmmetrics
from ucm.shared.trans import ucmtrans
import vllm._C
print('OK: versions và native imports; chưa kiểm tra GPU inference.')
PY
python -m pip check
```

Copy `.env.a800.example` → `.env.a800` hoặc `.env.l20.example` → `.env.l20`
nếu chưa có file cấu hình. Sửa đường dẫn model/data/output và UUID server mới,
đặt `PYTHON_BIN=/home/zhufangzhou/jh/envs/ucm/bin/python`.
Ở terminal mới, activate env và chạy lại các dòng `export`/`unset` ở bước 1.
Các bước collection tiếp theo: [guide TP2](A800_LONGBENCH_DATA.md).
