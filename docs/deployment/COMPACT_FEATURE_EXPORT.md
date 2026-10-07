# Export 100 attention feature thành NPZ

Script độc lập: [`scripts/export_features.py`](../../scripts/export_features.py).
Định nghĩa: [`features_list.md`](../../features_list.md), 100 scalar / 18 nhóm.

## Chạy trên A800, L20 hoặc máy CPU khác

Chỉ cần Python >=3.9 và NumPy >=1.22. Không cần model, vLLM, PyTorch, CUDA,
tokenizer, thư mục prepared gốc hay các module khác của repo. Có thể copy **riêng
file script** sang một thư mục công cụ ngoài checkout đang chạy; không cần đổi
source đã được pin của experiment. Các absolute path cũ trong metadata được
giữ làm provenance, không dùng để mở model/prepared/cache.

```bash
python scripts/export_features.py /path/to/experiment
```

Với Python trong môi trường A800 đã khai báo:

```bash
/mnt/sde/jh/envs/ucm/bin/python scripts/export_features.py \
  /mnt/sde/jh/projects/unified-cache-management/.results/longbench-v2-a800-tp2-auto95
```

Truyền đúng result root của experiment thực tế, không phải prepared dir hoặc
thư mục con `primary`/`extra`/`features`. Script không tự source `.env`.

Đầu ra mặc định, nhận diện dataset từ manifest:

```text
<experiment>/longbench-v2-features.npz
<experiment>/ruler-features.npz
```

Tùy chọn đường dẫn output khác hoặc giảm log:

```bash
python /path/to/export_features.py /path/to/experiment \
  --output /path/to/exports/ruler-200-features.npz --quiet
```

Script xử lý tuần tự từng prompt; bộ nhớ phụ thuộc một attention capture,
không phụ thuộc tổng số prompt. Các phép sorting và pairwise head-mask overlap
có thể tốn CPU/I/O; thời gian export phụ thuộc context và storage. Nếu thư viện
BLAS tự mở nhiều thread, có thể đặt `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`
cho process export.

## Experiment được hỗ trợ

- Layout collection `tp2-data-config-v2`: có `settings.json`, `plan.json`,
  `rows.json`, stage protocols/devices, committed records và completion receipts.
- LongBench v2: đủ 503 prompt, primary + extra, 12 action và 503 feature probes.
- RULER: đủ 13 × 100 hoặc 13 × 200 prompt; hỗ trợ cùng layout của RULER thinking
  13 × 30 trên L40 TP2/TP4. Metadata lưu nguyên execution/evaluation profile;
  không tự gộp thinking và non-thinking thành cùng protocol.
- Capture `coverage-five`: đủ 64 layer, đủ 64 Q head ở 8 layer đã chốt,
  và archive của mọi TP rank. Các layout benchmark lịch sử không được hỗ trợ.

Tất cả launcher phải hoàn tất và có receipt xác nhận owned engines đã thoát.
Không có chế độ partial hoặc bỏ qua method/prompt thiếu. Các run lock đang được
giữ trên hệ POSIX cũng chặn export. Khi chuyển experiment sang máy khác, giữ
nguyên cấu trúc tương đối, metadata, initialization/retirement receipts và records.
Không kiểm tra PID từ máy nguồn trên máy đích.

Exporter đọc source, không sửa source pins, không chạy inference/probe/training,
không dọn archive/cache. Output được kiểm tra rồi publish atomically; file có sẵn
không bị ghi đè. Muốn một bản export khác, dùng `--output` mới.

## Nội dung NPZ

| Trường | Shape / kiểu | Ý nghĩa |
|---|---|---|
| `X` | `[N,100]`, float32 | Thứ tự ID 001–100 trong feature list |
| `X_missing` | `[N,100]`, bool | True tại feature không xác định; giá trị X là NaN |
| `y_accuracy` | `[N,12]`, float32 | Native score từng prompt/method, 0–1 |
| `y_ttft` | `[N,12]`, float32 | TTFT control đã đo, giây |
| `feature_names`, `feature_groups` | `[100]`, Unicode | Tên và nhóm feature |
| `action_names` | `[12]`, Unicode | Thứ tự method |
| `prompt_ids`, `input_hashes`, `token_hashes` | `[N]`, Unicode | Ghép mẫu và provenance |
| `dataset` | scalar Unicode | `longbench-v2` hoặc `ruler` |
| `task`, `length_group` | `[N]`, Unicode | Nhóm báo cáo; length rỗng cho RULER |
| `source_ids`, `ordinals` | `[N]` | ID nguồn và ordinal frozen của prompt |
| `evaluation_protocol` | `[N]`, Unicode | Định danh cách đánh giá |
| `probe_overhead_seconds` | `[N]`, float32 | Overhead probe 5 feature đã đo, giữ riêng |
| `control_gpu_uuids` | `[N,12,TP]`, Unicode | GPU của từng outcome |
| `probe_gpu_uuids` | `[N,TP]`, Unicode | GPU của capture feature |
| `offline_extraction_seconds` | `[N]`, float64 | Thời gian đọc/check archive + tính feature trên CPU |
| `metadata_json` | scalar Unicode | Schema, định nghĩa, convention, source/timing provenance và checksum |

Action order cố định:

```text
nocache, prophetkv-1, prophetkv-5, prophetkv-10, prophetkv-20,
prophetkv-30, prophetkv-40, prophetkv-50, prophetkv-60,
prophetkv-70, prophetkv-80, prophetkv-90
```

Không có object array và không cần pickle. Runtime vẫn giữ 5 feature cũ;
extractor replay chúng trước khi chấp nhận 100 feature. Các phép tính dùng
float64, chỉ cast scalar khi export. Checksum nội dung arrays và metadata nằm
trong `metadata_json`; script cũng in SHA256 của toàn file NPZ khi hoàn tất.

Đọc bằng NumPy:

```python
import json
import numpy as np

with np.load("ruler-features.npz", allow_pickle=False) as data:
    X = data["X"]
    y_accuracy = data["y_accuracy"]
    y_ttft = data["y_ttft"]
    missing = data["X_missing"]
    feature_names = data["feature_names"].tolist()
    action_names = data["action_names"].tolist()
    metadata = json.loads(data["metadata_json"].item())
```

Nếu đã copy script vào Python path, có thể kiểm tra checksum bên trong và đọc:

```python
from export_features import verify_npz
arrays, metadata = verify_npz("ruler-features.npz")
```

## Kiểm chứng và giới hạn sử dụng

Exporter kiểm tra frozen metadata, completeness, input/action/GPU identity,
hash của result, initialization/retirement evidence và SHA256 attention được
dùng. Archive được kiểm tra shape/rank/head/layout, FP32 score reduction,
native masks/floor/ties và phép replay 5 feature cũ. Không đọc lại gold/tokenizer
để chấm điểm answer, và không replay toàn bộ diagnostic JSON của control.

Missing denominator/mask/variance theo đặc tả cho NaN có chủ đích; archive bị
hỏng hoặc thiếu rank là lỗi, không được đổi thành NaN để bỏ qua kiểm tra.
Script không tự impute; preprocessing khi train và online phải thống nhất.

Ba mảng chính của LongBench 503 + RULER 2600 chiếm **1.539.088 byte** trước nén,
chưa tính missing mask và metadata. Dung lượng NPZ thực tế tùy mức nén/provenance.

File phục vụ train/evaluate offline độc lập. Trainer JSON 5 feature hiện tại
chưa nhận trực tiếp NPZ 100 feature. Không fit tree hoặc chọn split khi export.
Khi chọn feature/hyperparameter, chỉ dùng training folds; báo rõ train overlap.

`y_ttft` là thời gian control cũ. `probe_overhead_seconds` là phép đo probe 5
feature cũ, không tự trở thành overhead của router 100 feature. Thời gian export
CPU cũng không phải latency online. Khi gộp server, giữ timing raw và chuẩn hóa
chi phí theo baseline từng prompt ở bước train/evaluate; báo thời gian riêng
từng dataset/hardware và không gọi lookup offline là end-to-end router latency.
