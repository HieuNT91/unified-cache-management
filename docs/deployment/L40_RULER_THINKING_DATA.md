# RULER thinking trên 8×L40 (`noah`)

Launcher: `scripts/launcher/l40_ruler_thinking_data.sh`, env `.env.l40.thinking`.
Dùng bốn cặp **TP2**, GPU(0,1), (2,3), (4,5), (6,7) theo tám UUID đã cung cấp.
Chia theo ordinal modulo4: hai nhóm đầu nhận104 prompts/nhóm (8/task), hai nhóm
sau nhận91 prompts/nhóm (7/task); mọi action và probe của một prompt dùng cùng cặp.
Đây là cấu hình mới; giữ checkout/run L20 và các kết quả cũ nguyên trạng.

Giữ protocol [RULER thinking](L20_RULER_THINKING_DATA.md): Qwen3-32B BF16,
13 task ×30 mẫu, seed42; dense + ProphetKV1/5/10/20/30/40/50/60/70/80/90%.
Tổng **390 prompts, 4.680 answers, 390 probes**. Thinking cap16.384, answer cap
theo task, sampling và token gốc không đổi. Controls hoàn tất trước feature probes;
export `ruler-data.json` và checksum tự động, không tự train.

Profile `l40-tp2` đặt **`gpu_memory_utilization=0.96`**. vLLM tự cấp KV theo bộ nhớ
còn lại trong budget96%; runtime yêu cầu tối thiểu1.286 block/rank và dung lượng
hai rank bằng nhau. Window82.304, YaRN4 table131.072, prefill16.384 và activation
tile4.096 không đổi. Nếu không đủ KV, khởi tạo báo lỗi; không tự giảm context,
thinking cap hay precision. Budget96% không phải khóa độc quyền GPU.

## 1. Push code từ máy local

Các lệnh dưới commit toàn bộ thay đổi hiện có trong worktree này. Bỏ qua nếu bản
TP2/0.96 đã được push:

```bash
cd /home/thnguyen/spark/unified-cache-management
git add -A
git commit -m "Configure L40 thinking collection for TP2 with 96 percent GPU memory"
git push origin prophetkv/tp2-router-data
```

## 2. Lấy code trên `noah`

Dùng checkout riêng để giữ nguyên code/settings TP4. Chỉ tạo worktree này một lần:

```bash
cd /home/zhufangzhou/jh/projects/unified-cache-management
git fetch origin
git worktree add --detach /home/zhufangzhou/jh/projects/ruler-l40-tp2 \
  origin/prophetkv/tp2-router-data
cd /home/zhufangzhou/jh/projects/ruler-l40-tp2
```

Dùng lại env đã cài theo [installation guide](NEW_SERVER_ENV.md). Launcher dùng
đường dẫn Python tuyệt đối nên không cần activate Conda. Model/source/data NLTK
trong các bước sau là những đường dẫn đã dùng trên `noah`.

## 3. Tạo `.env.l40.thinking`

File mới dưới đây dùng lại **thư mục prepared của TP4**. Tên thư mục có `tp4`
không ảnh hưởng dữ liệu: TP không nằm trong spec chuẩn bị prompts.

```bash
# Bỏ các giá trị cũ trong shell để loader dùng đúng file env mới.
unset UCM_ENV_FILE PROJECT PYTHON_BIN MODEL_PATH RULER_PATH EXPERIMENT_DIR PREPARED_DIR CACHE_ROOT TP GPU_DEVICES SAMPLES_PER_TASK SEED

cat > .env.l40.thinking <<'EOF'
PROJECT=/home/zhufangzhou/jh/projects/ruler-l40-tp2
PYTHON_BIN=/home/zhufangzhou/jh/envs/ucm/bin/python
CUDA_HOME=/home/zhufangzhou/jh/envs/cuda-12.8
LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64"
NLTK_DATA=/home/zhufangzhou/jh/nltk_data
MODEL_PATH=/home/zhufangzhou/jh/ckpts/Qwen3-32B
RULER_PATH=/home/zhufangzhou/jh/projects/unified-cache-management/.cache/vendor/RULER
PREPARED_DIR=/home/zhufangzhou/jh/projects/unified-cache-management/inputs/ruler-l40-tp4-thinking-30-v1
EXPERIMENT_DIR="$PROJECT/outputs/ruler-l40-tp2-thinking-30-v1"
CACHE_ROOT="$PROJECT/.cache/ruler-l40-tp2-thinking-30-v1"
TP=2
GPU_DEVICES=GPU-f1523b4e-7652-3c0a-a4ce-00b6a9035eb7,GPU-0ba0b63b-10f2-8b99-022f-53776add2f98,GPU-7e88e081-b436-d6b1-1566-fef586a00bd2,GPU-0a78dc8d-9e48-ee78-6f8c-16ab702264a4,GPU-b05b518f-f9cc-5fec-dece-a622a320171f,GPU-d1b70968-82cf-fa4f-b832-e07ec6141b51,GPU-fff11353-d6c1-e416-e681-a6c345fd8c67,GPU-d570360a-f461-72cb-d8df-d9ad0da25252
SAMPLES_PER_TASK=30
SEED=42
EOF
```

Sửa các đường dẫn trên nếu thực tế khác. Giá trị0.96 đã gắn vào profile
`l40-tp2`, không cần thêm biến env. Biến tồn tại trong shell có ưu tiên hơn file.
Settings được đóng băng khi configure: chọn đúng `PREPARED_DIR` trước bước tiếp.

## 4. Configure và liên kết dữ liệu prepared

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh configure
bash scripts/launcher/l40_ruler_thinking_data.sh prepare
```

Nếu dữ liệu cũ đã hoàn tất (`preparation.json`) và spec tương thích, lệnh prepare
kiểm tra các artifact đã lưu rồi tạo plan/metadata cho run TP2 mới, **không sinh
lại prompts**. Nếu chưa hoàn tất, nó dùng lại raw batches có receipt và tiếp tục
phần còn thiếu; raw file không có receipt sẽ báo lỗi để giữ dữ liệu. Không chạy
hai tiến trình prepare cùng lúc trên thư mục dùng chung. Không sửa spec/receipt
để ép chấp nhận dữ liệu không tương thích.

## 5. Chạy và xem kết quả

Khi bạn đã xác định cả tám GPU trống:

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh detach
bash scripts/launcher/l40_ruler_thinking_data.sh status
bash scripts/launcher/l40_ruler_thinking_data.sh status_same_count
bash scripts/launcher/l40_ruler_thinking_data.sh report
```

`detach` chạy nền; có thể đóng terminal. Theo yêu cầu, configure/launch không
quét inventory, GPU bận hay VRAM. Launcher không dừng job khác hoặc chờ GPU rảnh.
Runtime vẫn giữ kiểm tra worker/KV/kết quả/cache/retirement và các bước khởi tạo
cần thiết của vLLM. Receipt ghi `hardware_preflight=user-managed`.

Khi cần chạy tiếp phần thiếu hoặc dừng đúng run này:

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh resume
bash scripts/launcher/l40_ruler_thinking_data.sh stop
```

Sau khi configure, giữ nguyên checkout để resume. Log supervisor:
`outputs/ruler-l40-tp2-thinking-30-v1/ruler/supervisor.log`.
Dataset cuối: `outputs/ruler-l40-tp2-thinking-30-v1/ruler-data.json` và `.sha256`.
Portable export giữ TP2/L40 và hai UUID/action; importer vẫn hỗ trợ TP4 cũ riêng.

Chưa chạy test hoặc GPU inference cho thay đổi này, theo yêu cầu người dùng.

## Bổ sung 170 mẫu/task sau khi run 30 đã xong

`EXTEND_FROM` trỏ tới **thư mục outputs của run 30 đã hoàn tất**, có
`ruler-data.json` và checksum. Giữ cùng TP/profile với run gốc. Seed42, thinking
16K, token gốc và TP2 memory0.96 không đổi. Không sửa settings hoặc resume run30
để đổi cohort.

Sau khi code mới đã có trên remote, tạo checkout riêng trên `noah` (đổi
`OLD_PROJECT` nếu run30 nằm ở checkout khác):

```bash
OLD_PROJECT=/home/zhufangzhou/jh/projects/ruler-l40-tp2
cd "$OLD_PROJECT"
git fetch origin
git worktree add --detach /home/zhufangzhou/jh/projects/ruler-l40-extend170 \
  origin/prophetkv/tp2-router-data
cd /home/zhufangzhou/jh/projects/ruler-l40-extend170

# Dùng Python/model/source và tám UUID của env cũ; ghi kết quả vào paths mới.
export UCM_ENV_FILE="$OLD_PROJECT/.env.l40.thinking"
export EXTEND_FROM="$OLD_PROJECT/outputs/ruler-l40-tp2-thinking-30-v1"
export SAMPLES_PER_TASK=200 TP=2 SEED=42
export EXPERIMENT_DIR="$PWD/outputs/ruler-l40-tp2-thinking-extra170-v1"
export PREPARED_DIR="$PWD/inputs/ruler-l40-thinking-200-v1"
export CACHE_ROOT="$PWD/.cache/ruler-l40-tp2-thinking-extra170-v1"

bash scripts/launcher/l40_ruler_thinking_data.sh configure
bash scripts/launcher/l40_ruler_thinking_data.sh prepare
bash scripts/launcher/l40_ruler_thinking_data.sh detach
bash scripts/launcher/l40_ruler_thinking_data.sh status
```

`SAMPLES_PER_TASK=200` là tổng đích; settings của run bổ sung ghi170 và nguồn
parent riêng. `prepare` sinh full200/task trên CPU tại path mới, đối chiếu toàn
bộ30 mẫu đầu với raw/token/layout/policy cũ và từ chối prompt trùng. Nếu khác
generator/assets/tokenizer/versions hoặc prefix, dừng trước khi có plan cho GPU;
không tự đổi seed hay thay prompt. Chỉ hash batch-size được phép khác giữa hai
prepared samples. Run cũ và các receipt cũ được đọc, không sửa hoặc copy sang
protocol mới.

GPU chỉ chạy ordinal030–199: **2.210 prompts,26.520 answers,2.210 probes**;
bốn cặp nhận546/546/559/559 prompts. `status`/`report` của run mới chỉ báo phần170.
Khi xong, xuất tự động:

- `$EXPERIMENT_DIR/ruler-data.json`: phần170 mới.
- `$EXPERIMENT_DIR/ruler-data-200.json`: đủ2.600 prompts,31.200 answers và2.600
  probes từ30+170; giữ nguyên row/hash/kết quả cũ, ghi provenance của cả hai run.

Trong terminal mới, export lại các biến trên trước `status`, `resume` hoặc
`stop`; giữ nguyên checkout sau configure. `export_features.py` hỗ trợ riêng run
170 với ordinal gốc; file NPZ không tự gộp hai run. File JSON tổng hợp dùng được
cho portable trainer, không tự train. Không trộn TP2/TP4 trong một file tổng hợp.

Chế độ mở rộng đã kiểm tra bằng fixtures CPU; chưa sinh dữ liệu thật hoặc chạy
GPU trên `noah`.
