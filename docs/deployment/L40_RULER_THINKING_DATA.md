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

## Bổ sung 170 mẫu/task từ run TP2 đã xong answers

Run gốc trên `noah` là **TP2**, đã có **4.680/4.680 accepted answers** và
**13/390 probes** theo report người dùng cung cấp. `EXTEND_FROM` cần receipt
`ruler/controls-complete.json`; probes còn thiếu không chặn extension. Không cần `ruler/complete.json`/`ruler-data.json` khi
configure extension, không tự tạo các receipt này. Giữ cùng TP/profile với run
cũ: `l40-tp2` dùng96% memory và automatic KV sizing, tối thiểu1.286 blocks/rank.

Sau khi code mới đã có trên remote, dùng checkout riêng để giữ code của run cũ:

```bash
OLD_PROJECT=/home/zhufangzhou/jh/projects/unified-cache-management/.worktrees/ruler-l40-tp2
git -C "$OLD_PROJECT" fetch origin
git -C "$OLD_PROJECT" worktree add --detach \
  /home/zhufangzhou/jh/projects/unified-cache-management/.worktrees/ruler-l40-extend170 \
  origin/prophetkv/tp2-router-data
cd /home/zhufangzhou/jh/projects/unified-cache-management/.worktrees/ruler-l40-extend170
```

Nếu checkout extension đã tồn tại và configure trước đó lỗi thiếu
`ruler/complete.json`, cập nhật code trong checkout đó rồi chạy lại các bước dưới;
không cần tạo lại worktree. Lỗi đó xảy ra trước khi ghi settings extension.
Không cập nhật checkout đã configure/chạy thành công; giữ source pins của nó.

```bash
export UCM_ENV_FILE="/home/zhufangzhou/jh/projects/unified-cache-management/.env.l40.thinking"
export EXTEND_FROM="/home/zhufangzhou/jh/projects/unified-cache-management/outputs/ruler-l40-tp2-thinking-30-v1"
export SAMPLES_PER_TASK=200 TP=2 SEED=42
export EXPERIMENT_DIR="$PWD/outputs/ruler-l40-tp2-thinking-extra170-v1"
export PREPARED_DIR="$PWD/inputs/ruler-l40-thinking-200-v1"
export CACHE_ROOT="$PWD/.cache/ruler-l40-tp2-thinking-extra170-v1"

bash scripts/launcher/l40_ruler_thinking_data.sh configure
bash scripts/launcher/l40_ruler_thinking_data.sh prepare
bash scripts/launcher/l40_ruler_thinking_data.sh detach
bash scripts/launcher/l40_ruler_thinking_data.sh status
```

`200` là tổng đích; settings extension ghi170 và nguồn parent riêng. `prepare`
sinh full200/task trên CPU, đối chiếu30 mẫu đầu với raw/token/layout/policy cũ,
giữ seed42/thinking16K và từ chối prompt trùng hoặc khác generator/assets/versions.
Hash generator configuration được phép khác do batch-size và đúng trường hợp
đổi import adapter đã xác minh bên dưới; token/layout/policy vẫn phải khớp. Run cũ và mọi
probe/kết quả/receipt cũ chỉ được đọc.

GPU chỉ chạy ordinal030–199: **2.210 prompts,26.520 answers,2.210 probes**;
bốn cặpTP2 nhận546/546/559/559 prompts. `status`/`report` chỉ báo phần170.
Khi extension xong:

- `ruler-data.json`: phần170 mới, luôn xuất khi extension hoàn tất.
- `ruler-data-200.json`: tự xuất nếu run30 đã hoàn tất probes và export. Nếu
  chưa, in thông báo chờ; extension170 vẫn hoàn tất, không xuất bộ200 thiếu probes.

Sau khi run cũ đã xử lý xong lỗi probes và hoàn tất export, giữ env/path extension
ở trên và chạy lệnh CPU này để gộp, không chạy lại answers/probes:

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh merge
```

File tổng hợp giữ nguyên row/hash/kết quả cũ và provenance của hai run; đủ2.600
prompts,31.200 answers và2.600 probes. Không tự resume/sửa lỗi/chạy probes của run
cũ. Không chạy hai run dùng cùng GPU đồng thời; người dùng kiểm tra availability.
Trong terminal mới, export lại các biến trước `status`, `resume`, `stop`, `merge`.

TP4 chỉ dùng cho parentTP4 riêng; không trộn TP2/TP4 trong một file tổng hợp. Export NPZ hỗ trợ từng
run, không tự gộp. JSON tổng hợp dùng được cho portable trainer, không tự train.

Các thay đổi được kiểm tra bằng fixtures CPU; chưa chạy dữ liệu thật/GPU trên
`noah`.

### Prepare lỗi vì hash adapter sau khi đã sinh đủ 2.600 prompts

Cặp hash `65d47398…` → `d616631c…` của `scripts/ruler.py` chỉ khác dòng import
`check_model` từ `run` sang `runner.preparation` (commits `cd5a3c0` → `498e9dd`).
Hàm `check_model` giữ nguyên AST. Extension cho phép đúng cặp hash đầy đủ này,
ghi mapping vào `extension.json`, rồi vẫn đối chiếu toàn bộ390 mẫu prefix và
từ chối adapter khác/token khác. Không sửa receipt hoặc input đã sinh.

Sau khi push bản sửa, tạo checkout và output mới vì run configure trước đó đã
pin code cũ. Dùng lại **PREPARED_DIR đã sinh đủ200**, không sinh lại prompts:

```bash
REPO=/home/zhufangzhou/jh/projects/unified-cache-management
git -C "$REPO" fetch origin
git -C "$REPO" worktree add --detach \
  "$REPO/.worktrees/ruler-l40-extend170-adapterfix" origin/prophetkv/tp2-router-data
cd "$REPO/.worktrees/ruler-l40-extend170-adapterfix"

export UCM_ENV_FILE="$REPO/.env.l40.thinking"
export EXTEND_FROM="$REPO/outputs/ruler-l40-tp2-thinking-30-v1"
export SAMPLES_PER_TASK=200 TP=2 SEED=42
export PREPARED_DIR="$REPO/.worktrees/ruler-l40-extend170/inputs/ruler-l40-thinking-200-v1"
export EXPERIMENT_DIR="$PWD/outputs/ruler-l40-tp2-thinking-extra170-v1"
export CACHE_ROOT="$PWD/.cache/ruler-l40-tp2-thinking-extra170-v1"

bash scripts/launcher/l40_ruler_thinking_data.sh configure
bash scripts/launcher/l40_ruler_thinking_data.sh prepare && \
  bash scripts/launcher/l40_ruler_thinking_data.sh detach
```

`prepare` đọc lại receipt/hash của bộ200 hiện có rồi tạo plan170. Nếu kiểm tra
raw/token/layout/policy tiếp theo phát hiện khác biệt, vẫn dừng trước GPU launch.
