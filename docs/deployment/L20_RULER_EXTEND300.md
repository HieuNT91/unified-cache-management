# L20 RULER non-thinking: thêm300 mẫu/task vào run200

Parent: `/data/jh/unified-cache-management/.results/ruler-l20-tp2-200-v2`.
Người dùng báo31.200/31.200 answers,2.600/2.600 probes và đã có
`ruler-features.npz`. Giữ nguyên toàn bộ run200 và NPZ này.

Giữ Qwen3-32B BF16, non-thinking, seed42, năm cặpTP2, profile `l20-tp2`
(memory0.95, automatic KV), caps NIAH128/VT30/CWE120/FWE50/QA32. Configure
resolve GPU0–9 thành UUID và yêu cầu năm cặp đúng thứ tự của parent.

`SAMPLES_PER_TASK=500` là tổng đích. `prepare` sinh batch500/task tại path mới,
đối chiếu200 mẫu đầu với parent (spec, raw, token/layout/evaluation metadata),
rồi chỉ lập plan cho ordinal200–499: **3.900 prompts,46.800 answers,3.900 probes**,
780 prompts/cặpTP2. Không đổi seed, chọn mẫu thay thế hay chạy lại200 answers cũ.
Chỉ các mapping adapter đã xác minh được chấp nhận; metadata khác vẫn báo lỗi.

Sau khi commit/push code từ máy phát triển, trên L20 tạo checkout mới:

```bash
OLD_PROJECT=/data/jh/unified-cache-management/.worktrees/tp2-router-data
git -C "$OLD_PROJECT" fetch origin
git -C "$OLD_PROJECT" worktree add --detach \
  /data/jh/unified-cache-management/.worktrees/l20-ruler-extend300 \
  origin/prophetkv/tp2-router-data
cd /data/jh/unified-cache-management/.worktrees/l20-ruler-extend300

# Xóa override cũ để launcher đọc file env của run mới.
unset UCM_ENV_FILE PROJECT_DIR PYTHON_BIN MODEL_PATH RULER_PATH EXTEND_FROM \
  EXPERIMENT_DIR PREPARED_DIR CACHE_ROOT TP GPU_DEVICES SAMPLES_PER_TASK SEED

cat > .env.l20 <<'EOF'
PROJECT_DIR=/data/jh/unified-cache-management
PYTHON_BIN=/data/jh/envs/ucm/bin/python
MODEL_PATH=/data/jh/ckpts/Qwen3-32B
RULER_PATH=$PROJECT_DIR/.cache/vendor/RULER
EXTEND_FROM=$PROJECT_DIR/.results/ruler-l20-tp2-200-v2
EXPERIMENT_DIR=$PROJECT_DIR/.results/ruler-l20-tp2-extra300-v1
PREPARED_DIR=$PROJECT_DIR/.data/prepared/ruler-l20-tp2-500-v1
CACHE_ROOT=$PROJECT_DIR/.cache/ruler-l20-tp2-extra300-v1
TP=2
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9
SAMPLES_PER_TASK=500
SEED=42
EOF

bash scripts/launcher/l20_ruler_data.sh configure && \
  bash scripts/launcher/l20_ruler_data.sh prepare && \
  bash scripts/launcher/l20_ruler_data.sh detach

bash scripts/launcher/l20_ruler_data.sh status
```

Sau configure, giữ code/env/path mới để `resume`; không pull vào checkout đã
chạy. Nếu `prepare` dừng, giữ các raw/prepared đã sinh và xử lý nguyên nhân.
Không sửa settings/spec/receipt. Run cũ vẫn ở checkout ban đầu.

Khi xong, extension tự xuất:

- `.results/ruler-l20-tp2-extra300-v1/ruler-data.json`: chỉ300 mẫu mới/task.
- `.results/ruler-l20-tp2-extra300-v1/ruler-data-500.json`: union200+300,
  **6.500 prompts,78.000 answers,6.500 probes**, giữ row/hash/provenance cũ.

Union cần parent có final receipts và `ruler-data.json` cùng checksum; NPZ không
thay các receipt này. Nếu parent export chưa xong, delta300 vẫn được giữ và
`bash scripts/launcher/l20_ruler_data.sh merge` gộp sau đó trên CPU.

Sau khi answers **và probes mới** hoàn tất, xuất NPZ cho phần300:

```bash
/data/jh/envs/ucm/bin/python scripts/export_features.py \
  /data/jh/unified-cache-management/.results/ruler-l20-tp2-extra300-v1 \
  --output /data/jh/unified-cache-management/.results/ruler-l20-tp2-extra300-v1/ruler-features.npz
```

NPZ mới có3.900 rows, giữ ID ordinal200–499. NPZ parent2.600 rows giữ nguyên;
không tự gộp NPZ hoặc train. Không suy ra kích thước file từ1,4MiB của run cũ.
Chỉ kiểm tra bằng fixtures CPU tại máy phát triển; chưa chạy dữ liệu thật/GPU.
