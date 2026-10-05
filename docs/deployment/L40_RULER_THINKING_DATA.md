# RULER thinking trên 8×L40 (`noah`)

Launcher: `scripts/launcher/l40_ruler_thinking_data.sh`, env `.env.l40.thinking`.
Dùng hai nhóm **TP4**, GPU0–3 và4–7 theo tám UUID đã cung cấp. Mỗi nhóm nhận
195 prompts (15/task); mọi action và probe của một prompt dùng cùng nhóm.
Đây là cấu hình mới; giữ checkout/run L20 và các kết quả cũ nguyên trạng.

Giữ protocol [RULER thinking](L20_RULER_THINKING_DATA.md): Qwen3-32B BF16,
13 task ×30 mẫu, seed42; dense + ProphetKV1/5/10/20/30/40/50/60/70/80/90%.
Tổng **390 prompts, 4.680 answers, 390 probes**. Thinking cap16.384, answer cap
theo task, sampling và token gốc không đổi. Controls hoàn tất trước feature probes;
export `ruler-data.json` và checksum tự động, không tự train.

Profile `l40-tp4`: window82.304, KV cố định1.286 block/rank, memory budget90%,
YaRN4 table131.072, prefill16.384 và activation tile4.096. TP4 giảm bộ nhớ
weights/KV mỗi GPU so với TP2; không giảm context/output hay đổi precision.

## Chạy trên server

Tại checkout chứa thay đổi L40, sau khi cài [env](NEW_SERVER_ENV.md):

```bash
cd /home/zhufangzhou/jh/projects/unified-cache-management
cp .env.l40.thinking.example .env.l40.thinking
# Sửa MODEL_PATH và RULER_PATH cho model/source đã có trên server.
# Sửa PROJECT nếu dùng checkout ở vị trí khác.

bash scripts/launcher/l40_ruler_thinking_data.sh configure
bash scripts/launcher/l40_ruler_thinking_data.sh prepare
# Khi bạn đã xác định cả tám GPU trống:
bash scripts/launcher/l40_ruler_thinking_data.sh detach
bash scripts/launcher/l40_ruler_thinking_data.sh status
```

Theo yêu cầu, configure/launch **không quét nvidia-smi, VRAM hay GPU đang bận**;
người dùng tự kiểm tra. Launcher không dừng job khác hoặc chờ GPU rảnh.
UUID được nhận trực tiếp và đóng băng; chỉ chấp nhận tám UUID đầy đủ, khác nhau.
Runtime vẫn kiểm tra GPU của worker, KV, token/kết quả, cache và retirement;
vLLM vẫn thực hiện các bước khởi tạo/profiling cần thiết của engine.
Receipt ghi `hardware_preflight=user-managed`, không tuyên bố đã kiểm tra phần cứng.

Các lệnh khác:

```bash
bash scripts/launcher/l40_ruler_thinking_data.sh status_same_count
bash scripts/launcher/l40_ruler_thinking_data.sh report
bash scripts/launcher/l40_ruler_thinking_data.sh resume
bash scripts/launcher/l40_ruler_thinking_data.sh stop
```

`resume` chỉ chạy phần còn thiếu; `stop` chỉ dừng cây process thuộc run này.
Dùng đúng checkout đã configure và các đường dẫn riêng `ruler-l40-tp4-thinking-30-v1`.
Biến đã export trong shell có ưu tiên hơn `.env`; `UCM_ENV_FILE` chọn file khác.
Portable export giữ provenance TP4/L40 và bốn UUID/action; dùng importer/trainer
cùng phiên bản mới. TTFT giữa L40 TP4 và L20 TP2 thuộc các cấu hình phần cứng khác nhau.

Chưa chạy test hoặc GPU inference cho thay đổi này, theo yêu cầu người dùng.
