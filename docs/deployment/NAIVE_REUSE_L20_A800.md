# Naive reuse 0%: L20 RULER và A800 LongBench v2

Chỉ chạy action `naive-reuse`: tải/align KV của toàn bộ context, không chạy
attention scoring, không chọn hay sửa token context; tính suffix mới rồi decode.
Giữ original token IDs, boundaries, câu hỏi, template và output caps đã prepare.
Không chạy lại nocache/các ratio khác, không feature probes hay training.
KV construction/readiness/priming nằm ngoài answer TTFT như các control cũ.

| Server | Input có sẵn | Answers | GPU / TP2 | Memory |
|---|---|---:|---|---:|
| L20 | 13 task ×500, non-thinking | 6500 | 0/1,2/3,4/5,6/7,8/9 | 0.95 |
| A800 | Đủ503 LongBench v2, thinking cap16384 | 503 | 0/1,2/3 | 0.95 |

L20 chia1300 prompts/cặp. A800 chia252/251; một supervisor chỉ dùng GPU0,1,2,3,
không còn primary/extra chạy các method khác nhau. GPU index được resolve thành
UUID tại configure; các kiểm tra tài nguyên/ownership và equal KV blocks/rank
vẫn giữ nguyên. Chưa chạy GPU trên máy phát triển.

Sau khi commit/push code, tạo checkout mới trên từng server; giữ nguyên checkout
đã configure hoặc đang chạy experiment cũ. Chạy khi các GPU đã sẵn sàng.
Hai env template mới cần `git add -f .env.l20.naive.example .env.a800.naive.example`
khi commit vì pattern `.env.*` đang được ignore; không add các file env thật.

L20:

```bash
PROJECT_DIR=/data/jh/unified-cache-management
git -C "$PROJECT_DIR" fetch origin
git -C "$PROJECT_DIR" worktree add --detach "$PROJECT_DIR/.worktrees/l20-naive-reuse" origin/prophetkv/tp2-router-data
cd "$PROJECT_DIR/.worktrees/l20-naive-reuse"
cp .env.l20.naive.example .env.l20.naive
```

Env mẫu `.env.l20.naive.example`:

```bash
PROJECT_DIR=/data/jh/unified-cache-management
PYTHON_BIN=/data/jh/envs/ucm/bin/python
MODEL_PATH=/data/jh/ckpts/Qwen3-32B
RULER_PATH=$PROJECT_DIR/.cache/vendor/RULER
EXPERIMENT_DIR=$PROJECT_DIR/.results/ruler-l20-tp2-500-naive-v1
PREPARED_DIR=$PROJECT_DIR/.data/prepared/ruler-l20-tp2-500-v1
CACHE_ROOT=$PROJECT_DIR/.cache/ruler-l20-tp2-500-naive-v1
TP=2
GPU_DEVICES=0,1,2,3,4,5,6,7,8,9
SAMPLES_PER_TASK=500
SEED=42
```

A800:

```bash
PROJECT_DIR=/mnt/sde/jh/projects/unified-cache-management
git -C "$PROJECT_DIR" fetch origin
git -C "$PROJECT_DIR" worktree add --detach "$PROJECT_DIR/.worktrees/a800-naive-reuse" origin/prophetkv/tp2-router-data
cd "$PROJECT_DIR/.worktrees/a800-naive-reuse"
cp .env.a800.naive.example .env.a800.naive
```

Env mẫu `.env.a800.naive.example`:

```bash
PROJECT_DIR=/mnt/sde/jh/projects/unified-cache-management
PYTHON_BIN=/mnt/sde/jh/envs/ucm/bin/python
MODEL_PATH=/mnt/sde/jh/ckpts/Qwen3-32B
LONGBENCH_DATA=$PROJECT_DIR/.data/LongBench-v2/data.json
EXPERIMENT_DIR=$PROJECT_DIR/.results/longbench-v2-a800-tp2-naive-v1
PREPARED_DIR=$PROJECT_DIR/.data/prepared/longbench-v2-a800-tp2-auto95
CACHE_ROOT=$PROJECT_DIR/.cache/longbench-v2-a800-tp2-naive-v1
TP=2
GPU_DEVICES=0,1,2,3
SEED=42
```

Trong checkout mới, xóa override của run trước rồi chạy launcher tương ứng:

```bash
unset UCM_ENV_FILE PROJECT_DIR PYTHON_BIN MODEL_PATH RULER_PATH LONGBENCH_DATA \
  EXPERIMENT_DIR PREPARED_DIR CACHE_ROOT EXTEND_FROM SAMPLES_PER_TASK \
  TP SEED GPU_DEVICES GPU_PRIMARY GPU_EXTRA

# L20
bash scripts/launcher/l20_ruler_naive_reuse.sh configure &&
bash scripts/launcher/l20_ruler_naive_reuse.sh prepare &&
bash scripts/launcher/l20_ruler_naive_reuse.sh detach

# A800 (chạy trên A800 thay cho ba lệnh L20)
bash scripts/launcher/a800_longbench_naive_reuse.sh configure &&
bash scripts/launcher/a800_longbench_naive_reuse.sh prepare &&
bash scripts/launcher/a800_longbench_naive_reuse.sh detach
```

`prepare` chỉ kiểm tra receipt/hash/model/token/layout và lập plan từ files
hiện có, không sinh batch hay tokenize lại. Không cần `EXTEND_FROM`: L20 chạy
đủ500/task, gồm cả200 cũ và300 mới. Nếu muốn full cohort200 cũ thay vì500,
đổi đồng thời `PREPARED_DIR`, `SAMPLES_PER_TASK=200` và path result/cache mới.

Đổi `detach` thành `status`, `report`, `resume` hoặc `stop` để quản lý đúng run.
Resume chỉ chạy answers còn thiếu; không pull vào checkout sau configure.
Run mới lưu records dưới `ruler/records/naive-reuse/` (L20) hoặc
`primary/records/naive-reuse/` (A800). Khi xong có `final.json`, `final.csv`,
`final.md` với accuracy/TTFT/token counts theo task hoặc LongBench length.
Không xuất feature NPZ vì không thu feature probes. Reports đọc committed
results; không coi chúng là replay độc lập hoặc so speedup với baseline run cũ.
