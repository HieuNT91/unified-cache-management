# L40: timeout nạp KV (2026-10-16)

Phạm vi hiện tại chỉ xử lý nạp KV. Đã bỏ thay đổi Triton suffix attention được
đề xuất trước đó; giữ nguyên scoring, selection, attention và generation của
ProphetKV. Không triển khai importer baseline hoặc yêu cầu chạy lại no-cache.

## Điều log đã chứng minh

Trong `features-fbefd066f31045d4b37f5f0b40eccf37-group0.log`, cả hai TP2 rank
timeout ở task18 `PC::S2D`, sau120.000ms, khi prime `niah_single_2-000`.
Call chain là `Engine.begin` → `start_load_kv` → `store.load_data`/`store.wait`.
Backend trả lỗi `-50010`; connector đánh dấu các block chưa nạp thành công rồi
`PersistentBlendConnector` raise `Incomplete external cache load`. Dừng engine
là hành vi đúng khi chưa có đủ KV; không được bỏ kiểm tra này để chạy tiếp.

Code cũ submit toàn bộ context vào một tác vụ backend. Prompt này có64.512
context tokens, tương ứng1.008 blocks. Mỗi block có128 tensor pointers (64 layers
× K/V); `129024` trong log khớp1.008 ×128, không phải số tokens. Cả tác vụ chịu
một lần chờ120 giây. Lỗi xảy ra trước attention/scoring, nên đổi attention kernel
không sửa được lỗi này.

Log **chưa xác định** transfer chậm hay kẹt ở đâu: đọc storage, contention giữa
các TP2 pair, hoặc backend/device transfer. Chưa có bằng chứng OOM, shard thiếu
hay lỗi `.temp`. Hai rank timeout đồng thời gợi ý tài nguyên dùng chung, nhưng
không chứng minh ổ đĩa là nguyên nhân. Không thể khẳng định đã chữa tận gốc lỗi
server chỉ bằng tests CPU.

## Phần sửa giới hạn ở transfer

`runner/tree_profiles.py` đặt `max_load_blocks=64` cho cached `l40-tp2`.
`PersistentBlendConnector` truyền giới hạn vào store wrapper. `TrackedStore`
submit tối đa64 block/lô, chờ xong lô trước rồi mới submit lô sau. Tác vụ trong
log thành16 lô nếu hash không trùng. Mỗi block vẫn có đầy đủ128 pointers và đúng
destination slot; hash trùng được tách để không gây oversized file read.

Timeout giữ nguyên120.000ms/lô. Wait thất bại giữ task trong `pending`, không
submit tiếp, và vẫn fail incomplete load. Đây là biện pháp tránh việc toàn bộ
context lớn phải hoàn tất trong một deadline và giảm số read đang xếp hàng;
không bảo đảm sửa được backend bị kẹt hoặc storage có lỗi. TP4, A800/L20,
baseline và thuật toán ProphetKV giữ nguyên.

## Pull và resume cùng run

Theo yêu cầu ngày2026-10-16, các controller collection/corpus/router bỏ điều kiện
source hiện tại phải khớp hash lúc configure, bao gồm prepare và resume. Configure
lại cùng tham số giữ nguyên hash lịch sử. `settings.code` không bị sửa; metadata,
protocol, input/token/layout, policy, kết quả accepted và process ownership vẫn
được kiểm tra. Mỗi engine session mới ghi `execution_code` trong
`initialization.json`, bên cạnh cấu hình thực tế có `max_load_blocks=64`.

Máy local không truy cập được L40. Chạy block sau **trên L40** trong checkout mà
log đang dùng. `stop` chỉ dừng process thuộc run này, kiểm tra identity và giữ kết
quả. `pull --ff-only` dùng branch được chỉ định nên dùng được cả khi worktree đang
ở detached HEAD. Block dừng ngay nếu stop/pull lỗi; không reset hoặc ghi đè local edits.

```bash
(
set -e
REPO=/home/zhufangzhou/jh/projects/unified-cache-management
RUN="$REPO/outputs/ruler-l40-tp2-thinking-500-v3"
PY=/home/zhufangzhou/jh/envs/ucm/bin/python
cd "$REPO/.worktrees/ruler-l40-tp2-thinking-500-v3"
export CUDA_HOME=/home/zhufangzhou/jh/envs/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
export NLTK_DATA=/home/zhufangzhou/jh/nltk_data
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

"$PY" scripts/longbench_a800_control.py stop --root "$RUN" --role ruler
git pull --ff-only origin prophetkv/tp2-router-data
"$PY" scripts/longbench_a800_control.py resume --root "$RUN" --role ruler
"$PY" scripts/longbench_a800_control.py status --root "$RUN" --role ruler
)
```

Controller đọc model, prepared, cache-root, TP2 và bốn cặp GPU UUID từ metadata
của run cũ. Không cần nạp `.env` TP4/30 cũ, configure hoặc prepare lại. User vẫn
kiểm tra availability của các GPU đã gán trước resume. No-cache đã accepted được
bỏ qua; feature probes còn thiếu chạy tiếp, rồi các answer ProphetKV còn thiếu.
Inputs và seed giữ nguyên. Không xóa `records`, `validated.json`, settings, plan
hoặc prepared inputs. Nếu chỉ một phần no-cache hợp lệ thì resume chạy phần còn
thiếu; không giả định accepted cho kết quả lỗi/hỏng.

Không cần patch rời hoặc `cache_staging_recovery.py` để resume từ source hiện tại.
Helper recovery đó phục vụ bản frozen cũ và lỗi `.temp`; không phải luồng launch
hiện tại. Các bản patch attention được đề xuất trước đó đã rút lại.

Nếu từng lô64 vẫn timeout, giữ worker log và `failure.json`, đọc `cache_root` từ
settings rồi kiểm tra storage trên server:

```bash
CACHE_ROOT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_root"])' "$RUN/settings.json")
findmnt -T "$CACHE_ROOT"
df -hT "$CACHE_ROOT"
iostat -xz 1 5  # Nếu server đã có sysstat.
```

Các biến `PY`/`RUN` ở block resume nằm trong subshell; đặt lại chúng nếu chạy block
chẩn đoán riêng. Log chưa xác định storage hay backend là nguyên nhân gốc.

## Kiểm tra

Trên snapshot đúng phần commit:119 tests được gọi,116 pass;3 test không chạy được
do local thiếu vLLM/matplotlib (scheduler thật, export capture và plot report).
Cả12 test mới source update/transfer pass. Python parse theo3.10, shell syntax và
`git diff --check` pass. Không chạy SSH hoặc GPU L40.

Tests CPU cho source update kiểm tra resume bỏ qua no-cache đã accepted, giữ
nguyên bytes/mtime của metadata và kết quả, vẫn từ chối metadata/kết quả thay đổi,
cho phép reconfigure cùng tham số sau update nhưng từ chối đổi model/cohort/policy.
Các suite collection v3, TP2 và lifecycle là regression cho đường chạy này.

Tests CPU kiểm tra1.008 block ×128 pointers, thứ tự submit/wait, mọi destination
slot, hash trùng, task lỗi còn pending, dense bypass, profile L40 TP2 và truyền
cấu hình qua store wrappers. Không có GPU execution hoặc xác nhận throughput
storage/L40. Nếu từng lô64 vẫn timeout, cần bằng chứng I/O/backend của server;
không tiếp tục tăng timeout để che một transfer bị kẹt.
