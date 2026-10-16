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

### Phiên mới: timeout RPC sau300 giây

Log `features-104f74782b6d4ffbbcba62dd297afe25-group2.log` và receipt xác nhận
`max_load_blocks=64` đã có hiệu lực. Prime bắt đầu05:44:14, engine báo lỗi05:49:14.
Đây là deadline300 giây của RPC `execute_model` bao ngoài cả các lô KV tuần tự
và model forward. vLLM0.9.2 đặt mặc định này trong
[`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`](https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/envs.py#L843-L846).
Bản chia lô trước đó bỏ sót việc cấu hình deadline bao ngoài này.

Launcher L40 TP2 nay mặc định RPC2400 giây; chỉnh trực tiếp trong
`.env.l40.thinking` của checkout chạy launcher:

```bash
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="2400"
```

Launcher tự đọc và export file; không cần `source` hoặc export thủ công.
Biến đã có trong shell vẫn ưu tiên hơn file theo quy tắc loader hiện có.
Mốc2400 giây cho phép16 lô ×120 giây cùng phần tính toán còn lại; đây là ngân
sách hữu hạn, không phải cam kết mọi request hoàn thành trước hạn. Hash trùng
có thể tạo thêm lô. Timeout backend vẫn120 giây/lô; lỗi KV vẫn dừng engine.
TP4 không được tự đặt mặc định mới, còn cấu hình tường minh vẫn được truyền qua.

Log mới ghi `KV load pid=... load=... batch=.../... blocks=... submit`, rồi
`complete batch_seconds=... total_seconds=...` sau khi wait thành công, kể cả
lô cuối. `all batches complete` chỉ xuất hiện khi đã wait thành công lô cuối;
`wait failed` giữ nguyên task pending và lỗi gốc. PID giúp phân biệt các worker.
Nếu cả hai rank đã báo hoàn tất tất cả lô của lần load này trước RPC timeout,
cần xem phần thực thi sau load. Nếu dừng giữa các lô, log xác định lô đang chờ.

Log hiện có chưa phân biệt transfer chậm với phần tính toán sau đó. Khoảng264
giây từ populate cuối05:39:50 đến prime05:44:14 cũng chưa xác định thời gian nằm
ở xử lý chunk cuối, retirement hay readiness. Tăng RPC budget xử lý deadline
bao ngoài; không chứng minh đã sửa nguyên nhân chạy chậm trên server.

Theo yêu cầu ngày2026-10-16, các controller collection/corpus/router bỏ điều kiện
source hiện tại phải khớp hash lúc configure, bao gồm prepare và resume. Configure
lại cùng tham số giữ nguyên hash lịch sử. `settings.code` không bị sửa; metadata,
protocol, input/token/layout, policy, kết quả accepted và process ownership vẫn
được kiểm tra. Mỗi engine session mới ghi `execution_code` trong
`initialization.json`, bên cạnh cấu hình thực tế có `max_load_blocks=64`.

Máy local không truy cập được L40. Sau khi push thay đổi, chạy block sau **trên L40** trong checkout mà
log đang dùng. `stop` chỉ dừng process thuộc run này, kiểm tra identity và giữ kết
quả. `pull --ff-only` dùng branch được chỉ định nên dùng được cả khi worktree đang
ở detached HEAD. Block dừng ngay nếu stop/pull lỗi; không reset hoặc ghi đè local edits.

```bash
(
set -e
cd /home/zhufangzhou/jh/projects/unified-cache-management/.worktrees/ruler-l40-tp2-thinking-500-v3
bash scripts/launcher/l40_ruler_thinking_data.sh stop
git pull --ff-only origin prophetkv/tp2-router-data
bash scripts/launcher/l40_ruler_thinking_data.sh resume
bash scripts/launcher/l40_ruler_thinking_data.sh status
)
```

Controller đọc model, prepared, cache-root, TP2 và bốn cặp GPU UUID từ metadata
của run cũ. Giữ `.env.l40.thinking` hiện có và thêm dòng timeout trên; không chép
đè toàn bộ file. Không cần configure hoặc prepare lại. User vẫn
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
RUN=/home/zhufangzhou/jh/projects/unified-cache-management/outputs/ruler-l40-tp2-thinking-500-v3
PY=/home/zhufangzhou/jh/envs/ucm/bin/python
CACHE_ROOT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_root"])' "$RUN/settings.json")
findmnt -T "$CACHE_ROOT"
df -hT "$CACHE_ROOT"
iostat -xz 1 5  # Nếu server đã có sysstat.
```

Log chưa xác định storage hay backend là nguyên nhân gốc.

## Kiểm tra

Phần deadline RPC/log tiến độ:34 tests CPU pass (8 transfer/launcher,6 source
update,20 TP2 collection). Test fake-clock mô phỏng16 lô đều dưới120 giây nhưng
tổng324 giây; xác nhận chỉ báo hoàn tất sau wait lô cuối và giữ task lỗi pending.
Launcher được chạy với15 tổ hợp lệnh/TP/env, kiểm tra cấu hình được process con
kế thừa. Resume vẫn giữ no-cache accepted. Shell/Python3.10 syntax và diff check
pass; chưa kiểm chứng GPU hoặc tốc độ server.

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
