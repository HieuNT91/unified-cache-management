# Cây F3 + che lúc tính lại: tái dùng KV theo chunk cho ProphetKV

Phương pháp chọn và tính lại token khi tái dùng KV cache của các chunk được encode độc lập (CacheBlend/ProphetKV). Không dùng dấu câu, không chọn head, không huấn luyện. Cài trên nhánh ProphetKV của UCM (vLLM 0.9.2), model Qwen3-4B-Instruct-2507.

## Bối cảnh

Prompt = chunk 0 (prefix chính xác) + các chunk ngữ cảnh, mỗi chunk được encode **một mình** rồi lưu KV ("cache cũ") + phần hậu tố tươi (câu hỏi). Lúc trả lời, chỉ một phần nhỏ token ngữ cảnh (ngân sách 15% hoặc 20%) được tính lại; phần còn lại dùng KV cũ. Phương pháp quyết định **tính lại token nào** và **khi tính lại thì đọc gì**.

## 1. Độ dính giữa các token (F3) — lúc dựng cache, ngoài TTFT

Khi encode mỗi chunk, với mỗi khe giữa token t−1 và t:

F3(t) = khối lượng attention mà các token [t, t+16) đặt lên [t−16, t), chia cho khối lượng của chúng lên toàn bộ phần đứng trước (bỏ 4 token sink đầu chunk), lấy trung bình trên **mọi head của 1/4 số lớp đầu** (Qwen3-4B: 9/36 lớp, 0–8; Qwen3-32B: 16/64 lớp, 0–15).

F3 cao: hai bên khe đọc nhau, cùng một cụm. F3 thấp: liên kết yếu, khả năng là ranh giới. (`selection.F3Recorder`, ghi lúc chunk được dựng.)

## 2. Cây phân cấp

Cây Descartes cực tiểu trên F3: gốc là toàn bộ ngữ cảnh tái dùng; mỗi đoạn được tách đôi tại khe có F3 **nhỏ nhất trong đoạn**, đệ quy tới từng token. Đầu mỗi chunk được tách trước tiên (F3 = −∞).

Không có ngưỡng: ranh giới là "khe yếu nhất trong vùng" (so sánh cục bộ), nên cùng một quy tắc cho ra câu/mục trong văn xuôi, từng từ trong danh sách, khối trong code. Kết quả là cấu trúc lồng nhau: chunk → đoạn/mục → cụm → token.

## 3. Nhu cầu: thừa hưởng mật độ

Probe: các token hậu tố từ câu hỏi tới hết prompt đọc cache cũ (với 4 token đầu mỗi chunk đã được tính lại trước, như sink thật) và cho điểm u(t) = attention trung bình (mọi lớp, mọi head) lên token t, chuẩn hoá tổng bằng 1.

Mật độ của nút A: d(A) = Σ_{t∈A} u(t) / |A|.

**need(t) = max(u(t), max_{A ∋ t} d(A))**: một token cần tươi bằng nhóm "nóng" nhất chứa nó. Ví dụ: câu hỏi nhìn mạnh vào KEY của câu kim, nên nút "câu kim" có mật độ cao; các chữ số của value tự thân có u nhỏ (câu hỏi chưa biết đáp án) nhưng thừa hưởng mật độ của câu, nên cả câu được tính lại trọn vẹn. Tính bằng ngăn xếp đơn điệu, O(N log N), ~6 ms cho 64K token (`selection.tree_need`).

## 4. Chọn

Luôn tính lại 4 token đầu mỗi chunk (S4), rồi lấy top-k theo need tới đủ ngân sách ⌊tỉ lệ × số token tái dùng⌋.

## 5. Che: chỉ đọc KV hợp lệ khi dựng biểu diễn mới

KV cũ của một chunk được encode như thể chunk đứng đầu chuỗi; đọc nó từ chunk khác mang theo thiên vị theo chunk (ví dụ chọn nhầm kim ở chunk sau). Vì vậy:

| hàng query | được đọc | bị che |
|---|---|---|
| token được tính lại t (chunk c) | chunk 0 + token đã tính lại của các chunk trước c + toàn bộ phần trước t trong chunk c | token cũ của chunk khác |
| hậu tố (câu hỏi + mở đầu câu trả lời) | chunk 0 + mọi token đã tính lại + chính hậu tố (nhân quả) | mọi token cũ |
| decode | toàn bộ cache | không che |

Token cũ trong chính chunk của t được giữ: chúng được encode đúng với tiền tố cục bộ. Decode không che vì đếm/đọc rộng (cwe, qa) cần thấy toàn bộ ngữ cảnh.

Che làm tính lại **rẻ hơn ~2.5×**: ở 64K attention là phần lớn chi phí mỗi token tính lại, và mỗi hàng chỉ còn đọc ~15K thay vì ~65K khoá. Nhờ vậy ngân sách 15–20% có TTFT thấp hơn ngân sách 10% không che.

Cài đặt (chính xác, không xấp xỉ, không dựng ma trận mặt nạ):
- hậu tố: hai lời gọi FlashAttention varlen (khoá được phép, không nhân quả; chính hậu tố, nhân quả) gộp bằng log-sum-exp (`attention.masked_suffix_attention`);
- token tính lại: mỗi chunk một batch, khoá được phép được gom theo thứ tự vị trí và vị trí query được ánh xạ về chỉ số trong dãy gom, nên mặt nạ nhân quả theo vị trí của kernel ragged đúng bằng "được phép và đứng trước"; một lời gọi kernel mỗi lớp (`attention.rmask_attention`).

## Kết quả (RULER 64K, 13 task × 100 = 1300 mẫu)

| task | dense | **15%** | **20%** | ProphetKV gốc 10% |
|---|---|---|---|---|
| cwe | .653 | .490 | .530 | .500 |
| fwe | .920 | .983 | .983 | ≈1 |
| niah_multikey_1 | 1.000 | .960 | .940 | .870 |
| niah_multikey_2 | 1.000 | .990 | .990 | .550 |
| niah_multikey_3 | 1.000 | 1.000 | 1.000 | .310 |
| niah_multiquery | .998 | .998 | 1.000 | ≈1 |
| niah_multivalue | 1.000 | .995 | .995 | .983 |
| niah_single_1/2/3 | 1/1/1 | 1/1/1 | 1/1/1 | ≈1 |
| qa_1 | .720 | .810 | .790 | .760 |
| qa_2 | .590 | .530 | .540 | .440 |
| vt | .972 | .990 | .988 | .938 |
| **TB** | **.912** | **.904** | **.904** | .794 |

Điểm chất lượng đo bằng simulator (HF) dùng cùng thuật toán; bản UCM được đối chiếu khớp với simulator. ProphetKV gốc = probe câu hỏi + top-k token, không che.

TTFT trung vị (UCM vLLM, 26 prompt RULER 64K, 1 GPU):

| | dense | 15% | 20% |
|---|---|---|---|
| RTX 4090 | 12.79 s | 2.68 s (4.8×) | 3.12 s (4.1×) |
| A5000 | 32.24 s | 5.74 s (5.6×) | 6.98 s (4.6×) |

## Chạy

```bash
PYTHON_BIN=... MODEL=<Qwen3-4B-Instruct-2507> MANIFEST=<prepared_ruler13/manifest.jsonl> \
OUT=<out> CACHE_ROOT=<cache> GPUS="0 1" PERCENTAGES="15 20" bash scripts/ruler64k_tree_rmask.sh
```

Script bật `UCM_TREE=1 UCM_S4=1 UCM_PROBE_S4=1 UCM_MASK_QUESTION=1 UCM_RMASK=1` và `--probe-query suffix`; tắt hết các biến này là ProphetKV gốc. Prompt RULER chuẩn bị bằng `scripts/ruler_prepared.py`. Chạy được với tensor parallel: mỗi rank cộng F3 trên head của mình rồi all-reduce (tổng và số head), điểm probe được all-reduce như ProphetKV gốc, tập chọn của rank 0 được broadcast. Đã kiểm TP=2 (Qwen3-4B, 2×A5000): các rank chọn giống nhau, điểm khớp TP=1; tập chọn trùng 86–95% với TP=1 do sai khác số học bf16 khi chia head. Trên máy có GPU nối qua switch PCIe, NCCL có thể treo lúc khởi tạo — đặt `NCCL_P2P_DISABLE=1`.
