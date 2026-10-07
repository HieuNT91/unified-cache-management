# Danh sách 100 attention feature cho router

Ngày: 2026-10-07. Phiên bản đặc tả đề xuất: `attention-candidates-100-v1`.

**Tổng cộng 100 feature scalar, chia thành 18 nhóm**: giữ nguyên 5 feature
đang triển khai và tên/ý nghĩa của 35 feature đã đề xuất, bổ sung 60 feature.
Đây là danh sách ứng viên để lưu và chọn bằng cross-validation, không phải
cam kết rằng dùng cả 100 feature sẽ tăng accuracy hoặc giảm latency.

Trạng thái trong bảng:

- **Đang dùng**: 5 feature hiện có trong runtime `runner/longbench_features.py`.
- **Đã bàn**: 35 feature đã đề xuất trong trao đổi trước.
- **Mới**: 60 feature bổ sung trong tài liệu này.

Cả 100 feature đã có extractor offline trong
[`scripts/export_features.py`](scripts/export_features.py). Các nhãn trên ghi
nguồn gốc danh sách; runtime probe vẫn chỉ tính 5 feature hiện tại. Xem
[hướng dẫn export compact](docs/deployment/COMPACT_FEATURE_EXPORT.md).

## Dữ liệu đầu vào và phạm vi sử dụng

Cả 100 feature đều có thể tính trên CPU từ attention archive của probe
`coverage-five`, **nếu giữ đủ các file `attention.rank*.npz` hợp lệ**:

- `scores`: score ProphetKV tổng hợp, chỉ trên vùng context đủ điều kiện.
- `layers`: attention theo token của đủ 64 layer trên mỗi rank, đã mean qua
  question queries và các Q head của rank đó; có cả first chunk.
- `heads`: attention theo từng Q head, đã mean qua question queries, tại layer
  zero-based `[7, 15, 23, 31, 39, 47, 55, 63]`; có cả first chunk.
- `boundaries`, `context_positions`, `head_layers` và rank identity.

Với TP2, ghép đủ 32 Q head/rank thành 64 Q head/layer. Các thống kê head là
trên **512 cặp layer–head**, không phải 512 head ở cùng một layer. Không lấy
trung bình heads trước khi tính quantile hoặc độ phân tán theo head.

Không cần rerun answer/probe cho danh sách này nếu các archive trên còn đủ.
File compact chỉ có 5 feature cũ không đủ để khôi phục 95 feature còn lại.
Không cần token text, task ID, gold answer, prediction hay accuracy/TTFT để
tính bất kỳ X feature nào. Task/dataset/length chỉ là metadata để chia tập và
báo cáo, không phải đầu vào router. Các feature vị trí không dùng nghĩa của
token hoặc thông tin vị trí đáp án chuẩn.

## Quy ước toán học dùng chung

1. **Vùng eligible E**: context sau first chunk P, trước fresh suffix; `N=|E|`.
   Ngoại trừ nhóm 15, mọi feature chỉ xét E. Không có attention của từng
   query riêng, từng bước decode hay của các layer/head ngoài phạm vi archive.
2. `s[i]` là score native đã lưu trên E; `p[i]=s[i]/sum(s)` là phân phối
   attention tổng hợp. Giữ nguyên score FP32 native để xếp hạng/chọn mask;
   phép cộng, chia, entropy và thống kê tiếp theo dùng float64.
3. `a_l` là vector attention layer l sau equal-head mean giữa các TP rank,
   trước khi chuẩn hóa riêng layer. `q_l=a_l[E]/sum(a_l[E])`.
   `h_j` là vector head j chuẩn hóa tương tự trên E.
4. `S_b`: top `k_b=floor(N*b/100)` token theo s, tie-break bằng vị trí tăng dần.
   Đây là mask **chung**. `C_l(b)=sum(q_l[S_b])` và
   `C_j(b)=sum(h_j[S_b])` là coverage layer/head. Không chọn lại mask cho
   từng layer/head khi tính coverage.
5. `T_b(v)` là tập top `k_b` chọn **riêng** từ v, cùng tie-break. Chỉ dùng
   các tập này khi bảng ghi rõ so sánh mask riêng, chủ yếu nhóm 13–14.
6. `g_early=normalize(mean(a_l[E], l=0..31))`,
   `g_late=normalize(mean(a_l[E], l=32..63))`. Mean trên vector chưa chuẩn hóa
   riêng layer để giữ đúng định nghĩa `group_agreement` hiện tại.
   `m_l=normalize(mean(h_j trong layer l))` dùng cho nhóm head đã chuẩn hóa.
7. `H(v)=-sum(v*ln(v))`, quy ước `0*ln(0)=0`.
   `H_norm(v)=H(v)/ln(len(v))`.
   `JS(u,v)` là Jensen–Shannon divergence dùng log cơ số 2, mixture `(u+v)/2`,
   trong [0,1]. `cos` là cosine; `J(A,B)=|A∩B|/|A∪B|`.
8. `K_t(v)` là số token nhỏ nhất, theo thứ tự attention giảm dần, để cumulative
   mass đạt ít nhất t. `K_t(v)/N` là tỷ lệ token cần giữ. Đây là ngưỡng **mass**,
   không phải cam kết giữ accuracy và không nhất thiết là một trong 12 action.
9. Vị trí eligible `x_i=i/(N-1)` chạy từ 0 đến 1; depth `d_l=l/63` cũng vậy.
   `Q_t(v)` là vị trí x đầu tiên có cumulative mass theo thứ tự **vị trí** đạt t.
   Quantile trên danh sách layer/head dùng NumPy `method='linear'`;
   standard deviation là population std (`ddof=0`).
10. `slope(z_l)` là hệ số OLS của z theo d trên đủ 64 layer, có intercept.
    Run là một đoạn **liên tiếp theo vị trí token** trong mask; attention rank
    không quyết định tính liên tiếp. Gap là số token không được chọn giữa hai run.
11. Chia context thành B bin theo `bin(i)=floor(B*i/N)`; `b_B[r]` là tổng p trong
    bin r. Nhóm bin/multiscale yêu cầu `N>=64`, tránh bin rỗng do context quá ngắn.
12. Các giá trị dưới đây chưa ép float32. Giữ semantics của 5 feature hiện có,
    rồi mới cast sang float32 khi xuất compact.

## 01. Đường cong mass theo budget — 8 feature

Đo attention tổng hợp tập trung đến đâu ở các mức recomputation khác nhau.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 001 | `top1_mass` | `sum(p[S_1])`: mass giữ được ở budget 1%. | Đã bàn |
| 002 | `top5_mass` | `sum(p[S_5])`: mass giữ được ở budget 5%. | Đã bàn |
| 003 | `top10_mass` | `sum(p[S_10])`: mass giữ được ở budget 10%. | Đã bàn |
| 004 | `top20_mass` | `sum(p[S_20])`: mass giữ được ở budget 20%. | Đang dùng |
| 005 | `top30_mass` | `sum(p[S_30])`: phần đường cong budget trung bình. | Mới |
| 006 | `top50_mass` | `sum(p[S_50])`: mass giữ được khi chọn nửa context eligible. | Đã bàn |
| 007 | `top70_mass` | `sum(p[S_70])`: mass còn thu thêm ở budget cao. | Mới |
| 008 | `top90_mass` | `sum(p[S_90])`: kiểm tra lượng mass còn lại ở đuôi thấp nhất. | Mới |

## 02. Budget cần thiết để đạt một mức mass — 6 feature

Nhìn đường cong ở chiều ngược lại: phải chọn bao nhiêu token để đạt mục tiêu mass.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 009 | `mass50_token_fraction` | `K_0.50(p)/N`: budget tối thiểu để thu nửa mass. | Mới |
| 010 | `mass70_token_fraction` | `K_0.70(p)/N`: budget để thu 70% mass. | Mới |
| 011 | `mass80_token_fraction` | `K_0.80(p)/N`: budget để thu 80% mass. | Đã bàn |
| 012 | `mass90_token_fraction` | `K_0.90(p)/N`: budget để thu 90% mass. | Đã bàn |
| 013 | `mass95_token_fraction` | `K_0.95(p)/N`: độ dài phần đuôi cần giữ thêm. | Mới |
| 014 | `mass99_token_fraction` | `K_0.99(p)/N`: mức phân tán khi muốn giữ gần toàn bộ mass. | Mới |

## 03. Hình dạng phân phối và ranh giới chọn token — 6 feature

Phân biệt attention đều, có đỉnh nhọn, hoặc có nhiều token gần bằng score nhau.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 015 | `attention_entropy_norm` | `H_norm(p)`: gần 1 là phân tán, gần 0 là tập trung. | Đã bàn |
| 016 | `effective_support_ratio` | `1/(N*sum(p²))`: tỷ lệ token hiệu dụng theo inverse concentration. | Đã bàn |
| 017 | `max_token_mass` | `max(p)`: mức chi phối của một token duy nhất. | Mới |
| 018 | `top1_peak_share` | `max(p)/sum(p[S_1])`: trong top 1%, mass có bị một token chi phối không. | Mới |
| 019 | `attention_gini` | Với v là p sắp tăng: `2*sum(i*v_i)/N-(N+1)/N`, i=1..N; đo bất bình đẳng mass. | Mới |
| 020 | `top20_boundary_gap` | Với v là p sắp giảm: `(v[k_20-1]-v[k_20])/v[k_20-1]`; độ tách score tại biên mask. Không phải phép kiểm chứng độ ổn định dưới nhiễu. | Mới |

## 04. Coverage giữa 64 layer — 9 feature

Đánh giá mask ProphetKV chung có bao phủ cả các layer ít đồng thuận hay không.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 021 | `coverage1_median` | Median của 64 giá trị `C_l(1)`. | Đã bàn |
| 022 | `coverage1_min` | Min `C_l(1)`: layer coverage thấp nhất ở budget rất nhỏ. | Đã bàn |
| 023 | `coverage5_median` | Median `C_l(5)`: coverage của layer điển hình. | Đang dùng |
| 024 | `coverage5_min` | Min `C_l(5)`: coverage thấp nhất, không bị mean che khuất. | Đang dùng |
| 025 | `coverage5_p10` | Phân vị 10% của `C_l(5)`: nhóm layer coverage thấp. | Đã bàn |
| 026 | `coverage5_std` | Std `C_l(5)`: mức không đồng đều giữa layer. | Đã bàn |
| 027 | `coverage20_median` | Median `C_l(20)`: layer điển hình ở budget lớn hơn. | Đã bàn |
| 028 | `coverage20_min` | Min `C_l(20)`: tăng budget có còn bỏ sót mass của layer nào không. | Đã bàn |
| 029 | `coverage20_std` | Std `C_l(20)`: mức không đồng đều còn lại ở budget 20%. | Đã bàn |

## 05. Coverage giữa 512 cặp layer–head — 7 feature

Giữ riêng từng head thay vì chỉ nhìn vector attention đã mean qua heads.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 030 | `head_coverage1_p10` | Phân vị 10% của 512 giá trị `C_j(1)`. | Đang dùng |
| 031 | `head_coverage1_median` | Median `C_j(1)`: coverage head điển hình. | Đã bàn |
| 032 | `head_coverage1_min` | Min `C_j(1)`: head coverage thấp nhất trong phạm vi được capture. | Đã bàn |
| 033 | `head_coverage1_std` | Std `C_j(1)`: mức chênh lệch giữa head. | Đã bàn |
| 034 | `head_coverage5_p10` | Phân vị 10% của `C_j(5)`: nhóm head coverage thấp ở budget 5%. | Đã bàn |
| 035 | `head_coverage5_min` | Min `C_j(5)`: coverage thấp nhất khi tăng budget. | Đã bàn |
| 036 | `head_coverage5_std` | Std `C_j(5)`: tăng budget có làm coverage đồng đều hơn không. | Đã bàn |

## 06. Đồng thuận giữa layer và nhóm depth — 6 feature

So sánh hình dạng attention, không chỉ tổng mass nằm trong mask.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 037 | `group_agreement` | `cos(g_early,g_late)`: đồng thuận giữa layer 0–31 và 32–63. | Đang dùng |
| 038 | `group_top20_jaccard` | `J(T_20(g_early),T_20(g_late))`: trùng nhau giữa hai mask chọn riêng. | Đã bàn |
| 039 | `group_js_divergence` | `JS(g_early,g_late)`: khác biệt phân phối mass giữa hai nửa model. | Đã bàn |
| 040 | `adjacent_layer_cosine_mean` | Mean `cos(q_l,q_(l+1))` trên 63 cặp liền nhau. | Đã bàn |
| 041 | `adjacent_layer_cosine_min` | Min cosine của 63 cặp: mức đổi attention mạnh nhất giữa hai layer liền nhau. | Đã bàn |
| 042 | `layer_global_cosine_min` | Min `cos(q_l,p)`: layer khác mean tổng hợp nhất theo cosine. | Đã bàn |

## 07. Đồng thuận giữa heads và phân phối chung — 4 feature

Phân biệt head khác hướng toàn model với head khác các head cùng layer.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 043 | `head_global_cosine_mean` | Mean `cos(h_j,p)` trên 512 cặp layer–head. | Đã bàn |
| 044 | `head_global_cosine_p10` | Phân vị 10% của `cos(h_j,p)`: nhóm head ít giống global. | Đã bàn |
| 045 | `head_global_js_mean` | Mean `JS(h_j,p)`: mức phân kỳ về mass giữa head và global. | Mới |
| 046 | `within_layer_head_js_mean` | Mean `JS(h_j,m_l)` trên 64 heads/layer, rồi mean 8 layer: khác biệt nội bộ layer. | Mới |

## 08. Xu hướng thay đổi theo depth — 6 feature

Giữ thông tin chiều biến đổi mà mean/std trên layer không thể hiện.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 047 | `coverage5_depth_slope` | `slope(C_l(5))`: coverage tăng hay giảm theo depth. | Mới |
| 048 | `coverage5_early_late_delta` | Mean `C_l(5)` ở layer 32–63 trừ mean ở 0–31. | Mới |
| 049 | `layer_entropy_depth_slope` | `slope(H_norm(q_l))`: attention có phân tán dần theo depth không. | Mới |
| 050 | `layer_entropy_early_late_delta` | Mean entropy chuẩn hóa nửa sau trừ nửa đầu. | Mới |
| 051 | `layer_position_depth_slope` | `slope(sum(x*q_l))`: trọng tâm attention dịch về đầu hay cuối context theo depth. | Mới |
| 052 | `largest_adjacent_layer_shift_position` | Với l nhỏ nhất cực đại `JS(q_l,q_(l+1))`, trả `(l+0.5)/63`; vị trí depth của bước đổi mạnh nhất. Nếu mọi JS bằng 0 thì missing. | Mới |

## 09. Vị trí và độ trải rộng theo context — 7 feature

Mô tả vị trí tương đối để dùng chung cho context có độ dài khác nhau.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 053 | `attention_position_mean` | `sum(x*p)`: trọng tâm attention. | Đã bàn |
| 054 | `attention_position_std` | `sqrt(sum(p*(x-mean_x)²))`: độ trải rộng quanh trọng tâm. | Đã bàn |
| 055 | `first_quarter_mass` | Tổng p ở bin 0 của `floor(4*i/N)`: mass trong một phần tư đầu. | Đã bàn |
| 056 | `last_quarter_mass` | Tổng p ở bin 3 của `floor(4*i/N)`: mass trong một phần tư cuối. | Đã bàn |
| 057 | `top20_span_ratio` | `(max(S_20)-min(S_20)+1)/N`: độ rộng đoạn bao trùm mask top 20%. | Đã bàn |
| 058 | `attention_position_q10` | `Q_0.10(p)`: vị trí đạt 10% cumulative mass theo thứ tự context. | Mới |
| 059 | `attention_position_q90` | `Q_0.90(p)`: vị trí đạt 90% cumulative mass theo thứ tự context. | Mới |

## 10. Độ phân mảnh của mask theo vị trí — 5 feature

Phân biệt mask thành cụm liên tiếp với mask rải rác trên context.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 060 | `top20_run_count_ratio` | Số run trong `S_20` chia `k_20`; cao nghĩa là nhiều token/cụm rời nhau. | Đã bàn |
| 061 | `top20_max_run_ratio` | Độ dài run lớn nhất của `S_20` chia `k_20`; tỷ lệ mask nằm trong cụm lớn nhất. | Đã bàn |
| 062 | `top5_run_count_ratio` | Số run trong `S_5` chia `k_5`; độ phân mảnh ở budget nhỏ. | Mới |
| 063 | `top5_span_ratio` | `(max(S_5)-min(S_5)+1)/N`: top 5% nằm gần nhau hay trải khắp context. | Mới |
| 064 | `top20_max_gap_ratio` | Gap lớn nhất giữa hai run của `S_20` chia N; bằng 0 nếu chỉ có một run. | Mới |

## 11. Độ gồ ghề và cấu trúc nhiều thang vị trí — 5 feature

Phân biệt attention thay đổi mượt theo vùng với các đỉnh nhỏ xen kẽ nhau.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 065 | `attention_total_variation` | `0.5*sum(abs(p[i+1]-p[i]))`, không nối vòng hoặc thêm endpoint 0: mức thay đổi giữa token kề nhau. | Mới |
| 066 | `attention_lag1_autocorrelation` | Pearson correlation giữa `p[:-1]` và `p[1:]`; cần cả hai vector có variance dương. | Mới |
| 067 | `local_peak_mass` | Tổng `p[i]` tại vị trí nội bộ có `p[i]>p[i-1]` và `p[i]>p[i+1]`; không tính plateau/endpoint. | Mới |
| 068 | `multiscale_entropy_gap` | `H(b_16)/ln(16) - H(b_64)/ln(64)`: chênh lệch mức phân tán ở hai độ phân giải; có thể âm. | Mới |
| 069 | `attention_high_frequency_ratio` | RFFT của `p-1/N`; tổng `abs(F[f])²` ở f>0.25 chu kỳ/token chia tổng ở f>0. Dùng bins RFFT một phía, không nhân đôi; zero power thì missing. | Mới |

## 12. Phân bố mass giữa các vùng context lớn — 5 feature

Chia E thành 16 vùng theo vị trí; bổ sung góc nhìn thô, ít phụ thuộc token đơn lẻ.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 070 | `bin16_entropy_norm` | `H(b_16)/ln(16)`: mass trải trên bao nhiêu vùng lớn. | Mới |
| 071 | `bin16_max_mass` | `max(b_16)`: vùng lớn được chú ý nhiều nhất chiếm bao nhiêu mass. | Mới |
| 072 | `bin16_top2_mass` | Tổng hai phần tử lớn nhất của `b_16`: hai vùng có chi phối không. | Mới |
| 073 | `bin16_effective_support_ratio` | `1/(16*sum(b_16²))`: tỷ lệ vùng hiệu dụng theo concentration. | Mới |
| 074 | `top20_bin16_coverage` | Số bin có ít nhất một token thuộc `S_20` chia 16; độ phủ vị trí của mask. | Mới |

## 13. Đồng thuận giữa mask riêng của layer — 6 feature

Khác coverage: so sánh tập vị trí layer tự chọn, không trọng số theo attention mass.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 075 | `layer_top5_global_jaccard_mean` | Mean `J(T_5(q_l),S_5)` trên 64 layer. | Mới |
| 076 | `layer_top5_global_jaccard_p10` | Phân vị 10% của `J(T_5(q_l),S_5)`: nhóm layer chọn khác global. | Mới |
| 077 | `layer_top20_global_jaccard_mean` | Mean `J(T_20(q_l),S_20)`: đồng thuận ở budget lớn hơn. | Mới |
| 078 | `layer_top20_global_jaccard_min` | Min `J(T_20(q_l),S_20)`: layer có mask khác global nhất. | Mới |
| 079 | `adjacent_layer_top20_jaccard_mean` | Mean `J(T_20(q_l),T_20(q_(l+1)))` trên 63 cặp layer. | Mới |
| 080 | `group_top5_jaccard` | `J(T_5(g_early),T_5(g_late))`: đồng thuận giữa hai nhóm ở budget nhỏ. | Mới |

## 14. Đồng thuận và độ phủ của mask riêng từng head — 5 feature

Đo độ đa dạng lựa chọn token giữa các head được capture.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 081 | `head_top1_global_jaccard_mean` | Mean `J(T_1(h_j),S_1)` trên 512 cặp layer–head. | Mới |
| 082 | `head_top5_global_jaccard_p10` | Phân vị 10% của `J(T_5(h_j),S_5)`: nhóm head có lựa chọn khác global. | Mới |
| 083 | `within_layer_head_top5_jaccard_mean` | Mean Jaccard trên 2.016 cặp head không thứ tự trong mỗi layer, rồi mean 8 layer; không gồm self-pair. | Mới |
| 084 | `head_top5_consensus90_fraction` | Số token thuộc `T_5(h_j)` của ít nhất `ceil(0.9*512)` cặp layer–head, chia `k_5`. Có thể lớn hơn 1, tối đa `512/ceil(0.9*512)`. | Mới |
| 085 | `head_top5_union_ratio` | Kích thước hợp của 512 mask `T_5(h_j)`, chia N: tổng độ phủ của các lựa chọn riêng. | Mới |

## 15. Attention vào first chunk được reuse — 5 feature

**Ngoại lệ phạm vi:** dùng vector đầy đủ trên P∪E, trước fresh suffix.
Không sửa mẫu số của các feature coverage nhóm 4–5. Nhóm này mô tả lượng
attention hướng vào prefix được reuse chính xác, chứ không xem prefix là
phần cần recompute.

Đặt `u=normalize(mean(a_l đầy đủ, l=0..63))`. Với layer/head, `u_l` và
`u_j` là vector attention riêng được chuẩn hóa trên P∪E. Các vector full này
được lấy từ archive, không cố khôi phục bằng `scores` vốn chỉ chứa E.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 086 | `first_chunk_mass_global` | `sum(u[P])`: tỷ lệ full-context mass nằm trong first chunk. | Mới |
| 087 | `first_chunk_mass_layer_std` | Std `sum(u_l[P])` trên 64 layer: mức khác nhau về chú ý vào prefix. | Mới |
| 088 | `first_chunk_mass_head_p90` | Phân vị 90% của `sum(u_j[P])`: nhóm head tập trung mạnh vào prefix. | Mới |
| 089 | `first_chunk_entropy_norm` | `H(normalize(u[P]))/ln(|P|)`: attention trong prefix tập trung hay phân tán. | Mới |
| 090 | `first_chunk_max_token_mass` | `max(u[P])`: mass của token prefix mạnh nhất, mẫu số vẫn là full context. | Mới |

## 16. Độ đa dạng về concentration giữa layer/head — 4 feature

Phân biệt mean attention phân tán vì mọi kênh đều phân tán, hay vì mỗi kênh
tập trung vào một nơi khác nhau. Không thể suy ra nhóm này chỉ từ entropy global.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 091 | `layer_entropy_mean` | Mean `H_norm(q_l)`: độ phân tán trung bình của từng layer. | Mới |
| 092 | `layer_entropy_std` | Std `H_norm(q_l)`: layer có khác nhau về mức tập trung không. | Mới |
| 093 | `head_entropy_mean` | Mean `H_norm(h_j)`: độ phân tán trung bình của từng head. | Mới |
| 094 | `head_entropy_std` | Std `H_norm(h_j)`: mức đa dạng concentration giữa head. | Mới |

## 17. Độ lệch vị trí của đỉnh attention giữa các kênh — 2 feature

Xét token có attention lớn nhất của từng kênh; tie-break lấy vị trí nhỏ nhất.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 095 | `layer_peak_position_std` | Std của `x[argmax(q_l)]` trên 64 layer: đỉnh attention dịch chuyển bao xa giữa layer. | Mới |
| 096 | `head_peak_position_std` | Std của `x[argmax(h_j)]` trên 512 cặp layer–head: các head chú ý cùng vùng hay khác vùng. | Mới |

## 18. Sự không đồng đều về budget cần thiết giữa các kênh — 4 feature

Mỗi layer/head tự xếp hạng attention để đạt 80% mass; không dùng mask global
cho nhóm này. Đây là độ khó nội tại của phân phối, khác coverage mask chung.

| ID | Feature | Định nghĩa và ý nghĩa | Trạng thái |
|---:|---|---|---|
| 097 | `layer_mass80_fraction_max` | Max `K_0.80(q_l)/N`: layer cần budget lớn nhất để tự giữ 80% mass. | Mới |
| 098 | `layer_mass80_fraction_std` | Std `K_0.80(q_l)/N`: budget cần thiết khác nhau thế nào giữa layer. | Mới |
| 099 | `head_mass80_fraction_p90` | Phân vị 90% của `K_0.80(h_j)/N`: nhóm head cần budget lớn. | Mới |
| 100 | `head_mass80_fraction_std` | Std `K_0.80(h_j)/N`: mức không đồng đều về budget giữa head. | Mới |

## Quy tắc tính, lưu và kiểm chứng khi triển khai extractor

- **Phân biệt dữ liệu hỏng với feature không xác định.** Archive sai checksum,
  thiếu rank, sai head/layer identity, shape sai, attention âm/nonfinite phải
  báo lỗi, không đổi thành missing để tiếp tục âm thầm.
- **Missing có định nghĩa:** denominator bằng 0, entropy trên vector dài <=1,
  mask rỗng, correlation có variance bằng 0, hoặc thiếu điều kiện tối thiểu của
  feature thì trả null (JSON), hoặc NaN kèm missing mask (NPZ). Không tự thêm
  epsilon/pseudocount vì sẽ đổi ý nghĩa feature. Nếu một layer/head không hợp lệ
  cho một thống kê nhóm, đánh dấu thống kê đó missing; không bỏ kênh để lấy mean.
- `top20_boundary_gap` yêu cầu `1<=k_20<N` và score cuối tập được chọn >0.
  Quantile cần đủ các kênh đã định nghĩa. Feature global có thể vẫn hợp lệ khi
  một feature head bị missing; không thay đổi phạm vi head tùy từng prompt.
- Dùng float64 cho reductions/statistics, **cast float32 ở bước export**.
  Train/evaluate/online cần thống nhất preprocessing, tên, thứ tự và phiên bản.
  Feature mới không được làm thay đổi phép tính của 5 feature cũ.
- Không dùng chung validator cũ ép mọi giá trị vào [0,1]: slope, delta,
  correlation và `multiscale_entropy_gap` có thể âm; consensus90 ở ID 084 có
  thể vượt 1. Ghi miền giá trị theo từng định nghĩa khi triển khai.
- 100 feature này là **scalar statistics của attention**, không phải attention
  matrices thô. Không giả định recover được attention từng query, attention
  của head tại 56 layer chưa capture, gradient, hidden state hoặc decode trace.
- Tính toán offline có thể tốn CPU/I/O. Tái sử dụng sorting, masks và cumulative
  sums; xử lý từng prompt. Nhóm 14, đặc biệt ID 083, cần chú ý chi phí pairwise
  mask intersections; không tạo tensor pairwise theo mọi token nếu có thể dùng
  bitsets. Không mặc định 100 feature có cùng overhead với 5 feature hiện tại.
- Khi train, chọn/tune feature chỉ trong training folds. Giữ thông tin dataset,
  prompt identity và input hashes để tránh leakage; tách báo cáo LongBench/RULER
  và train-overlap/held-out. Các curve features có tương quan là chủ ý để cây nông
  có ngưỡng dễ diễn giải, không phải 100 chiều độc lập thống kê.
- Accuracy/TTFT của 12 control giữ nguyên khi chỉ thay X. Chi phí probe/extractor
  mới phải đo riêng nếu dùng để ước lượng offline router cost. Muốn báo latency
  end-to-end thực tế vẫn cần đo router online.

## Gói compact dự kiến

```text
X          float32 [N_prompts, 100]
y_accuracy float32 [N_prompts, 12]
y_ttft     float32 [N_prompts, 12]   # giây, kết quả control đã đo
```

Lưu kèm feature_names/definitions/version, action_names/order, prompt ID,
input hashes, dataset/task/length metadata, missing convention/mask,
probe overhead và measurement provenance/checksum. Metadata không tính là
feature và không đưa vào X. Giữ TTFT raw, việc chuẩn hóa theo baseline làm
ở bước train/evaluate, với timing báo riêng theo dataset/hardware.

Ba mảng chứa `(100+12+12)*4 = 496 byte/prompt`, chưa nén và chưa có metadata:

| Dataset | Prompt | Byte | MB thập phân |
|---|---:|---:|---:|
| LongBench v2 | 503 | 249488 | 0.249488 |
| RULER 100 mẫu/task | 1300 | 644800 | 0.644800 |
| RULER 200 mẫu/task | 2600 | 1289600 | 1.289600 |
| LongBench + RULER 200 mẫu/task | 3103 | 1539088 | 1.539088 |

## Tình trạng tài liệu

Extractor offline `scripts/export_features.py` triển khai danh sách này và xuất
NPZ riêng có schema `attention-router-compact-v1`. Runtime, protocol, tree schema
và exporter JSON 5 feature không thay đổi. Trainer hiện tại chưa tự nhận schema
NPZ 100 feature; file này dành cho training/evaluation offline độc lập.
Giữ nguyên kết quả, archive và checksum của dataset 5 feature.
