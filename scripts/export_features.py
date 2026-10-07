#!/usr/bin/env python3
"""Export 100 scalar attention features and 12 measured actions (CPU only).

Python >=3.9 and NumPy >=1.22. This file is standalone: copy it to any server.
Usage: python export_features.py EXPERIMENT_DIR [--output FILE.npz]
Supported: completed tp2-data-config-v2 LongBench v2 / RULER collections,
including L40 thinking TP2/TP4. No model, prepared inputs, CUDA or repo imports.
Source results are read-only. Existing output files are never overwritten.
"""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import zipfile


class ExportError(ValueError):
    """Invalid or incomplete source data; never silently export a partial cohort."""

try:
    import numpy as np
except ImportError:
    raise SystemExit("NumPy is required: install numpy>=1.22 in this Python environment.")

VERSION = "attention-candidates-100-v1"
HEAD_LAYERS = (7, 15, 23, 31, 39, 47, 55, 63)
RATIOS = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90)
ACTION_NAMES = ('nocache',) + tuple(f'prophetkv-{p}' for p in RATIOS)
ACTIONS = [dict(id='nocache', method='baseline', ratio=None)] + [
    dict(id=f'prophetkv-{p}', method='prophetkv', ratio=p/100) for p in RATIOS]
PRIMARY = ['nocache'] + [f'prophetkv-{p}' for p in (1, 5, 20, 40, 70, 90)]
EXTRA = [f'prophetkv-{p}' for p in (10, 30, 50, 60, 80)]
TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
         'vt', 'cwe', 'fwe', 'qa_1', 'qa_2')
OLD_FEATURES = ('coverage5_median', 'head_coverage1_p10', 'coverage5_min',
                'group_agreement', 'top20_mass')
NATIVE = 'native-answer-diagnostics-v1'
PROBE = 'independent-probe-replay-v1'
PROMPT_PROTOCOL = 'rpkv-original-tokens-v1'
# Embedded so copying this single script preserves names, grouping and definitions.
FEATURE_SPEC = [('top1_mass', '01. Đường cong mass theo budget', '`sum(p[S_1])`: mass giữ được ở budget 1%.'),
 ('top5_mass', '01. Đường cong mass theo budget', '`sum(p[S_5])`: mass giữ được ở budget 5%.'),
 ('top10_mass', '01. Đường cong mass theo budget', '`sum(p[S_10])`: mass giữ được ở budget 10%.'),
 ('top20_mass', '01. Đường cong mass theo budget', '`sum(p[S_20])`: mass giữ được ở budget 20%.'),
 ('top30_mass', '01. Đường cong mass theo budget', '`sum(p[S_30])`: phần đường cong budget trung bình.'),
 ('top50_mass', '01. Đường cong mass theo budget', '`sum(p[S_50])`: mass giữ được khi chọn nửa context eligible.'),
 ('top70_mass', '01. Đường cong mass theo budget', '`sum(p[S_70])`: mass còn thu thêm ở budget cao.'),
 ('top90_mass',
  '01. Đường cong mass theo budget',
  '`sum(p[S_90])`: kiểm tra lượng mass còn lại ở đuôi thấp nhất.'),
 ('mass50_token_fraction',
  '02. Budget cần thiết để đạt một mức mass',
  '`K_0.50(p)/N`: budget tối thiểu để thu nửa mass.'),
 ('mass70_token_fraction', '02. Budget cần thiết để đạt một mức mass', '`K_0.70(p)/N`: budget để thu 70% mass.'),
 ('mass80_token_fraction', '02. Budget cần thiết để đạt một mức mass', '`K_0.80(p)/N`: budget để thu 80% mass.'),
 ('mass90_token_fraction', '02. Budget cần thiết để đạt một mức mass', '`K_0.90(p)/N`: budget để thu 90% mass.'),
 ('mass95_token_fraction',
  '02. Budget cần thiết để đạt một mức mass',
  '`K_0.95(p)/N`: độ dài phần đuôi cần giữ thêm.'),
 ('mass99_token_fraction',
  '02. Budget cần thiết để đạt một mức mass',
  '`K_0.99(p)/N`: mức phân tán khi muốn giữ gần toàn bộ mass.'),
 ('attention_entropy_norm',
  '03. Hình dạng phân phối và ranh giới chọn token',
  '`H_norm(p)`: gần 1 là phân tán, gần 0 là tập trung.'),
 ('effective_support_ratio',
  '03. Hình dạng phân phối và ranh giới chọn token',
  '`1/(N*sum(p²))`: tỷ lệ token hiệu dụng theo inverse concentration.'),
 ('max_token_mass',
  '03. Hình dạng phân phối và ranh giới chọn token',
  '`max(p)`: mức chi phối của một token duy nhất.'),
 ('top1_peak_share',
  '03. Hình dạng phân phối và ranh giới chọn token',
  '`max(p)/sum(p[S_1])`: trong top 1%, mass có bị một token chi phối không.'),
 ('attention_gini',
  '03. Hình dạng phân phối và ranh giới chọn token',
  'Với v là p sắp tăng: `2*sum(i*v_i)/N-(N+1)/N`, i=1..N; đo bất bình đẳng mass.'),
 ('top20_boundary_gap',
  '03. Hình dạng phân phối và ranh giới chọn token',
  'Với v là p sắp giảm: `(v[k_20-1]-v[k_20])/v[k_20-1]`; độ tách score tại biên mask. Không phải phép kiểm chứng '
  'độ ổn định dưới nhiễu.'),
 ('coverage1_median', '04. Coverage giữa 64 layer', 'Median của 64 giá trị `C_l(1)`.'),
 ('coverage1_min', '04. Coverage giữa 64 layer', 'Min `C_l(1)`: layer coverage thấp nhất ở budget rất nhỏ.'),
 ('coverage5_median', '04. Coverage giữa 64 layer', 'Median `C_l(5)`: coverage của layer điển hình.'),
 ('coverage5_min', '04. Coverage giữa 64 layer', 'Min `C_l(5)`: coverage thấp nhất, không bị mean che khuất.'),
 ('coverage5_p10', '04. Coverage giữa 64 layer', 'Phân vị 10% của `C_l(5)`: nhóm layer coverage thấp.'),
 ('coverage5_std', '04. Coverage giữa 64 layer', 'Std `C_l(5)`: mức không đồng đều giữa layer.'),
 ('coverage20_median', '04. Coverage giữa 64 layer', 'Median `C_l(20)`: layer điển hình ở budget lớn hơn.'),
 ('coverage20_min',
  '04. Coverage giữa 64 layer',
  'Min `C_l(20)`: tăng budget có còn bỏ sót mass của layer nào không.'),
 ('coverage20_std', '04. Coverage giữa 64 layer', 'Std `C_l(20)`: mức không đồng đều còn lại ở budget 20%.'),
 ('head_coverage1_p10', '05. Coverage giữa 512 cặp layer–head', 'Phân vị 10% của 512 giá trị `C_j(1)`.'),
 ('head_coverage1_median', '05. Coverage giữa 512 cặp layer–head', 'Median `C_j(1)`: coverage head điển hình.'),
 ('head_coverage1_min',
  '05. Coverage giữa 512 cặp layer–head',
  'Min `C_j(1)`: head coverage thấp nhất trong phạm vi được capture.'),
 ('head_coverage1_std', '05. Coverage giữa 512 cặp layer–head', 'Std `C_j(1)`: mức chênh lệch giữa head.'),
 ('head_coverage5_p10',
  '05. Coverage giữa 512 cặp layer–head',
  'Phân vị 10% của `C_j(5)`: nhóm head coverage thấp ở budget 5%.'),
 ('head_coverage5_min',
  '05. Coverage giữa 512 cặp layer–head',
  'Min `C_j(5)`: coverage thấp nhất khi tăng budget.'),
 ('head_coverage5_std',
  '05. Coverage giữa 512 cặp layer–head',
  'Std `C_j(5)`: tăng budget có làm coverage đồng đều hơn không.'),
 ('group_agreement',
  '06. Đồng thuận giữa layer và nhóm depth',
  '`cos(g_early,g_late)`: đồng thuận giữa layer 0–31 và 32–63.'),
 ('group_top20_jaccard',
  '06. Đồng thuận giữa layer và nhóm depth',
  '`J(T_20(g_early),T_20(g_late))`: trùng nhau giữa hai mask chọn riêng.'),
 ('group_js_divergence',
  '06. Đồng thuận giữa layer và nhóm depth',
  '`JS(g_early,g_late)`: khác biệt phân phối mass giữa hai nửa model.'),
 ('adjacent_layer_cosine_mean',
  '06. Đồng thuận giữa layer và nhóm depth',
  'Mean `cos(q_l,q_(l+1))` trên 63 cặp liền nhau.'),
 ('adjacent_layer_cosine_min',
  '06. Đồng thuận giữa layer và nhóm depth',
  'Min cosine của 63 cặp: mức đổi attention mạnh nhất giữa hai layer liền nhau.'),
 ('layer_global_cosine_min',
  '06. Đồng thuận giữa layer và nhóm depth',
  'Min `cos(q_l,p)`: layer khác mean tổng hợp nhất theo cosine.'),
 ('head_global_cosine_mean',
  '07. Đồng thuận giữa heads và phân phối chung',
  'Mean `cos(h_j,p)` trên 512 cặp layer–head.'),
 ('head_global_cosine_p10',
  '07. Đồng thuận giữa heads và phân phối chung',
  'Phân vị 10% của `cos(h_j,p)`: nhóm head ít giống global.'),
 ('head_global_js_mean',
  '07. Đồng thuận giữa heads và phân phối chung',
  'Mean `JS(h_j,p)`: mức phân kỳ về mass giữa head và global.'),
 ('within_layer_head_js_mean',
  '07. Đồng thuận giữa heads và phân phối chung',
  'Mean `JS(h_j,m_l)` trên 64 heads/layer, rồi mean 8 layer: khác biệt nội bộ layer.'),
 ('coverage5_depth_slope',
  '08. Xu hướng thay đổi theo depth',
  '`slope(C_l(5))`: coverage tăng hay giảm theo depth.'),
 ('coverage5_early_late_delta',
  '08. Xu hướng thay đổi theo depth',
  'Mean `C_l(5)` ở layer 32–63 trừ mean ở 0–31.'),
 ('layer_entropy_depth_slope',
  '08. Xu hướng thay đổi theo depth',
  '`slope(H_norm(q_l))`: attention có phân tán dần theo depth không.'),
 ('layer_entropy_early_late_delta',
  '08. Xu hướng thay đổi theo depth',
  'Mean entropy chuẩn hóa nửa sau trừ nửa đầu.'),
 ('layer_position_depth_slope',
  '08. Xu hướng thay đổi theo depth',
  '`slope(sum(x*q_l))`: trọng tâm attention dịch về đầu hay cuối context theo depth.'),
 ('largest_adjacent_layer_shift_position',
  '08. Xu hướng thay đổi theo depth',
  'Với l nhỏ nhất cực đại `JS(q_l,q_(l+1))`, trả `(l+0.5)/63`; vị trí depth của bước đổi mạnh nhất. Nếu mọi JS '
  'bằng 0 thì missing.'),
 ('attention_position_mean', '09. Vị trí và độ trải rộng theo context', '`sum(x*p)`: trọng tâm attention.'),
 ('attention_position_std',
  '09. Vị trí và độ trải rộng theo context',
  '`sqrt(sum(p*(x-mean_x)²))`: độ trải rộng quanh trọng tâm.'),
 ('first_quarter_mass',
  '09. Vị trí và độ trải rộng theo context',
  'Tổng p ở bin 0 của `floor(4*i/N)`: mass trong một phần tư đầu.'),
 ('last_quarter_mass',
  '09. Vị trí và độ trải rộng theo context',
  'Tổng p ở bin 3 của `floor(4*i/N)`: mass trong một phần tư cuối.'),
 ('top20_span_ratio',
  '09. Vị trí và độ trải rộng theo context',
  '`(max(S_20)-min(S_20)+1)/N`: độ rộng đoạn bao trùm mask top 20%.'),
 ('attention_position_q10',
  '09. Vị trí và độ trải rộng theo context',
  '`Q_0.10(p)`: vị trí đạt 10% cumulative mass theo thứ tự context.'),
 ('attention_position_q90',
  '09. Vị trí và độ trải rộng theo context',
  '`Q_0.90(p)`: vị trí đạt 90% cumulative mass theo thứ tự context.'),
 ('top20_run_count_ratio',
  '10. Độ phân mảnh của mask theo vị trí',
  'Số run trong `S_20` chia `k_20`; cao nghĩa là nhiều token/cụm rời nhau.'),
 ('top20_max_run_ratio',
  '10. Độ phân mảnh của mask theo vị trí',
  'Độ dài run lớn nhất của `S_20` chia `k_20`; tỷ lệ mask nằm trong cụm lớn nhất.'),
 ('top5_run_count_ratio',
  '10. Độ phân mảnh của mask theo vị trí',
  'Số run trong `S_5` chia `k_5`; độ phân mảnh ở budget nhỏ.'),
 ('top5_span_ratio',
  '10. Độ phân mảnh của mask theo vị trí',
  '`(max(S_5)-min(S_5)+1)/N`: top 5% nằm gần nhau hay trải khắp context.'),
 ('top20_max_gap_ratio',
  '10. Độ phân mảnh của mask theo vị trí',
  'Gap lớn nhất giữa hai run của `S_20` chia N; bằng 0 nếu chỉ có một run.'),
 ('attention_total_variation',
  '11. Độ gồ ghề và cấu trúc nhiều thang vị trí',
  '`0.5*sum(abs(p[i+1]-p[i]))`, không nối vòng hoặc thêm endpoint 0: mức thay đổi giữa token kề nhau.'),
 ('attention_lag1_autocorrelation',
  '11. Độ gồ ghề và cấu trúc nhiều thang vị trí',
  'Pearson correlation giữa `p[:-1]` và `p[1:]`; cần cả hai vector có variance dương.'),
 ('local_peak_mass',
  '11. Độ gồ ghề và cấu trúc nhiều thang vị trí',
  'Tổng `p[i]` tại vị trí nội bộ có `p[i]>p[i-1]` và `p[i]>p[i+1]`; không tính plateau/endpoint.'),
 ('multiscale_entropy_gap',
  '11. Độ gồ ghề và cấu trúc nhiều thang vị trí',
  '`H(b_16)/ln(16) - H(b_64)/ln(64)`: chênh lệch mức phân tán ở hai độ phân giải; có thể âm.'),
 ('attention_high_frequency_ratio',
  '11. Độ gồ ghề và cấu trúc nhiều thang vị trí',
  'RFFT của `p-1/N`; tổng `abs(F[f])²` ở f>0.25 chu kỳ/token chia tổng ở f>0. Dùng bins RFFT một phía, không nhân '
  'đôi; zero power thì missing.'),
 ('bin16_entropy_norm',
  '12. Phân bố mass giữa các vùng context lớn',
  '`H(b_16)/ln(16)`: mass trải trên bao nhiêu vùng lớn.'),
 ('bin16_max_mass',
  '12. Phân bố mass giữa các vùng context lớn',
  '`max(b_16)`: vùng lớn được chú ý nhiều nhất chiếm bao nhiêu mass.'),
 ('bin16_top2_mass',
  '12. Phân bố mass giữa các vùng context lớn',
  'Tổng hai phần tử lớn nhất của `b_16`: hai vùng có chi phối không.'),
 ('bin16_effective_support_ratio',
  '12. Phân bố mass giữa các vùng context lớn',
  '`1/(16*sum(b_16²))`: tỷ lệ vùng hiệu dụng theo concentration.'),
 ('top20_bin16_coverage',
  '12. Phân bố mass giữa các vùng context lớn',
  'Số bin có ít nhất một token thuộc `S_20` chia 16; độ phủ vị trí của mask.'),
 ('layer_top5_global_jaccard_mean',
  '13. Đồng thuận giữa mask riêng của layer',
  'Mean `J(T_5(q_l),S_5)` trên 64 layer.'),
 ('layer_top5_global_jaccard_p10',
  '13. Đồng thuận giữa mask riêng của layer',
  'Phân vị 10% của `J(T_5(q_l),S_5)`: nhóm layer chọn khác global.'),
 ('layer_top20_global_jaccard_mean',
  '13. Đồng thuận giữa mask riêng của layer',
  'Mean `J(T_20(q_l),S_20)`: đồng thuận ở budget lớn hơn.'),
 ('layer_top20_global_jaccard_min',
  '13. Đồng thuận giữa mask riêng của layer',
  'Min `J(T_20(q_l),S_20)`: layer có mask khác global nhất.'),
 ('adjacent_layer_top20_jaccard_mean',
  '13. Đồng thuận giữa mask riêng của layer',
  'Mean `J(T_20(q_l),T_20(q_(l+1)))` trên 63 cặp layer.'),
 ('group_top5_jaccard',
  '13. Đồng thuận giữa mask riêng của layer',
  '`J(T_5(g_early),T_5(g_late))`: đồng thuận giữa hai nhóm ở budget nhỏ.'),
 ('head_top1_global_jaccard_mean',
  '14. Đồng thuận và độ phủ của mask riêng từng head',
  'Mean `J(T_1(h_j),S_1)` trên 512 cặp layer–head.'),
 ('head_top5_global_jaccard_p10',
  '14. Đồng thuận và độ phủ của mask riêng từng head',
  'Phân vị 10% của `J(T_5(h_j),S_5)`: nhóm head có lựa chọn khác global.'),
 ('within_layer_head_top5_jaccard_mean',
  '14. Đồng thuận và độ phủ của mask riêng từng head',
  'Mean Jaccard trên 2.016 cặp head không thứ tự trong mỗi layer, rồi mean 8 layer; không gồm self-pair.'),
 ('head_top5_consensus90_fraction',
  '14. Đồng thuận và độ phủ của mask riêng từng head',
  'Số token thuộc `T_5(h_j)` của ít nhất `ceil(0.9*512)` cặp layer–head, chia `k_5`. Có thể lớn hơn 1, tối đa '
  '`512/ceil(0.9*512)`.'),
 ('head_top5_union_ratio',
  '14. Đồng thuận và độ phủ của mask riêng từng head',
  'Kích thước hợp của 512 mask `T_5(h_j)`, chia N: tổng độ phủ của các lựa chọn riêng.'),
 ('first_chunk_mass_global',
  '15. Attention vào first chunk được reuse',
  '`sum(u[P])`: tỷ lệ full-context mass nằm trong first chunk.'),
 ('first_chunk_mass_layer_std',
  '15. Attention vào first chunk được reuse',
  'Std `sum(u_l[P])` trên 64 layer: mức khác nhau về chú ý vào prefix.'),
 ('first_chunk_mass_head_p90',
  '15. Attention vào first chunk được reuse',
  'Phân vị 90% của `sum(u_j[P])`: nhóm head tập trung mạnh vào prefix.'),
 ('first_chunk_entropy_norm',
  '15. Attention vào first chunk được reuse',
  '`H(normalize(u[P]))/ln(|P|)`: attention trong prefix tập trung hay phân tán.'),
 ('first_chunk_max_token_mass',
  '15. Attention vào first chunk được reuse',
  '`max(u[P])`: mass của token prefix mạnh nhất, mẫu số vẫn là full context.'),
 ('layer_entropy_mean',
  '16. Độ đa dạng về concentration giữa layer/head',
  'Mean `H_norm(q_l)`: độ phân tán trung bình của từng layer.'),
 ('layer_entropy_std',
  '16. Độ đa dạng về concentration giữa layer/head',
  'Std `H_norm(q_l)`: layer có khác nhau về mức tập trung không.'),
 ('head_entropy_mean',
  '16. Độ đa dạng về concentration giữa layer/head',
  'Mean `H_norm(h_j)`: độ phân tán trung bình của từng head.'),
 ('head_entropy_std',
  '16. Độ đa dạng về concentration giữa layer/head',
  'Std `H_norm(h_j)`: mức đa dạng concentration giữa head.'),
 ('layer_peak_position_std',
  '17. Độ lệch vị trí của đỉnh attention giữa các kênh',
  'Std của `x[argmax(q_l)]` trên 64 layer: đỉnh attention dịch chuyển bao xa giữa layer.'),
 ('head_peak_position_std',
  '17. Độ lệch vị trí của đỉnh attention giữa các kênh',
  'Std của `x[argmax(h_j)]` trên 512 cặp layer–head: các head chú ý cùng vùng hay khác vùng.'),
 ('layer_mass80_fraction_max',
  '18. Sự không đồng đều về budget cần thiết giữa các kênh',
  'Max `K_0.80(q_l)/N`: layer cần budget lớn nhất để tự giữ 80% mass.'),
 ('layer_mass80_fraction_std',
  '18. Sự không đồng đều về budget cần thiết giữa các kênh',
  'Std `K_0.80(q_l)/N`: budget cần thiết khác nhau thế nào giữa layer.'),
 ('head_mass80_fraction_p90',
  '18. Sự không đồng đều về budget cần thiết giữa các kênh',
  'Phân vị 90% của `K_0.80(h_j)/N`: nhóm head cần budget lớn.'),
 ('head_mass80_fraction_std',
  '18. Sự không đồng đều về budget cần thiết giữa các kênh',
  'Std `K_0.80(h_j)/N`: mức không đồng đều về budget giữa head.')]

FEATURE_NAMES = tuple(s[0] for s in FEATURE_SPEC)

LEGACY_DEFINITIONS = {'version': 'attention-coverage-five-v1',
 'head_layers': [7, 15, 23, 31, 39, 47, 55, 63],
 'masks': 'native ascending FP32 all64 mean then TP mean; floor eligible_count*ratio; ascending position ties',
 'eligible': 'context after the exact first chunk, before fresh suffix',
 'attention': 'FP32 softmax over ALL context keys, mean over complete question queries',
 'coverage': 'selected eligible mass / total eligible mass, per layer or per head; prefix excluded from both',
 'layers': 'equal-head TP mean in float64, all 64 layers',
 'heads': 'all Q heads on all TP ranks (64 total Q heads/layer) at the eight fixed zero-based layers; no head '
          'averaging before quantile',
 'quantile': 'numpy linear quantile over all selected-layer/global-head coverage values, q=0.10',
 'agreement': 'cosine of eligible attention means for layers 0-31 and 32-63',
 'missing': 'nocache; zero mass/undefined feature is null; malformed or nonfinite arrays halt'}


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    total = v.sum()
    return v / total if np.isfinite(total) and total > 0 else np.full(v.shape, np.nan)


def entropy(v):
    if len(v) <= 1 or not np.isfinite(v).all():
        return np.nan
    positive = v[v > 0]
    return float(-np.dot(positive, np.log(positive)) / np.log(len(v)))


def cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else np.nan


def js(a, b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    m = (a+b)/2
    def part(v):
        use = v > 0
        return np.dot(v[use], np.log2(v[use]/m[use]))
    return float((part(a)+part(b))/2)


def ranked(v):
    return np.argsort(-v, kind='stable')


def mask(order, percent):
    # Same arithmetic as native selection: floor(N * (percent / 100)).
    return np.sort(order[:math.floor(len(order)*(percent/100))])


def jaccard(a, b):
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = len(a)+len(b)-intersection
    return intersection/union if union else np.nan


def mass_fraction(v, order, target):
    if not np.isfinite(v).all():
        return np.nan
    cumulative = np.cumsum(v[order], dtype=np.float64)
    return min(len(v), int(np.searchsorted(cumulative, target, side='left'))+1)/len(v)


def quantile(v, q):
    return float(np.quantile(v, q, method='linear'))


def coverage(raw, indices):
    total = np.sum(raw, axis=-1, dtype=np.float64)
    selected = np.sum(raw[..., indices], axis=-1, dtype=np.float64)
    if not len(indices):
        return np.full(np.shape(total), np.nan)
    return np.divide(selected, total, out=np.full(np.shape(total), np.nan), where=total > 0)


def compute_features(layer_full, head_full, scores, prefix):
    """One prompt at a time; heads remain FP32 and are normalized one at a time.

    layer_full: float64 equal-rank means [64, end].
    head_full: FP32 [8, 64, end], assembled by global head identity.
    scores: native FP32 [end-prefix]. No token text or outcome inputs.
    """
    end = layer_full.shape[-1]
    if (layer_full.shape != (64, end) or head_full.shape != (8, 64, end)
            or scores.shape != (end-prefix,) or not 0 < prefix < end):
        raise ExportError('Invalid feature input shapes/boundaries')
    for a in (layer_full, head_full, scores):
        if not np.isfinite(a).all() or (a < 0).any():
            raise ExportError('Malformed/nonfinite/negative attention')
    f = dict.fromkeys(FEATURE_NAMES, np.nan)
    raw = layer_full[:, prefix:]
    n = len(scores)
    p = normalize(scores)
    order = ranked(scores)
    masks = {b: mask(order, b) for b in RATIOS}
    x = np.linspace(0., 1., n) if n > 1 else np.full(n, np.nan)
    for b in (1, 5, 10, 20, 30, 50, 70, 90):
        if len(masks[b]):
            f[f'top{b}_mass'] = coverage(scores, masks[b]).item()
    for t in (50, 70, 80, 90, 95, 99):
        f[f'mass{t}_token_fraction'] = mass_fraction(p, order, t/100)
    f['attention_entropy_norm'] = entropy(p)
    if np.isfinite(p).all():
        f['effective_support_ratio'] = 1/(n*np.dot(p, p))
        f['max_token_mass'] = p.max()
        if len(masks[1]):
            f['top1_peak_share'] = p.max()/p[masks[1]].sum()
        f['attention_gini'] = 2*np.dot(np.arange(1, n+1), p[order[::-1]])/n-(n+1)/n
        k = len(masks[20])
        if 0 < k < n and p[order[k-1]] > 0:
            f['top20_boundary_gap'] = (p[order[k-1]]-p[order[k]])/p[order[k-1]]
        position_mean = np.dot(x, p)
        f['attention_position_mean'] = position_mean
        f['attention_position_std'] = np.sqrt(np.dot(p, (x-position_mean)**2))
        quarters = np.minimum(3, np.arange(n)*4//n)
        f['first_quarter_mass'] = p[quarters == 0].sum()
        f['last_quarter_mass'] = p[quarters == 3].sum()
        for t in (10, 90):
            i = min(n-1, int(np.searchsorted(np.cumsum(p), t/100)))
            f[f'attention_position_q{t}'] = x[i]
        f['attention_total_variation'] = .5*np.abs(np.diff(p)).sum()
        if n > 2:
            left, right = p[:-1]-p[:-1].mean(), p[1:]-p[1:].mean()
            f['attention_lag1_autocorrelation'] = cosine(left, right)
            peaks = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
            f['local_peak_mass'] = p[1:-1][peaks].sum()
        power = np.abs(np.fft.rfft(p-1/n))**2
        if power[1:].sum() > 0:
            f['attention_high_frequency_ratio'] = power[np.fft.rfftfreq(n) > .25].sum()/power[1:].sum()
        if n >= 64:
            bins = {b: np.bincount(np.arange(n)*b//n, weights=p, minlength=b) for b in (16, 64)}
            b = bins[16]
            f['bin16_entropy_norm'] = entropy(b)
            f['bin16_max_mass'] = b.max()
            f['bin16_top2_mass'] = np.sort(b)[-2:].sum()
            f['bin16_effective_support_ratio'] = 1/(16*np.dot(b, b))
            f['multiscale_entropy_gap'] = entropy(b)-entropy(bins[64])
            if len(masks[20]):
                f['top20_bin16_coverage'] = len(np.unique(masks[20]*16//n))/16
    for b in (5, 20):
        selected = masks[b]
        if len(selected):
            differences = np.diff(selected)
            cuts = np.flatnonzero(differences > 1)+1
            lengths = np.diff(np.r_[0, cuts, len(selected)])
            f[f'top{b}_span_ratio'] = (selected[-1]-selected[0]+1)/n
            f[f'top{b}_run_count_ratio'] = len(lengths)/len(selected)
            if b == 20:
                f['top20_max_run_ratio'] = lengths.max()/len(selected)
                f['top20_max_gap_ratio'] = max(0, int(differences.max())-1)/n if len(differences) else 0.
    layer_c = {b: coverage(raw, masks[b]) for b in (1, 5, 20)}
    for b in (1, 5, 20):
        f[f'coverage{b}_median'] = np.median(layer_c[b])
        f[f'coverage{b}_min'] = layer_c[b].min()
    for b in (5, 20):
        f[f'coverage{b}_std'] = layer_c[b].std()
    f['coverage5_p10'] = quantile(layer_c[5], .1)
    q = np.array([normalize(v) for v in raw])
    early, late = normalize(raw[:32].mean(0)), normalize(raw[32:].mean(0))
    # Compute the legacy cosine on raw group means, preserving its arithmetic.
    f['group_agreement'] = cosine(raw[:32].mean(0), raw[32:].mean(0))
    f['group_js_divergence'] = js(early, late)
    if np.isfinite(early).all() and np.isfinite(late).all():
        for b in (5, 20):
            f[f'group_top{b}_jaccard'] = jaccard(mask(ranked(early), b), mask(ranked(late), b))
    adj_cos = [cosine(q[l], q[l+1]) for l in range(63)]
    adj_js = np.array([js(q[l], q[l+1]) for l in range(63)])
    f['adjacent_layer_cosine_mean'] = np.mean(adj_cos)
    f['adjacent_layer_cosine_min'] = np.min(adj_cos)
    f['layer_global_cosine_min'] = np.min([cosine(v, p) for v in q])
    layer_entropy = np.array([entropy(v) for v in q])
    layer_positions = q@x
    d = np.linspace(0., 1., 64)-.5
    def slope(z):
        return np.dot(d, z)/np.dot(d, d)
    f['coverage5_depth_slope'] = slope(layer_c[5])
    f['coverage5_early_late_delta'] = layer_c[5][32:].mean()-layer_c[5][:32].mean()
    f['layer_entropy_depth_slope'] = slope(layer_entropy)
    f['layer_entropy_early_late_delta'] = layer_entropy[32:].mean()-layer_entropy[:32].mean()
    f['layer_position_depth_slope'] = slope(layer_positions)
    if np.isfinite(adj_js).all() and adj_js.max() > 0:
        f['largest_adjacent_layer_shift_position'] = (np.argmax(adj_js)+.5)/63
    f['layer_entropy_mean'], f['layer_entropy_std'] = layer_entropy.mean(), layer_entropy.std()
    own = {5: [], 20: []}
    fractions, peaks = [], []
    for v in q:
        valid = np.isfinite(v).all()
        r = ranked(v)
        for b in own:
            own[b].append(mask(r, b) if valid else None)
        fractions.append(mass_fraction(v, r, .8))
        peaks.append(x[r[0]] if valid else np.nan)
    f['layer_peak_position_std'] = np.std(peaks)
    f['layer_mass80_fraction_max'] = np.max(fractions)
    f['layer_mass80_fraction_std'] = np.std(fractions)
    for b in own:
        values = [jaccard(v, masks[b]) if v is not None else np.nan for v in own[b]]
        f[f'layer_top{b}_global_jaccard_mean'] = np.mean(values)
        f[f'layer_top{b}_global_jaccard_'+('p10' if b == 5 else 'min')] = quantile(values, .1) if b == 5 else np.min(values)
    f['adjacent_layer_top20_jaccard_mean'] = np.mean([
        jaccard(a, b) if a is not None and b is not None else np.nan
        for a, b in zip(own[20][:-1], own[20][1:])])
    u = normalize(layer_full.mean(0))
    f['first_chunk_mass_global'] = u[:prefix].sum()
    f['first_chunk_entropy_norm'] = entropy(normalize(u[:prefix]))
    f['first_chunk_max_token_mass'] = u[:prefix].max()
    f['first_chunk_mass_layer_std'] = np.std([normalize(v)[:prefix].sum() for v in layer_full])
    hc = {1: [], 5: []}
    hcos, hjs, within_js, hent, hpeak, hfrac, hpref = [], [], [], [], [], [], []
    hj1, hj5, pairwise = [], [], []
    votes = np.zeros(n, dtype=np.int16)
    all_heads_valid = True
    bitcounts = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    for group in head_full:
        # At most one [64, N] normalized group; no [512, N] float64 copy.
        normalized = np.array([normalize(v[prefix:]) for v in group])
        mean_head = normalize(normalized.mean(0))
        packed = []
        for full, v in zip(group, normalized):
            valid = np.isfinite(v).all()
            all_heads_valid &= valid
            raw_head = full[prefix:].astype(np.float64)
            for b in hc:
                hc[b].append(coverage(raw_head, masks[b]).item())
            hpref.append(normalize(full)[:prefix].sum())
            hcos.append(cosine(v, p)); hjs.append(js(v, p)); within_js.append(js(v, mean_head))
            hent.append(entropy(v))
            r = ranked(v)
            hpeak.append(x[r[0]] if valid else np.nan)
            hfrac.append(mass_fraction(v, r, .8))
            own1, own5 = mask(r, 1), mask(r, 5)
            hj1.append(jaccard(own1, masks[1]) if valid else np.nan)
            hj5.append(jaccard(own5, masks[5]) if valid else np.nan)
            votes[own5] += 1
            bitmap = np.zeros(n, dtype=np.uint8); bitmap[own5] = 1
            packed.append(np.packbits(bitmap))
        packed = np.asarray(packed)
        if np.isfinite(normalized).all() and len(masks[5]):
            # Exact intersections of bitsets, bounded by 64 heads in one layer.
            for i in range(63):
                intersection = bitcounts[np.bitwise_and(packed[i+1:], packed[i])].sum(axis=1)
                pairwise.extend(intersection/(2*len(masks[5])-intersection))
        else:
            pairwise.append(np.nan)
    for b in hc:
        f[f'head_coverage{b}_min'] = np.min(hc[b])
        f[f'head_coverage{b}_std'] = np.std(hc[b])
        f[f'head_coverage{b}_p10'] = quantile(hc[b], .1)
    f['head_coverage1_median'] = np.median(hc[1])
    f['head_global_cosine_mean'], f['head_global_cosine_p10'] = np.mean(hcos), quantile(hcos, .1)
    f['head_global_js_mean'], f['within_layer_head_js_mean'] = np.mean(hjs), np.mean(within_js)
    f['head_top1_global_jaccard_mean'], f['head_top5_global_jaccard_p10'] = np.mean(hj1), quantile(hj5, .1)
    f['within_layer_head_top5_jaccard_mean'] = np.mean(pairwise)
    if all_heads_valid and len(masks[5]):
        f['head_top5_consensus90_fraction'] = np.count_nonzero(votes >= math.ceil(.9*512))/len(masks[5])
        f['head_top5_union_ratio'] = np.count_nonzero(votes)/n
    f['first_chunk_mass_head_p90'] = quantile(hpref, .9)
    f['head_entropy_mean'], f['head_entropy_std'] = np.mean(hent), np.std(hent)
    f['head_peak_position_std'] = np.std(hpeak)
    f['head_mass80_fraction_p90'], f['head_mass80_fraction_std'] = quantile(hfrac, .9), np.std(hfrac)
    if set(f) != set(FEATURE_NAMES):
        raise AssertionError('Feature implementation and schema differ')
    values = np.array([f[name] for name in FEATURE_NAMES], dtype=np.float64)
    if np.isinf(values).any():
        raise ExportError('Unexpected infinite feature')
    return values


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ExportError(message)


def safe_path(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and bool(p.parts), f'Unsafe artifact path: {name}')
    result = root/p
    require(result.resolve().is_relative_to(root.resolve()), f'Artifact escapes directory: {name}')
    return result


def identity(protocol):
    return digest({k: v for k, v in protocol.items() if k not in ('prepared', 'model', 'cache_root')})


def number(value, lower=0., upper=float('inf'), positive=False):
    return (type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper
            and (not positive or value > 0))


class Collection:
    """Read pinned, committed records without importing the experiment runtime."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.pins = {}
        self.record_digest = hashlib.sha256()
        self.settings = self.read('settings.json')
        self.plan = self.read('plan.json')
        self.rows = self.read('rows.json')
        settings, plan, rows = self.settings, self.plan, self.rows
        require(settings.get('schema') == 'tp2-data-config-v2' and plan.get('schema') == 'tp2-data-plan-v2',
                'Unsupported experiment layout: expected tp2-data-config-v2 / tp2-data-plan-v2')
        for name in ('settings', 'rows'):
            require(self.pins[f'{name}.json'] == plan[f'{name}_sha256'], f'Frozen {name} checksum mismatch')
        self.dataset = settings['dataset']
        self.tp = settings['tp']
        require(self.dataset in ('ruler', 'longbench-v2') and type(self.tp) is int and self.tp in (2, 4),
                'Unsupported dataset/TP profile')
        require(settings.get('seed') == 42, 'Unexpected collection seed')
        self.schedule = ({'ruler': list(ACTION_NAMES), 'features': ['probe']} if self.dataset == 'ruler'
                         else {'primary': PRIMARY, 'extra': EXTRA, 'features': ['probe']})
        require(plan['actions'] == ACTIONS and plan['schedule'] == self.schedule
                and plan['feature_profile'] == LEGACY_DEFINITIONS, 'Unsupported action/feature protocol')
        require(isinstance(rows, list) and bool(rows), 'Empty collection')
        ids = [r['id'] for r in rows]
        require(len(set(ids)) == len(ids) and all(re.fullmatch('[a-zA-Z0-9_-]+', s) for s in ids),
                'Duplicate or invalid prompt IDs')
        require(all(r['dataset'] == self.dataset and type(r['ordinal']) is int and r['ordinal'] >= 0
                    and re.fullmatch('[0-9a-f]{64}', r['sha256']) and r.get('subtask') for r in rows),
                'Invalid row identity/provenance')
        if self.dataset == 'longbench-v2':
            require(len(rows) == 503 and len({r['source_id'] for r in rows}) == 503
                    and {r['ordinal'] for r in rows} == set(range(503))
                    and all(r.get('length') in ('short', 'medium', 'long') for r in rows),
                    'LongBench requires all 503 unique prompts and official length groups')
        else:
            count = settings.get('samples_per_task', 100)
            allowed = (30,) if settings.get('execution_profile') == 'ruler-thinking' else (100, 200)
            require(type(count) is int and count in allowed, 'Unsupported RULER samples_per_task')
            require(Counter(r['subtask'] for r in rows) == Counter({t: count for t in TASKS})
                    and all(sorted(r['ordinal'] for r in rows if r['subtask'] == t) == list(range(count)) for t in TASKS),
                    f'RULER requires all 13 tasks x {count} unique ordinals')
        require(plan['samples'] == len(rows) and plan['answers'] == len(rows)*12 and plan['probes'] == len(rows),
                'Plan counts differ from dataset membership')
        self.protocols = {}
        for role, cases in self.schedule.items():
            proto = self.read(f'{role}/protocol.json')
            device_role = ('ruler' if self.dataset == 'ruler' else 'primary') if role == 'features' else role
            devices = self.read(f'{device_role}/devices.json')
            groups = devices['groups']
            flat = [u for group in groups for u in group]
            require(bool(groups) and all(len(g) == self.tp for g in groups) and len(set(flat)) == len(flat)
                    and all(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', u) for u in flat),
                    f'Invalid GPU group identity: {role}')
            require(proto.get('schema') == 'tp2-data-stage-v2' and proto['dataset'] == self.dataset
                    and proto['tp'] == self.tp and proto['plan_sha256'] == self.pins['plan.json']
                    and proto['actions'] == ACTIONS and proto['scheduled_actions'] == cases and proto['groups'] == groups
                    and proto.get('answer_validation') == (PROBE if role == 'features' else NATIVE),
                    f'Invalid stage protocol: {role}')
            if role == 'features':
                require(proto.get('feature_profile') == LEGACY_DEFINITIONS, 'Unsupported capture profile: need coverage-five')
            for key in ('execution_profile', 'evaluation_protocol'):
                require(proto.get(key) == settings.get(key), f'Stage {key} mismatch: {role}')
            self.protocols[role] = proto
            marker = 'complete.json' if role == 'features' else 'controls-complete.json'
            completion = self.read(f'{role}/{marker}')
            require(completion.get('owned_engines_exited') is True
                    and completion['protocol_sha256'] == self.pins[f'{role}/protocol.json']
                    and completion['probes' if role == 'features' else 'answers'] == len(rows)*len(cases),
                    f'Stage not complete or engine-exit evidence missing: {role}')
            if role != 'features':
                final = self.read(f'{role}/complete.json')
                require(final.get('complete') is True and final.get('owned_engines_exited') is True
                        and final['plan_sha256'] == self.pins['plan.json'], f'Launcher not complete: {role}')
        if self.dataset == 'longbench-v2':
            a = {u for g in self.protocols['primary']['groups'] for u in g}
            b = {u for g in self.protocols['extra']['groups'] for u in g}
            require(not a & b, 'Primary and extra GPU groups overlap')

    def read(self, name, expected=None):
        path = safe_path(self.root, name)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as e:
            raise ExportError(f'Missing required file: {path}') from e
        sha = hashlib.sha256(raw).hexdigest()
        require(expected is None or sha == expected, f'Checksum mismatch: {path}')
        require(name not in self.pins or self.pins[name] == sha, f'Source changed during export: {path}')
        self.pins[name] = sha
        return json.loads(raw)

    def record(self, role, case, row):
        base = f'{role}/records/{case}/{row["id"]}'
        receipt = self.read(f'{base}/validated.json')
        proto = self.protocols[role]
        require(receipt.get('complete') is True and receipt['protocol_sha256'] == identity(proto),
                f'Uncommitted or mismatched record: {base}')
        files = receipt['files']
        require({'result.json', 'diagnostics.json'} <= set(files), f'Incomplete receipt: {base}')
        # Keep the existing committed-results semantics for large diagnostic files.
        # Attention files used below are additionally hashed and structurally replayed.
        for name, sha in files.items():
            path = safe_path(self.root, f'{base}/{name}')
            require(path.is_file() and re.fullmatch('[0-9a-f]{64}', sha), f'Missing/invalid committed artifact: {path}')
        result = self.read(f'{base}/result.json', files['result.json'])
        group = row['ordinal'] % len(proto['groups'])
        require(result['prompt_id'] == row['id'] and result['method'] == case
                and result['input_sha256'] == row['sha256'] and result['group'] == group
                and result['gpu_uuids'] == proto['groups'][group] and result.get('cache_immutable') is True
                and result.get('answer_validation', PROBE) == proto['answer_validation']
                and result.get('prompt_protocol') == PROMPT_PROTOCOL
                and result.get('evaluation_protocol') == row.get('evaluation_protocol'),
                f'Prompt/action/protocol identity mismatch: {base}')
        retired = result['retirement']
        require(sorted(r['rank'] for r in retired) == list(range(self.tp))
                and all(r.get('quiescent') and not r['transfers']['pending'] and not r['request_bookkeeping'] for r in retired),
                f'Missing rank retirement evidence: {base}')
        init_path = safe_path(self.root/role, result['initialization']).relative_to(self.root).as_posix()
        initial = self.read(init_path, result['initialization_sha256'])
        require(initial.get('validated') is True, f'Unvalidated initialization: {base}')
        require(all(number(t) for k, t in result['timings'].items() if not (
            k == 'first_answer_content_seconds' and t is None and result.get('generated_answer_tokens') == 0)),
            f'Invalid timings: {base}')
        if case == 'probe':
            artifacts = result['artifacts']
            require(result.get('internal_tokens') == 1
                    and sorted(a['rank'] for a in artifacts) == list(range(self.tp))
                    and len({a['path'] for a in artifacts}) == self.tp
                    and all(files.get(a['path']) == a['sha256'] for a in artifacts),
                    f'Missing/duplicate/unpinned attention rank: {base}')
        else:
            require(result['executed_action'] == case and number(result['accuracy'], upper=1)
                    and number(result['timings']['ttft_seconds'], positive=True), f'Invalid outcome: {base}')
            require(result.get('evaluation_protocol') == row.get('evaluation_protocol'), f'Evaluation protocol mismatch: {base}')
            if case == 'nocache':
                require(result['num_cached_tokens'] == 0 and not initial['engine_config'].get('kv_transfer_config'),
                        f'Baseline is not connector-free: {base}')
        self.record_digest.update(canonical([base, self.pins[f'{base}/validated.json']]))
        return result, self.root/base

    def unchanged(self):
        for name, sha in self.pins.items():
            require(file_hash(safe_path(self.root, name)) == sha, f'Source changed during export: {name}')


def reduction_matches(means, scores, prefix):
    def trees(values):
        if len(values) == 1:
            yield values[0]
        else:
            for split in range(1, len(values)):
                for left in trees(values[:split]):
                    for right in trees(values[split:]):
                        yield np.add(left, right, dtype=np.float32)
    matched = np.zeros(scores.shape, dtype=bool)
    for order in itertools.permutations(means):
        for total in trees(order):
            matched |= (total/np.float32(len(means)))[prefix:] == scores
            if matched.all():
                return True
    return False


def load_attention(folder, record, row, tp):
    """Verify every used rank archive; no external prepared/model paths needed."""
    require(sorted(a['rank'] for a in record['artifacts']) == list(range(tp))
            and len({a['path'] for a in record['artifacts']}) == tp,
            'Missing/duplicate attention rank')
    layers = heads = scores = layout = None
    means = []
    for artifact in sorted(record['artifacts'], key=lambda a: a['rank']):
        path = safe_path(folder, artifact['path'])
        require(file_hash(path) == artifact['sha256'], f'Attention checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            b = data['boundaries']
            require(b.dtype == np.int64 and b.ndim == 1 and len(b) >= 4 and b[0] == 0
                    and (np.diff(b) > 0).all() and b[-1] == row['input_tokens']
                    and (b[:-1] % 64 == 0).all() and (np.diff(b[:-1]) <= 4096).all(),
                    f'Invalid original-token boundaries: {path}')
            prefix, end = int(b[1]), int(b[-2])
            current = {k: data[k] for k in ('boundaries', 'context_positions', 'question_positions', 'original_to_formatted')}
            question = current['question_positions']
            mapping = current['original_to_formatted']
            require(np.array_equal(current['context_positions'], np.arange(end))
                    and question.dtype == np.int64 and question.ndim == 1 and len(question) > 0
                    and (np.diff(question) > 0).all() and question[0] >= end and question[-1] < b[-1]
                    and end == max(0, min(int(question[0]), int(b[-1])-256)//64*64)
                    and mapping.dtype == np.int64 and mapping.ndim == 1,
                    f'Invalid context/question position metadata: {path}')
            if layout is None:
                layout = current
                layers = np.zeros((64, end), dtype=np.float64)
                heads = np.empty((8, 64, end), dtype=np.float32)
            else:
                require(all(np.array_equal(v, layout[k]) for k, v in current.items()), f'Rank layout mismatch: {path}')
            local, mean, native = data['layers'], data['local_mean'], data['scores']
            for a, shape in ((local, (64, end)), (mean, (end,)), (native, (end-prefix,))):
                require(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all() and (a >= 0).all(),
                        f'Invalid FP32 attention arrays: {path}')
            replay = np.zeros(end, dtype=np.float32)
            for v in local:
                np.add(replay, v, out=replay)
            replay /= np.float32(64)
            require(np.array_equal(replay, mean), f'Native layer score replay mismatch: {path}')
            if scores is not None:
                require(np.array_equal(native, scores), f'TP score disagreement: {path}')
            else:
                scores = native
            order = ranked(native)
            for bpercent in RATIOS:
                require(np.array_equal(data[f'prophetkv-{bpercent}'], mask(order, bpercent)+prefix),
                        f'Native selection floor/tie mismatch: {path}, budget {bpercent}')
            head = data['heads']
            require(np.array_equal(data['head_layers'], HEAD_LAYERS) and head.dtype == np.float32
                    and head.shape == (8, 64//tp, end) and np.isfinite(head).all() and (head >= 0).all(),
                    f'Missing/invalid all-64-Q-head capture: {path}')
            rank = artifact['rank']
            heads[:, rank*(64//tp):(rank+1)*(64//tp)] = head
            layers += local
            means.append(mean)
    require(reduction_matches(means, scores, prefix), f'Native TP reduction mismatch: {folder}')
    layers /= tp
    return layers, heads, scores, prefix


@contextmanager
def source_read_lock(root):
    """Read-only shared locks where POSIX flock exists; receipts are mandatory everywhere."""
    try:
        import fcntl
    except ImportError:
        fcntl = None
    with ExitStack() as stack:
        if fcntl is not None:
            for role in ('primary', 'extra', 'ruler', 'features'):
                path = root/role/'run.lock'
                if path.exists():
                    stream = stack.enter_context(path.open('rb'))
                    try:
                        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError as e:
                        raise ExportError(f'Launcher is still running: {role}; export after completion') from e
        yield


CONVENTIONS = {
    'eligible': 'E=[boundaries[1],boundaries[-2]); P=[0,boundaries[1]); N=len(E).',
    'p': 'Native FP32 scores restricted to E, then normalize in float64.',
    'q_l': 'Equal-head mean of rank layer arrays, restricted to E, normalized per layer.',
    'h_j': 'Each saved Q head restricted to E and normalized independently; 8 layers x 64 heads.',
    'S_b': 'Common native-score top floor(N*(b/100)) mask; descending score, ascending position ties.',
    'T_b': 'Same top selection applied to the named individual distribution.',
    'coverage': 'Selected eligible mass / total eligible mass, using the common S_b.',
    'groups': 'g_early/g_late normalize raw eligible layer means for layers 0..31 / 32..63.',
    'm_l': 'Normalized mean of the 64 independently normalized saved head distributions in layer l.',
    'H_norm': 'Shannon entropy / ln(vector length), with 0*ln(0)=0.',
    'JS': 'Jensen-Shannon divergence, base-2 logs, equal mixture; range [0,1].',
    'K_t': 'Minimum descending-score token count reaching mass t; divide by N.',
    'Q_t': 'First normalized context position reaching cumulative mass t in position order.',
    'x_depth': 'x=i/(N-1); d_l=l/63; slopes use OLS with intercept on all 64 layers.',
    'quantile_std': 'NumPy linear quantile; population std ddof=0; never drop missing channels.',
    'bins': 'bin(i)=floor(B*i/N); b_B holds mass per bin; require N>=64 for bin features.',
    'prefix': 'u is normalized mean of full layer vectors on P union E, before fresh suffix.',
    'missing': 'NaN plus X_missing boolean; undefined denominators/empty masks are not imputed.',
    'precision': 'Native selection uses archived FP32 scores; reductions float64, export float32.',
}


def array_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        require(not a.dtype.hasobject, f'Object array forbidden: {name}')
        h.update(canonical([name, a.dtype.str, list(a.shape)]))
        h.update(a.tobytes())
    return h.hexdigest()


def verify_npz(path):
    """Verify an exported file without pickle or repository dependencies."""
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(set(saved.files)), 'Duplicate NPZ member')
        metadata = json.loads(str(saved['metadata_json'].item()))
        arrays = {name: saved[name] for name in saved.files if name != 'metadata_json'}
    require(metadata.get('schema') == 'attention-router-compact-v1', 'Unknown compact schema')
    expected = metadata.pop('payload_sha256')
    require(metadata['arrays_sha256'] == array_digest(arrays)
            and digest(metadata) == expected, 'Compact payload checksum mismatch')
    return arrays, dict(metadata, payload_sha256=expected)


def publish_npz(output, arrays, metadata):
    """Validate a temporary NPZ, then atomically publish without replacing anything."""
    require(output.suffix == '.npz', 'Output must end in .npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists() and not output.is_symlink(), f'Output exists; choose another --output: {output}')
    metadata = dict(metadata, arrays_sha256=array_digest(arrays))
    metadata['payload_sha256'] = digest(metadata)
    payload = dict(arrays, metadata_json=np.array(canonical(metadata).decode()))
    fd, name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **payload)
            stream.flush(); os.fsync(stream.fileno())
        verify_npz(temporary)
        # Same-filesystem hard link gives atomic no-clobber publication, even on races.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return file_hash(output)


def export(root, output=None, quiet=False):
    root = Path(root).resolve()
    with source_read_lock(root):
        source = Collection(root)
        output = Path(output).resolve() if output else root/f'{source.dataset}-features.npz'
        require(output.suffix == '.npz', 'Output must end in .npz')
        require(not output.exists(), f'Output exists; choose another --output: {output}')
        n, tp = len(source.rows), source.tp
        X = np.empty((n, 100), dtype=np.float32)
        accuracy = np.empty((n, 12), dtype=np.float32)
        ttft = np.empty((n, 12), dtype=np.float32)
        overhead = np.empty(n, dtype=np.float32)
        offline_seconds = np.empty(n, dtype=np.float64)
        control_gpus = np.empty((n, 12, tp), dtype='<U40')
        probe_gpus = np.empty((n, tp), dtype='<U40')
        token_hashes = []
        if not quiet:
            print(f'{source.dataset}: {n} prompts, 100 features, 12 actions; CPU only.', flush=True)
        last_print = time.monotonic()
        for i, row in enumerate(source.rows):
            try:
                probe, folder = source.record('features', 'probe', row)
                overhead[i] = probe['timings']['routing_overhead_seconds']
                probe_gpus[i] = probe['gpu_uuids']
                token_hash = probe['token_sha256']
                require(isinstance(token_hash, str) and re.fullmatch('[0-9a-f]{64}', token_hash), 'Invalid probe token hash')
                token_hashes.append(token_hash)
                for j, case in enumerate(ACTION_NAMES):
                    role = ('ruler' if source.dataset == 'ruler' else 'primary' if case in PRIMARY else 'extra')
                    result, _ = source.record(role, case, row)
                    require(result['token_sha256'] == token_hash, 'Answer and probe token hashes differ')
                    if source.settings.get('execution_profile') == 'ruler-thinking':
                        require(result.get('ruler_thinking_budget') == row.get('ruler_thinking_budget')
                                and result.get('ruler_thinking_budget') is not None, 'Thinking budget mismatch')
                    accuracy[i, j] = result['accuracy']
                    ttft[i, j] = result['timings']['ttft_seconds']
                    control_gpus[i, j] = result['gpu_uuids']
                started = time.perf_counter()
                layers, heads, scores, prefix = load_attention(folder, probe, row, tp)
                values = compute_features(layers, heads, scores, prefix)
                del layers, heads, scores
                offline_seconds[i] = time.perf_counter()-started
                for name in OLD_FEATURES:
                    saved = probe['features'][name]
                    value = values[FEATURE_NAMES.index(name)]
                    require((saved is None and np.isnan(value)) or (number(saved) and np.isclose(value, saved, rtol=1e-10, atol=1e-12)),
                            f'Legacy feature replay mismatch: {name}')
                X[i] = values
            except (ValueError, KeyError, TypeError, OSError, IndexError) as e:
                raise ExportError(f'Prompt {row["id"]}: {e}') from e
            if not quiet and (i == 0 or (i+1) % 10 == 0 or i+1 == n or time.monotonic()-last_print >= 30):
                print(f'Exported scalars {i+1}/{n}: {row["id"]}', flush=True)
                last_print = time.monotonic()
        require(not np.isinf(X).any() and np.isfinite(accuracy).all()
                and np.isfinite(ttft).all() and (ttft > 0).all() and np.isfinite(overhead).all(),
                'Float32 conversion produced invalid/underflowed values')
        source.unchanged()
        arrays = dict(
            X=X, X_missing=np.isnan(X), y_accuracy=accuracy, y_ttft=ttft,
            feature_names=np.array(FEATURE_NAMES), feature_groups=np.array([s[1] for s in FEATURE_SPEC]),
            action_names=np.array(ACTION_NAMES), prompt_ids=np.array([r['id'] for r in source.rows]),
            input_hashes=np.array([r['sha256'] for r in source.rows]), token_hashes=np.array(token_hashes),
            dataset=np.array(source.dataset), task=np.array([r['subtask'] for r in source.rows]),
            length_group=np.array([r.get('length', '') for r in source.rows]),
            source_ids=np.array([str(r.get('source_id', r['id'])) for r in source.rows]),
            ordinals=np.array([r['ordinal'] for r in source.rows], dtype=np.int64),
            evaluation_protocol=np.array([r.get('evaluation_protocol') or '' for r in source.rows]),
            probe_overhead_seconds=overhead, control_gpu_uuids=control_gpus, probe_gpu_uuids=probe_gpus,
            offline_extraction_seconds=offline_seconds)
        metadata = dict(schema='attention-router-compact-v1', feature_version=VERSION,
            feature_definitions=[dict(name=name, group=group, definition=definition) for name, group, definition in FEATURE_SPEC],
            conventions=CONVENTIONS, prompt_protocol=PROMPT_PROTOCOL, dataset=source.dataset,
            samples=n, actions=ACTIONS, feature_count=100, feature_layers=list(HEAD_LAYERS), total_q_heads=64,
            accuracy_units='0..1; native dataset score per prompt', ttft_units='seconds',
            timing='Measured fixed-control TTFT; independent stored five-feature probe overhead is separate. '
                   'offline_extraction_seconds includes archive reads/checks and CPU feature extraction; '
                   'it is not online routing overhead or end-to-end router latency.',
            validation='Frozen metadata and committed result hashes/identities, completion/retirement/initialization receipts; '
                       'attention SHA256, layout/rank/head shapes, native FP32 score/mask replay, and five legacy feature replay. '
                       'No model/tokenizer, answer re-scoring, diagnostic JSON replay, or new GPU verification.',
            missing='NaN in X with X_missing mask; corrupt evidence halts export; no implicit imputation.',
            split='Complete cohort, no split/refit performed. Split and select features within training folds.',
            source_settings=source.settings, plan_sha256=source.pins['plan.json'],
            source_protocols=source.protocols, source_record_receipts_sha256=source.record_digest.hexdigest(),
            source_control_files={k: v for k, v in source.pins.items() if '/records/' not in k},
            exporter_sha256=file_hash(Path(__file__)), numpy_version=np.__version__)
        checksum = publish_npz(output, arrays, metadata)
    if not quiet:
        print(f'Saved {output}\nSHA256 {checksum}\nShapes: X={X.shape}, y_accuracy={accuracy.shape}, y_ttft={ttft.shape}', flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment_dir', type=Path, help='Completed collection root containing settings.json, plan.json and rows.json')
    parser.add_argument('--output', type=Path, help='New output .npz path (default: EXPERIMENT_DIR/DATASET-features.npz)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')
    args = parser.parse_args(argv)
    try:
        export(args.experiment_dir, args.output, args.quiet)
    except (ValueError, OSError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile) as e:
        parser.exit(1, f'Export failed: {e}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
