#!/usr/bin/env python3
"""
Phân tích 2x2 Ripple Search: điều kiện dừng × phạm vi hỏi.

    python3 analyze_axes.py

Đọc file result_full_*.json đã sinh, không chạy lại thí nghiệm.
Chỉ dùng trường CHẮC CHẮN có trong JSON: recall_at_5, disc_rpcs, disc_rounds,
overlap_mean, stop_rule, frontier_scope. Bản trước gãy vì đoán nhầm tên
'nodes_touched' — trường đó không tồn tại, 'nodes' là số node của mạng.
"""
import glob
import json
import statistics as st
from collections import defaultdict

g = defaultdict(list)
for f in glob.glob('result_full_code_N10000_*_nq500.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    sr, fs = d.get('stop_rule'), d.get('frontier_scope')
    if sr and fs and d.get('recall_at_5') is not None:
        g[(sr, fs)].append(d)

if not g:
    raise SystemExit('Không thấy file nào có stop_rule/frontier_scope.')


def m(v, k):
    x = [d[k] for d in v if d.get(k) is not None]
    return st.mean(x) if x else float('nan')


def sd(v, k):
    x = [d[k] for d in v if d.get(k) is not None]
    return st.stdev(x) if len(x) > 1 else 0.0


print('=' * 68)
print('RIPPLE SEARCH — TÁCH HAI TRỤC')
print('=' * 68)
print(f"{'dừng':9s} {'phạm vi':8s} {'n':>2s} {'Recall@5':>13s} {'RPC':>7s} "
      f"{'vòng':>7s} {'overlap':>8s}")
print('-' * 60)
res, ovl = {}, {}
for sr in ('stable', 'exhaust'):
    for fs in ('all', 'topk'):
        v = g.get((sr, fs), [])
        if not v:
            print(f'{sr:9s} {fs:8s} {"(chưa chạy)":>14s}')
            continue
        r = m(v, 'recall_at_5')
        o = m(v, 'overlap_mean')
        res[(sr, fs)] = r
        ovl[(sr, fs)] = o
        print(f"{sr:9s} {fs:8s} {len(v):>2} {r:>8.1f}±{sd(v,'recall_at_5'):<4.1f} "
              f"{m(v,'disc_rpcs'):>7.0f} {m(v,'disc_rounds'):>7.0f} "
              f"{100*o:>7.1f}%")

if len(res) == 4:
    base = res[('stable', 'all')]
    d_stop = res[('exhaust', 'all')] - base
    d_scope = res[('stable', 'topk')] - base
    d_both = res[('exhaust', 'topk')] - base
    print()
    print('=' * 68)
    print('ĐÓNG GÓP CỦA TỪNG TRỤC (so với cấu hình hiện dùng)')
    print('=' * 68)
    print(f"  mốc (stable, all)         {base:>7.1f}%")
    print(f"  chỉ đổi ĐIỀU KIỆN DỪNG    {d_stop:>+7.1f} điểm")
    print(f"  chỉ đổi PHẠM VI HỎI       {d_scope:>+7.1f} điểm")
    print(f"  đổi cả hai                {d_both:>+7.1f} điểm")
    print(f"  cộng tính: {d_stop:+.1f} + {d_scope:+.1f} = {d_stop+d_scope:+.1f} "
          f"so với {d_both:+.1f}")
    print()
    if abs(d_scope) > abs(d_stop) + 1:
        print('  => PHẠM VI HỎI là trục chính.')
        print('     Mô tả trong bài: Ripple Search chủ động hỏi ra NGOÀI frontier')
        print('     hiện tại, vì mục tiêu là ĐỘ PHỦ chứ không phải hội tụ.')
    elif abs(d_stop) > abs(d_scope) + 1:
        print('  => ĐIỀU KIỆN DỪNG là trục chính.')
        print('     Mô tả trong bài: Ripple Search chấp nhận frontier xấp xỉ thay vì')
        print('     tiêu thêm RPC để hội tụ về peer XOR-gần hơn trên toàn cục.')
    else:
        print('  => Hai trục đóng góp tương đương.')
        if abs(d_stop + d_scope - d_both) > 1.5:
            print('     Và chúng TƯƠNG TÁC (cộng tính không khớp) — phải mô tả cả hai.')

print()
print('=' * 68)
print('OVERLAP VỚI TẬP XOR-GẦN NHẤT TOÀN CỤC vs RECALL')
print('=' * 68)
pts = sorted((ovl[k], res[k], f'{k[0]}/{k[1]}') for k in res)
for o, r, lbl in pts:
    print(f'  {lbl:16s} overlap {100*o:>5.1f}%   recall {r:>5.1f}%')
if len(pts) >= 2:
    lo_o, lo_r, lo_l = pts[0]
    hi_o, hi_r, hi_l = pts[-1]
    print()
    print(f'  overlap thấp nhất : {lo_l:14s} {100*lo_o:>5.1f}%  ->  recall {lo_r:.1f}%')
    print(f'  overlap cao nhất  : {hi_l:14s} {100*hi_o:>5.1f}%  ->  recall {hi_r:.1f}%')
    print()
    if lo_r > hi_r + 0.5:
        print('  => Cấu hình có overlap THẤP HƠN lại cho recall CAO HƠN.')
        print('     Bằng chứng TRỰC TIẾP rằng hội tụ về tập XOR-gần nhất KHÔNG phải')
        print('     mục tiêu đúng cho khám phá ứng viên ngữ nghĩa. Đưa số này vào bài.')
    elif hi_r > lo_r + 0.5:
        print('  => Overlap cao đi kèm recall cao. Luận điểm "hội tụ XOR không phải')
        print('     mục tiêu" KHÔNG được số liệu ủng hộ; nên mô tả thuật toán là xấp')
        print('     xỉ vì ngân sách, chứ không vì mục tiêu khác.')
    else:
        print('  => Recall gần như không phụ thuộc overlap. Nói được rằng hội tụ XOR')
        print('     là điều kiện KHÔNG CẦN THIẾT, dù chưa nói được nó có hại.')
