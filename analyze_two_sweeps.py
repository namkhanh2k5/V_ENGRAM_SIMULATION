#!/usr/bin/env python3
"""Phân tích hai sweep còn lại: projection-independence và replication/failure."""
import glob, json, math, re, statistics as st
from collections import defaultdict


def load(pat):
    out = []
    for f in glob.glob(pat):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def ms(v, k):
    xs = [x[k] for x in v if x.get(k) is not None]
    if not xs:
        return float('nan'), 0.0, 0
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs))


# ---------------------------------------------------------------- A
print('=' * 72)
print('A. PROJECTION-INDEPENDENCE — GỘP QUA MỌI SEED')
print('=' * 72)
for ds in ('code', 'scifact'):
    v = load(f'indep_{ds}_s*.json')
    v = [x for x in v if x.get('p_bar') is not None]
    if not v:
        print(f'  {ds}: chưa có dữ liệu'); continue
    pb, pbs, n = ms(v, 'p_bar')
    oo, oos, _ = ms(v, 'or_observed')
    oi, ois, _ = ms(v, 'or_if_independent')
    sh, shs, _ = ms(v, 'or_shortfall')
    print()
    print(f'  {ds}  (n={n} seed)')
    print(f'    p trung bình mỗi bảng     {100*pb:>6.1f} ± {100*pbs:.1f}')
    print(f'    OR đo được                {100*oo:>6.1f} ± {100*oos:.1f}')
    print(f'    OR nếu độc lập 1-(1-p)^L  {100*oi:>6.1f} ± {100*ois:.1f}')
    print(f'    thiếu hụt                 {100*sh:>+6.1f} ± {100*shs:.1f}')
    # kiểm định một mẫu: thiếu hụt có khác 0 không
    xs = [x['or_shortfall'] for x in v if x.get('or_shortfall') is not None]
    if len(xs) > 1:
        se = st.stdev(xs) / math.sqrt(len(xs))
        t = st.mean(xs) / se if se else float('nan')
        tc = {9: 2.26, 4: 2.78, 2: 4.30}.get(len(xs) - 1, 2.26)
        v_ = 'ĐẠT' if abs(t) > tc else 'chưa đạt'
        print(f'    t = {t:.2f} (ngưỡng {tc}), {v_}')
        if abs(t) > tc:
            print(f'    => Thiếu hụt là THẬT, không phải nhiễu của lần rút ma trận.')
        else:
            print(f'    => Chưa phân biệt được với 0. Các bảng chưa chứng minh là')
            print(f'       phụ thuộc nhau ở cỡ mẫu này.')
    # phân tán p giữa các bảng
    pe = [x['p_each'] for x in v if x.get('p_each')]
    if pe:
        allp = [p for row in pe for p in row]
        print(f'    p từng bảng: min {100*min(allp):.1f}, max {100*max(allp):.1f}, '
              f'chênh {max(allp)/max(min(allp),1e-9):.1f} lần')

# ---------------------------------------------------------------- B
print()
print('=' * 72)
print('B. HEALTHY REPLICATION — 10 SEED')
print('=' * 72)
print(f"{'r':>3s} {'n':>3s} {'Recall@5':>13s} {'Reach':>13s} {'peer%':>7s} {'cand%':>7s}")
print('-' * 52)
for r in (1, 2, 3, 4):
    v = load(f'repl_r{r}_s*.json')
    v = [x for x in v if x.get('recall5') is not None]
    if not v:
        print(f'{r:>3} {"(chưa chạy)":>16s}'); continue
    m1, s1, n = ms(v, 'recall5')
    m2, s2, _ = ms(v, 'reachable_recall5')
    pc, _, _ = ms(v, 'pct_network_touched')
    cd = st.mean(100 * x['mean_unique_candidates'] / 20000 for x in v
                 if x.get('mean_unique_candidates'))
    print(f'{r:>3} {n:>3} {m1:>8.1f}±{s1:<4.1f} {m2:>8.1f}±{s2:<4.1f} '
          f'{pc:>6.1f}% {cd:>6.1f}%')

# ---------------------------------------------------------------- C
print()
print('=' * 72)
print('C. STATIC FAILURE — 10 SEED, r=1')
print('=' * 72)
print(f"{'loss':>5s} {'n':>3s} {'Recall@5':>13s} {'Reach':>13s} {'Δ so 0%':>8s}")
print('-' * 48)
base = None
for L in (0, 10, 20, 30):
    v = load(f'fail_L{L}_s*.json')
    v = [x for x in v if x.get('recall5') is not None]
    if not v:
        print(f'{L:>4}% {"(chưa chạy)":>16s}'); continue
    m1, s1, n = ms(v, 'recall5')
    m2, s2, _ = ms(v, 'reachable_recall5')
    if base is None:
        base = m1
    print(f'{L:>4}% {n:>3} {m1:>8.1f}±{s1:<4.1f} {m2:>8.1f}±{s2:<4.1f} {m1-base:>+7.1f}')

print()
print('=' * 72)
print('  Ba sweep này dùng mô hình closest-peer lý tưởng hoá, không đi walk,')
print('  nên không bị ảnh hưởng bởi bảng định tuyến hay cách bootstrap.')
