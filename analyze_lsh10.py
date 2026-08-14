#!/usr/bin/env python3
"""
Phân tích Bucket-LSH mười seed — TODO 3 điểm 2-3.

    python3 analyze_lsh10.py

Với mỗi (corpus, T, seed), chọn bề rộng bucket b có pool ứng viên GẦN NHẤT với
V-Engram, rồi tổng hợp. Ghi lại b và pool để bảng trong bài lặp lại được.
"""
import glob
import re
import statistics as st
from collections import defaultdict

# ngân sách ứng viên của V-Engram, dùng để chọn bề rộng bucket
VENGRAM_CAND = {'code': 4510, 'scifact': 1845}
# recall V-Engram cùng cấu hình, mười seed, mô hình closest-peer lý tưởng hoá
VENGRAM_R5 = {'code': 80.0, 'scifact': 80.0}


def parse(f):
    """Trả list (b, recall, pool) từ các dòng 'b=.. T=..: Recall@5=..'."""
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return []
    rows = []
    for m in re.finditer(
            r'b=\s*(\d+)\s+T=\d+:\s*Recall@5=\s*([\d.]+)%.*?pool TB=\s*([\d,]+)', t):
        rows.append((int(m.group(1)), float(m.group(2)),
                     float(m.group(3).replace(',', ''))))
    return rows


g = defaultdict(list)
for f in glob.glob('lsh10_*_T*_s*.txt'):
    m = re.match(r'lsh10_(\w+)_T(\d+)_s(\d+)\.txt', f)
    if not m:
        continue
    ds, T = m.group(1), int(m.group(2))
    rows = parse(f)
    if not rows:
        continue
    target = VENGRAM_CAND.get(ds, 4510)
    # chọn b có pool gần ngân sách V-Engram nhất
    b, r, pool = min(rows, key=lambda x: abs(x[2] - target))
    g[(ds, T)].append({'b': b, 'r': r, 'pool': pool})

if not g:
    raise SystemExit('Chưa có file lsh10_*.txt. Chạy run_lsh_10seeds.sh trước.')

for ds in sorted({a for a, _ in g}):
    ve = VENGRAM_R5.get(ds, 80.0)
    print()
    print('=' * 74)
    print(f'{ds}  —  V-Engram cùng cấu hình, mười seed: {ve:.1f}%')
    print(f'         ngân sách ứng viên để khớp: {VENGRAM_CAND.get(ds, 0):,}')
    print('=' * 74)
    print(f"{'T':>3s} {'n':>3s} {'Recall@5':>13s} {'b đã chọn':>11s} "
          f"{'pool ứng viên':>15s} {'tỉ lệ V-E':>10s}")
    print('-' * 62)
    base = None
    for T in sorted(b for a, b in g if a == ds):
        v = g[(ds, T)]
        r = st.mean(x['r'] for x in v)
        sd = st.stdev([x['r'] for x in v]) if len(v) > 1 else 0.0
        pool = st.mean(x['pool'] for x in v)
        bs = [x['b'] for x in v]
        # bề rộng chọn có ổn định giữa các seed không
        bmode = st.mode(bs)
        bstr = f'{bmode}' if len(set(bs)) == 1 else f'{bmode} ({min(bs)}-{max(bs)})'
        if base is None:
            base = r
        print(f'{T:>3} {len(v):>3} {r:>8.1f}±{sd:<4.1f} {bstr:>11s} '
              f'{pool:>14,.0f} {ve/r:>9.2f}x')
    Ts = sorted(b for a, b in g if a == ds)
    if len(Ts) >= 2:
        r1 = st.mean(x['r'] for x in g[(ds, Ts[0])])
        r8 = st.mean(x['r'] for x in g[(ds, Ts[-1])])
        closed = 100 * (r8 - r1) / (ve - r1) if ve > r1 else 0
        print()
        print(f'  Multi-probe đóng {closed:.0f}% khoảng cách '
              f'({r1:.1f} -> {r8:.1f}, đích {ve:.1f})')
        pool8 = st.mean(x['pool'] for x in g[(ds, Ts[-1])])
        tgt = VENGRAM_CAND.get(ds, 0)
        print(f'  Pool ở T={Ts[-1]}: {pool8:,.0f} so với V-Engram {tgt:,} '
              f'(lệch {100*abs(pool8-tgt)/tgt:.0f}%)')

print()
print('=' * 74)
print('DÒNG CHO BẢNG TRONG BÀI')
print('=' * 74)
for ds in sorted({a for a, _ in g}):
    cells = []
    for T in sorted(b for a, b in g if a == ds):
        v = g[(ds, T)]
        r = st.mean(x['r'] for x in v)
        sd = st.stdev([x['r'] for x in v]) if len(v) > 1 else 0.0
        cells.append(f'${r:.1f}\\pm{sd:.1f}$')
    name = 'CodeSearchNet' if ds == 'code' else 'SciFact'
    ve = VENGRAM_R5.get(ds, 80.0)
    print(f'{name:14s} & ' + ' & '.join(cells) +
          f' & $\\mathbf{{{ve:.1f}}}$ \\\\')

print()
print('CÂU CHO CAPTION (TODO 3 điểm 3):')
for ds in sorted({a for a, _ in g}):
    parts = []
    for T in sorted(b for a, b in g if a == ds):
        v = g[(ds, T)]
        bs = [x['b'] for x in v]
        pool = st.mean(x['pool'] for x in v)
        bmode = st.mode(bs)
        parts.append(f'$T{{=}}{T}$: $b{{=}}{bmode}$, {pool:,.0f} candidates')
    name = 'CodeSearchNet' if ds == 'code' else 'SciFact'
    print(f'  {name}: ' + '; '.join(parts) + '.')

print()
print('KIỂM CHO TODO 6:')
ns = {len(v) for v in g.values()}
if ns == {10}:
    print('  ✓ Mọi cấu hình đủ mười seed. Rút được caveat về số mẫu lệch nhau.')
else:
    print(f'  ○ Số seed chưa đồng đều: {sorted(ns)}. Chưa rút caveat được.')
bs_all = {tuple(sorted({x['b'] for x in v})) for v in g.values()}
if all(len(b) == 1 for b in bs_all):
    print('  ✓ Bề rộng bucket ổn định qua các seed — ghi được một giá trị.')
else:
    print('  ○ Bề rộng bucket đổi theo seed; caption phải ghi khoảng thay vì một số.')
