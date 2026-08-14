#!/usr/bin/env python3
"""
Phân tích rerun k-bucket, kèm so với số ring cũ để biết chỗ nào trong bài đổi.

    python3 analyze_kbucket_full.py
"""
import glob
import json
import math
import re
import statistics as st
from collections import defaultdict

# Số cũ đo bằng small-world ring, để đối chiếu
RING = {
    'A': {'semantic': 73.9, 'keyed_lookup': 33.5,
          'random_slots': 33.8, 'random_unique': 22.7},
    'B': {('stable', 'all'): 74.0, ('stable', 'topk'): 69.4,
          ('exhaust', 'all'): 76.3, ('exhaust', 'topk'): 69.2},
    'C': {'margin': 74.0, 'random': 66.2},
}
IDEAL = {'semantic': 80.0, 'keyed_lookup': 33.4,
         'random_slots': 34.5, 'random_unique': 22.1}


def read(f):
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    g = lambda p: (float(m.group(1).replace(',', ''))
                   if (m := re.search(p, t)) else None)
    return {'recall': g(r'Recall@5\s*:\s*([\d.]+)%'),
            'rpc': g(r'RPC/query\s*:\s*([\d,.]+)'),
            'rounds': g(r'Rounds/query\s*:\s*([\d,.]+)'),
            'nodes': g(r'Unique nodes contacted\s+([\d,]+)'),
            'bytes': g(r'Bytes/query\s+([\d,]+)'),
            'p50': g(r'Latency p50 \(ms\)\s+([\d,.]+)')}


def mean(v, k):
    x = [d[k] for d in v if d and d.get(k) is not None]
    return st.mean(x) if x else float('nan')


def sd(v, k):
    x = [d[k] for d in v if d and d.get(k) is not None]
    return st.stdev(x) if len(x) > 1 else 0.0


def paired(fa, fb, key='recall'):
    """So ghép cặp theo seed giữa hai mẫu file."""
    A, B = {}, {}
    for f in glob.glob(fa):
        m = re.search(r'_s(\d+)\.txt$', f)
        if m and (r := read(f)) and r[key] is not None:
            A[m.group(1)] = r[key]
    for f in glob.glob(fb):
        m = re.search(r'_s(\d+)\.txt$', f)
        if m and (r := read(f)) and r[key] is not None:
            B[m.group(1)] = r[key]
    d = [A[s] - B[s] for s in sorted(set(A) & set(B))]
    if len(d) < 2:
        return None
    m_, s_ = st.mean(d), st.stdev(d)
    se = s_ / math.sqrt(len(d))
    tc = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
          6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(len(d) - 1, 2.0)
    return {'n': len(d), 'mean': m_, 'se': se,
            't': m_ / se if se else float('nan'), 'tc': tc,
            'wins': sum(1 for x in d if x > 0)}


# ------------------------------------------------------------------ A
print('=' * 74)
print('A. HEADLINE VỚI K-BUCKET')
print('=' * 74)
print(f"{'chế độ':16s} {'n':>2s} {'Recall@5':>13s} {'ring cũ':>8s} {'Δ':>6s} "
      f"{'RPC':>7s} {'peer':>6s}")
print('-' * 64)
new = {}
for m_ in ('semantic', 'keyed_lookup', 'random_slots', 'random_unique'):
    v = [read(f) for f in glob.glob(f'kbf_A_{m_}_s*.txt')]
    v = [x for x in v if x and x['recall'] is not None]
    if not v:
        print(f'{m_:16s} {"(chưa chạy)":>16s}'); continue
    r = mean(v, 'recall')
    new[m_] = r
    old = RING['A'][m_]
    print(f"{m_:16s} {len(v):>2} {r:>8.1f}±{sd(v,'recall'):<4.1f} {old:>8.1f} "
          f"{r-old:>+5.1f} {mean(v,'rpc'):>7.0f} {mean(v,'nodes'):>6.0f}")

if 'semantic' in new:
    print()
    print(f"{'tỉ lệ so':18s} {'k-bucket':>9s} {'ring cũ':>9s} {'lý tưởng':>9s}")
    print('-' * 50)
    for m_ in ('keyed_lookup', 'random_slots', 'random_unique'):
        if new.get(m_):
            print(f'{m_:18s} {new["semantic"]/new[m_]:>8.2f}x '
                  f'{RING["A"]["semantic"]/RING["A"][m_]:>8.2f}x '
                  f'{IDEAL["semantic"]/IDEAL[m_]:>8.2f}x')

# ------------------------------------------------------------------ B
print()
print('=' * 74)
print('B. TERMINATION ABLATION VỚI K-BUCKET')
print('=' * 74)
diag = defaultdict(list)
for f in glob.glob('kbf_termabl_*.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    sr, fs = d.get('stop_rule'), d.get('frontier_scope')
    if sr and fs:
        diag[(sr, fs)].append(d)

resB = {}
if diag:
    print(f"{'dừng':9s} {'phạm vi':8s} {'n':>2s} {'Recall@5':>9s} {'ring cũ':>8s} "
          f"{'RPC':>7s} {'Jaccard':>8s} {'XOR rank':>9s}")
    print('-' * 68)
    for sr in ('stable', 'exhaust'):
        for fs in ('all', 'topk'):
            v = diag.get((sr, fs), [])
            if not v:
                print(f'{sr:9s} {fs:8s} {"(chưa chạy)":>12s}'); continue

            def a(k, _v=v):
                xs = [x[k] for x in _v if x.get(k) is not None]
                return st.mean(xs) if xs else float('nan')

            r = a('recall_at_5')
            resB[(sr, fs)] = r
            old = RING['B'].get((sr, fs), float('nan'))
            print(f"{sr:9s} {fs:8s} {len(v):>2} {r:>8.1f}% {old:>7.1f}% "
                  f"{a('disc_rpcs'):>7.0f} {a('jaccard_mean'):>8.3f} "
                  f"{a('xor_rank_mean'):>9.1f}")
    if len(resB) == 4:
        b = resB[('stable', 'all')]
        print()
        print('  TÁCH HAI TRỤC (so với stable/all):')
        for k, lbl in [(('exhaust', 'all'), 'chỉ đổi ĐIỀU KIỆN DỪNG'),
                       (('stable', 'topk'), 'chỉ đổi PHẠM VI HỎI'),
                       (('exhaust', 'topk'), 'đổi cả hai')]:
            print(f'    {lbl:24s} {resB[k]-b:>+6.1f} điểm')

# ------------------------------------------------------------------ C
print()
print('=' * 74)
print('C. MARGIN ABLATION VỚI K-BUCKET')
print('=' * 74)
for po in ('margin', 'random'):
    v = [read(f) for f in glob.glob(f'kbf_C_{po}_s*.txt')]
    v = [x for x in v if x and x['recall'] is not None]
    if v:
        print(f"  {po:8s} n={len(v):<2} Recall@5 {mean(v,'recall'):>5.1f}±"
              f"{sd(v,'recall'):<4.1f}  (ring cũ {RING['C'][po]:.1f})")
pc = paired('kbf_C_margin_s*.txt', 'kbf_C_random_s*.txt')
if pc:
    print()
    print(f"  ghép cặp: chênh {pc['mean']:+.1f} điểm, SE {pc['se']:.2f}, "
          f"t {pc['t']:.2f} (ngưỡng {pc['tc']}), thắng {pc['wins']}/{pc['n']}")
    print(f"  => {'ĐẠT ý nghĩa' if abs(pc['t']) > pc['tc'] else 'CHƯA đạt'}")

# ------------------------------------------------------------------ D
print()
print('=' * 74)
print('D. BẢNG CHI PHÍ VỚI K-BUCKET')
print('=' * 74)
v = [read(f) for f in glob.glob('kbf_D_cost_s*.txt')]
v = [x for x in v if x and x['rpc'] is not None]
if v:
    print(f'  n={len(v)} seed')
    for k, lbl, u in [('rounds', 'vòng/query', ''), ('rpc', 'RPC/query', ''),
                      ('bytes', 'bytes/query', ''), ('nodes', 'peer chạm', ''),
                      ('p50', 'latency p50', 'ms')]:
        print(f'  {lbl:14s} {mean(v,k):>10,.0f} ± {sd(v,k):>6,.0f} {u}')
else:
    print('  chưa có dữ liệu')

# ------------------------------------------------------------------ tổng kết
print()
print('=' * 74)
print('CHỖ NÀO TRONG BÀI PHẢI ĐỔI')
print('=' * 74)
if 'semantic' in new:
    print(f"  headline semantic : {RING['A']['semantic']:.1f} -> {new['semantic']:.1f}")
for k in sorted(resB):
    print(f"  {k[0]}/{k[1]:5s}       : {RING['B'][k]:.1f} -> {resB[k]:.1f}")
if v:
    print(f"  bảng chi phí RPC  : 1,148 -> {mean(v,'rpc'):,.0f}")
