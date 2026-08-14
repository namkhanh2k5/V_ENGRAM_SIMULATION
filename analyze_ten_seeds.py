#!/usr/bin/env python3
"""
Phân tích termination và margin ablation ở mười seed, kèm phép kiểm chéo.

    python3 analyze_ten_seeds.py
"""
import glob
import json
import math
import re
import statistics as st
from collections import defaultdict


def read(f):
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    m = re.search(r'Recall@5\s*:\s*([\d.]+)%', t)
    r = re.search(r'RPC/query\s*:\s*([\d,.]+)', t)
    return {'recall': float(m.group(1)) if m else None,
            'rpc': float(r.group(1).replace(',', '')) if r else None}


def by_seed(pattern, key='recall'):
    d = {}
    for f in glob.glob(pattern):
        m = re.search(r'_s(\d+)\.txt$', f)
        if m and (x := read(f)) and x[key] is not None:
            d[m.group(1)] = x[key]
    return d


def json_by_seed(pattern, key='recall_at_5'):
    """Đọc {seed: giá trị} từ các file JSON. Bỏ qua file hỏng hoặc thiếu khoá.

    Bản trước gọi re.search(...).group(1) thẳng trong dict comprehension: nếu
    JSON hỏng hoặc thiếu khoá thì gãy, và lỗi khó truy vì nằm trong comprehension.
    """
    d = {}
    for f in glob.glob(pattern):
        m = re.search(r'_s(\d+)\.json$', f)
        if not m:
            continue
        try:
            v = json.load(open(f)).get(key)
        except Exception:
            continue
        if v is not None:
            d[m.group(1)] = v
    return d


def paired(A, B):
    common = sorted(set(A) & set(B))
    d = [A[s] - B[s] for s in common]
    if len(d) < 2:
        return None
    m_, s_ = st.mean(d), st.stdev(d)
    se = s_ / math.sqrt(len(d))
    tc = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
          7: 2.36, 8: 2.31, 9: 2.26}.get(len(d) - 1, 2.0)
    return {'n': len(d), 'mean': m_, 'sd': s_, 'se': se,
            't': m_ / se if se else float('nan'), 'tc': tc,
            'wins': sum(1 for x in d if x > 0)}


# ---------------------------------------------------------------- B
print('=' * 72)
print('B. TERMINATION ABLATION — MƯỜI SEED')
print('=' * 72)
diag = defaultdict(list)
for f in glob.glob('kbf_termabl_*.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if (sr := d.get('stop_rule')) and (fs := d.get('frontier_scope')):
        diag[(sr, fs)].append(d)

resB = {}
if diag:
    print(f"{'dừng':9s} {'phạm vi':8s} {'n':>3s} {'Recall@5':>13s} {'Reach':>7s} "
          f"{'RPC':>7s} {'Jaccard':>8s} {'XOR rank':>9s}")
    print('-' * 70)
    for sr in ('stable', 'exhaust'):
        for fs in ('all', 'topk'):
            v = diag.get((sr, fs), [])
            if not v:
                print(f'{sr:9s} {fs:8s} {"(chưa chạy)":>14s}'); continue

            def a(k, _v=v):
                xs = [x[k] for x in _v if x.get(k) is not None]
                return st.mean(xs) if xs else float('nan')

            rs = [x['recall_at_5'] for x in v if x.get('recall_at_5') is not None]
            sd = st.stdev(rs) if len(rs) > 1 else 0.0
            resB[(sr, fs)] = a('recall_at_5')
            print(f"{sr:9s} {fs:8s} {len(v):>3} {a('recall_at_5'):>8.1f}±{sd:<4.1f} "
                  f"{a('reachable_recall5'):>6.1f}% {a('disc_rpcs'):>7.0f} "
                  f"{a('jaccard_mean'):>8.3f} {a('xor_rank_mean'):>9.1f}")

    if len(resB) == 4:
        base = resB[('stable', 'all')]
        print()
        print('  TÁCH HAI TRỤC (ghép cặp theo seed):')
        for ka, kb, lbl in [
            (('exhaust', 'all'), ('stable', 'all'), 'ĐIỀU KIỆN DỪNG'),
            (('stable', 'topk'), ('stable', 'all'), 'PHẠM VI HỎI'),
        ]:
            A = json_by_seed(f'kbf_termabl_{ka[0]}-{ka[1]}_s*.json')
            B = json_by_seed(f'kbf_termabl_{kb[0]}-{kb[1]}_s*.json')
            p = paired(A, B)
            if p:
                verdict = 'ĐẠT' if abs(p['t']) > p['tc'] else 'chưa đạt'
                print(f"    {lbl:16s} {p['mean']:>+6.2f} điểm  SE {p['se']:.2f}  "
                      f"t {p['t']:>6.2f} (ngưỡng {p['tc']})  {verdict}")

# ---------------------------------------------------------------- C
print()
print('=' * 72)
print('C. MARGIN ABLATION — MƯỜI SEED')
print('=' * 72)
M = by_seed('kbf_C_margin_s*.txt')
R = by_seed('kbf_C_random_s*.txt')
for lbl, D in [('margin', M), ('random', R)]:
    if D:
        vals = list(D.values())
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f'  {lbl:8s} n={len(D):<2} Recall@5 {st.mean(vals):>5.1f}±{sd:<4.1f}')
pc = paired(M, R)
if pc:
    print()
    print(f"  ghép cặp: {pc['mean']:+.1f} điểm, SD của hiệu {pc['sd']:.2f}, "
          f"SE {pc['se']:.2f}")
    print(f"  t = {pc['t']:.2f} (ngưỡng {pc['tc']}, df={pc['n']-1}), "
          f"thắng {pc['wins']}/{pc['n']}")
    print(f"  => {'ĐẠT ý nghĩa' if abs(pc['t']) > pc['tc'] else 'CHƯA đạt'}")

# ---------------------------------------------------------------- kiểm chéo
print()
print('=' * 72)
print('PHÉP KIỂM CHÉO: stable/all có bằng headline semantic không?')
print('=' * 72)
print('  Nhóm A chạy ROUTING_MODE=semantic; nhóm B chạy mặc định "auto" và bật')
print('  MEASURE_OVERLAP. Nếu auto giải ra semantic và MEASURE_OVERLAP chỉ đo')
print('  thêm, hai đường chạy phải cho kết quả Y HỆT ở cùng seed.')
print()
A = by_seed('kbf_A_semantic_s*.txt')
B = json_by_seed('kbf_termabl_stable-all_s*.json')
common = sorted(set(A) & set(B), key=int)
if common:
    print(f"  {'seed':>10s} {'nhóm A':>8s} {'nhóm B':>8s} {'chênh':>7s}")
    print('  ' + '-' * 36)
    diffs = []
    for s in common:
        d = A[s] - B[s]
        diffs.append(d)
        flag = '' if abs(d) < 0.05 else '  <-- LỆCH'
        print(f'  {s:>10s} {A[s]:>7.1f}% {B[s]:>7.1f}% {d:>+6.2f}{flag}')
    print()
    mx = max(abs(x) for x in diffs)
    if mx < 0.05:
        print(f'  ✓ Khớp hoàn toàn trên {len(common)} seed chung.')
        print('    Hai đường chạy nhất quán; chênh 76,1 / 76,3 trước đây CHỈ do số seed.')
    else:
        print(f'  ✗ Lệch tối đa {mx:.2f} điểm trên {len(common)} seed.')
        print('    Có gì đó khác giữa hai đường chạy ngoài số seed — phải truy.')
    print()
    print(f'  Trung bình nhóm A ({len(A)} seed): {st.mean(A.values()):.1f}%')
    print(f'  Trung bình nhóm B ({len(B)} seed): {st.mean(B.values()):.1f}%')
    if len(A) == len(B):
        print('  => Hai bảng giờ cùng số seed, số phải trùng nhau trong bài.')
else:
    print('  chưa đủ dữ liệu để so')