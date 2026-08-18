#!/usr/bin/env python3
"""Phân tích kết quả bootstrap join, đối chiếu với oracle."""
import glob, json, math, re, statistics as st
from collections import defaultdict

# số cũ đo với PQ m=256 (main_simulation.py hardcode bản không suffix)
ORACLE = {'semantic': 71.8, 'keyed_lookup': 33.7, 'random_slots': 33.6,
          'random_unique': 22.1,
          ('stable','all'): 71.8, ('stable','topk'): 67.0,
          ('exhaust','all'): 75.1, ('exhaust','topk'): 64.4,
          'margin': 71.8, 'random': 63.8}


def read(f):
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    g = lambda p: (float(m.group(1).replace(',', '')) if (m := re.search(p, t)) else None)
    return {'recall': g(r'Recall@5\s*:\s*([\d.]+)%'),
            'rpc': g(r'RPC/query\s*:\s*([\d,.]+)'),
            'rounds': g(r'Rounds/query\s*:\s*([\d,.]+)'),
            'nodes': g(r'Unique nodes contacted\s+([\d,]+)'),
            'contact': g(r'contact/peer: TB (\d+)'),
            'bytes': g(r'Bytes/query\s+([\d,]+)'),
            'p50': g(r'Latency p50 \(ms\)\s+([\d,.]+)')}


def agg(pat):
    v = [x for f in glob.glob(pat) if (x := read(f)) and x['recall'] is not None]
    return v


def m(v, k):
    xs = [d[k] for d in v if d.get(k) is not None]
    return st.mean(xs) if xs else float('nan')


def sd(v, k):
    xs = [d[k] for d in v if d.get(k) is not None]
    return st.stdev(xs) if len(xs) > 1 else 0.0


def json_by_seed(pat, key='recall_at_5'):
    d = {}
    for f in glob.glob(pat):
        mm = re.search(r'_s(\d+)\.json$', f)
        if not mm:
            continue
        try:
            val = json.load(open(f)).get(key)
        except Exception:
            continue
        if val is not None:
            d[mm.group(1)] = val
    return d


def txt_by_seed(pat, key='recall'):
    """Đọc {seed: giá trị} từ file .txt. Bỏ qua file không khớp tên hoặc lỗi.

    Bản trước gọi re.search(...).group(1) thẳng trong dict comprehension — nếu
    tên file không khớp thì gãy. Đây là lần thứ tư cùng lỗi trong dự án, nên
    mọi chỗ đọc seed từ tên file giờ đi qua hàm có guard.
    """
    d = {}
    for f in glob.glob(pat):
        mm = re.search(r'_s(\d+)\.txt$', f)
        if not mm:
            continue
        x = read(f)
        if x and x.get(key) is not None:
            d[mm.group(1)] = x[key]
    return d


def paired(A, B):
    d = [A[s] - B[s] for s in sorted(set(A) & set(B))]
    if len(d) < 2:
        return None
    mm, ss = st.mean(d), st.stdev(d)
    se = ss / math.sqrt(len(d))
    tc = {1:12.71,2:4.30,3:3.18,4:2.78,5:2.57,6:2.45,7:2.36,8:2.31,9:2.26}.get(len(d)-1, 2.0)
    t = mm/se if se else float('nan')
    # p-value hai phía cho phân bố t, tính bằng hàm beta không cần scipy.
    # Thầy yêu cầu exact p-value ở mục 6.
    nu = len(d) - 1
    try:
        x = nu / (nu + t*t)
        # I_x(nu/2, 1/2) qua liên phân số Lentz
        def betacf(a, b, x, it=200):
            qab, qap, qam = a+b, a+1.0, a-1.0
            c, d_, h = 1.0, 1.0 - qab*x/qap, 1.0
            d_ = 1e-30 if abs(d_) < 1e-30 else d_
            d_, h = 1.0/d_, 1.0/d_
            for m_ in range(1, it):
                m2 = 2*m_
                aa = m_*(b-m_)*x/((qam+m2)*(a+m2))
                d_ = 1.0 + aa*d_; c = 1.0 + aa/c
                d_ = 1e-30 if abs(d_) < 1e-30 else d_
                c = 1e-30 if abs(c) < 1e-30 else c
                d_ = 1.0/d_; h *= d_*c
                aa = -(a+m_)*(qab+m_)*x/((a+m2)*(qap+m2))
                d_ = 1.0 + aa*d_; c = 1.0 + aa/c
                d_ = 1e-30 if abs(d_) < 1e-30 else d_
                c = 1e-30 if abs(c) < 1e-30 else c
                d_ = 1.0/d_
                if abs(d_*c - 1.0) < 3e-9:
                    h *= d_*c; break
                h *= d_*c
            return h
        a, b = nu/2.0, 0.5
        lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b))
        if x < (a+1)/(a+b+2):
            ib = math.exp(a*math.log(x) + b*math.log(1-x) - lbeta) * betacf(a, b, x) / a
        else:
            ib = 1 - math.exp(b*math.log(1-x) + a*math.log(x) - lbeta) * betacf(b, a, 1-x) / b
        pval = max(min(ib, 1.0), 0.0)
    except Exception:
        pval = float('nan')
    return {'n': len(d), 'mean': mm, 'sd': ss, 'se': se, 't': t, 'tc': tc,
            'p': pval, 'wins': sum(1 for x in d if x > 0),
            'ci': (mm - tc*se, mm + tc*se)}


print('=' * 74)
print('A. HEADLINE VỚI BOOTSTRAP JOIN')
print('=' * 74)
print(f"{'chế độ':16s} {'n':>2s} {'Recall@5':>13s} {'m256':>7s} {'Δ':>6s} "
      f"{'RPC':>7s} {'contact':>8s}")
print('-' * 66)
new = {}
for mo in ('semantic', 'keyed_lookup', 'random_slots', 'random_unique'):
    v = agg(f'q5_A_{mo}_s*.txt')
    if not v:
        print(f'{mo:16s} {"(chưa chạy)":>16s}'); continue
    r = m(v, 'recall'); new[mo] = r
    o = ORACLE[mo]
    print(f"{mo:16s} {len(v):>2} {r:>8.1f}±{sd(v,'recall'):<4.1f} {o:>7.1f} "
          f"{r-o:>+5.1f} {m(v,'rpc'):>7.0f} {m(v,'contact'):>8.0f}")
if 'semantic' in new:
    print()
    for mo in ('keyed_lookup', 'random_slots', 'random_unique'):
        if new.get(mo):
            print(f'  tỉ lệ vs {mo:16s} {new["semantic"]/new[mo]:>6.2f}x  '
                  f'(m256 {ORACLE["semantic"]/ORACLE[mo]:.2f}x)')

print()
print('=' * 74)
print('B. TERMINATION ABLATION VỚI BOOTSTRAP JOIN')
print('=' * 74)
diag = defaultdict(list)
for f in glob.glob('q5_termabl_*.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if (sr := d.get('stop_rule')) and (fs := d.get('frontier_scope')):
        diag[(sr, fs)].append(d)
if diag:
    print(f"{'dừng':9s} {'phạm vi':8s} {'n':>3s} {'Recall@5':>9s} {'m256':>7s} "
          f"{'Reach':>7s} {'RPC':>7s} {'XOR rank':>9s} {'rt RPC':>8s} "
          f"{'ev RPC':>7s} {'Jaccard':>8s}")
    print('-' * 68)
    for sr in ('stable', 'exhaust'):
        for fs in ('all', 'topk'):
            v = diag.get((sr, fs), [])
            if not v:
                print(f'{sr:9s} {fs:8s} {"(chưa chạy)":>13s}'); continue
            a = lambda k, _v=v: (st.mean([x[k] for x in _v if x.get(k) is not None])
                                 if any(x.get(k) is not None for x in _v) else float('nan'))
            print(f"{sr:9s} {fs:8s} {len(v):>3} {a('recall_at_5'):>8.1f}% "
                  f"{ORACLE[(sr,fs)]:>6.1f}% {a('reachable_recall5'):>6.1f}% "
                  f"{a('disc_rpcs'):>7.0f} {a('xor_rank_mean'):>9.2f} "
                  f"{a('rpc_routing_mean'):>8.0f} {a('rpc_eval_mean'):>7.0f} "
                  f"{a('jaccard_mean'):>8.4f}")
    print()
    print('  XOR rank với oracle là 9,50 ở CẢ BỐN cấu hình — đúng mean của [0..19],')
    print('  tức lookup tìm đúng global top-20. Nếu join cho số KHÁC 9,50 thì con')
    print('  số cũ đúng là artifact của bootstrap.')
    print()
    for ka, kb, lbl in [(('exhaust','all'), ('stable','all'), 'ĐIỀU KIỆN DỪNG'),
                        (('stable','topk'), ('stable','all'), 'PHẠM VI HỎI')]:
        p = paired(json_by_seed(f'q5_termabl_{ka[0]}-{ka[1]}_s*.json'),
                   json_by_seed(f'q5_termabl_{kb[0]}-{kb[1]}_s*.json'))
        if p:
            v = 'ĐẠT' if abs(p['t']) > p['tc'] else 'chưa đạt'
            print(f"  {lbl:16s} {p['mean']:>+6.2f} điểm  CI95 [{p['ci'][0]:+.2f}, "
                  f"{p['ci'][1]:+.2f}]  t {p['t']:>6.2f}  p {p['p']:.2g}  {v}")

print()
print('=' * 74)
print('C. MARGIN ABLATION VỚI BOOTSTRAP JOIN')
print('=' * 74)
M = txt_by_seed('q5_C_margin_s*.txt')
R = txt_by_seed('q5_C_random_s*.txt')
for lbl, D, o in [('margin', M, 76.1), ('random', R, 67.5)]:
    if D:
        vals = list(D.values())
        s_ = st.stdev(vals) if len(vals) > 1 else 0.0
        print(f'  {lbl:8s} n={len(D):<3} Recall@5 {st.mean(vals):>5.1f}±{s_:<4.1f}  '
              f'(m256 {o:.1f})')
pc = paired(M, R)
if pc:
    print()
    print(f"  ghép cặp: {pc['mean']:+.2f} điểm, CI95 [{pc['ci'][0]:+.2f}, {pc['ci'][1]:+.2f}]")
    print(f"  t = {pc['t']:.2f}, p = {pc['p']:.3g} (df={pc['n']-1}), "
          f"thắng {pc['wins']}/{pc['n']}")
    print(f"  => {'ĐẠT ý nghĩa' if abs(pc['t']) > pc['tc'] else 'CHƯA đạt'}")

print()
print('=' * 74)
print('D. BẢNG CHI PHÍ')
print('=' * 74)
v = [x for f in glob.glob('q5_D_cost_s*.txt') if (x := read(f)) and x['rpc']]
if v:
    print(f'  n={len(v)} seed')
    for k, lbl, u in [('rounds','vòng/query',''), ('rpc','RPC/query',''),
                      ('bytes','bytes/query',''), ('nodes','peer chạm',''),
                      ('p50','latency p50','ms')]:
        print(f'  {lbl:14s} {m(v,k):>10,.0f} ± {sd(v,k):>6,.0f} {u}')
else:
    print('  chưa có dữ liệu')