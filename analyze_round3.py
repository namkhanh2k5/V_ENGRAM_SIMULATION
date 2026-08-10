#!/usr/bin/env python3
"""
Phân tích kết quả vòng 3.

    python3 analyze_round3.py

Đọc r3_*.txt và result_full_*.json đã sinh. Không chạy lại thí nghiệm.
Chỉ dùng trường đã xác nhận có trong JSON.
"""
import glob
import json
import re
import statistics as st
from collections import defaultdict

# tham chiếu: bản lý tưởng hoá (idealized closest-peer model)
IDEAL = {'semantic': 80.0, 'keyed_lookup': 33.4,
         'random_slots': 34.5, 'random_unique': 22.1}


def read_txt(f):
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    g = lambda p: (float(m.group(1).replace(',', ''))
                   if (m := re.search(p, t)) else None)
    return {'recall': g(r'Recall@5\s*:\s*([\d.]+)%'),
            'reach': g(r'reachable[^:]*:\s*[\d.]+%\s+([\d.]+)%'),
            'rounds': g(r'Rounds/query\s*:\s*([\d,.]+)'),
            'rpc': g(r'RPC/query\s*:\s*([\d,.]+)'),
            'nodes': g(r'Unique nodes contacted\s+([\d,]+)'),
            'cand': g(r'Unique candidates\s+([\d,]+)'),
            'p50': g(r'Latency p50 \(ms\)\s+([\d,.]+)'),
            'bytes': g(r'Bytes/query\s+([\d,]+)')}


def agg(files):
    v = [r for f in files if (r := read_txt(f)) and r['recall'] is not None]
    return v


def mean(v, k):
    x = [d[k] for d in v if d.get(k) is not None]
    return st.mean(x) if x else float('nan')


def sd(v, k):
    x = [d[k] for d in v if d.get(k) is not None]
    return st.stdev(x) if len(x) > 1 else 0.0


# ------------------------------------------------------------------ A
print('=' * 76)
print('A. KẾT QUẢ ITERATIVE VỚI SIMULATOR ĐÃ SỬA')
print('   (lookup tiêu thời gian + 40 probe song song)')
print('=' * 76)
print(f"{'chế độ':16s} {'n':>2s} {'Recall@5':>13s} {'RPC':>7s} {'vòng':>7s} "
      f"{'node':>6s} {'tham chiếu':>11s}")
print('-' * 68)
new = {}
for m_ in ('semantic', 'keyed_lookup', 'random_slots', 'random_unique'):
    v = agg(glob.glob(f'r3_A_{m_}_s*.txt'))
    if not v:
        print(f'{m_:16s} {"(chưa chạy)":>16s}'); continue
    r = mean(v, 'recall')
    new[m_] = r
    print(f"{m_:16s} {len(v):>2} {r:>8.1f}±{sd(v,'recall'):<4.1f} "
          f"{mean(v,'rpc'):>7.0f} {mean(v,'rounds'):>7.0f} "
          f"{mean(v,'nodes'):>6.0f} {IDEAL[m_]:>10.1f}")
if 'semantic' in new:
    print()
    print(f"{'tỉ lệ so':18s} {'đo được':>9s} {'tham chiếu':>11s}")
    print('-' * 42)
    for m_ in ('keyed_lookup', 'random_slots', 'random_unique'):
        if new.get(m_):
            print(f'{m_:18s} {new["semantic"]/new[m_]:>8.2f}x '
                  f'{IDEAL["semantic"]/IDEAL[m_]:>10.2f}x')

# ------------------------------------------------------------------ B
print()
print('=' * 76)
print('B. TERMINATION ABLATION + CHẨN ĐOÁN CƠ CHẾ')
print('=' * 76)
diag = defaultdict(list)
for f in glob.glob('result_full_code_N10000_*_nq500.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    sr, fs = d.get('stop_rule'), d.get('frontier_scope')
    if sr and fs and d.get('jaccard_mean') is not None:
        diag[(sr, fs)].append(d)

res = {}
if diag:
    print(f"{'dừng':9s} {'phạm vi':8s} {'n':>2s} {'Recall@5':>9s} {'reach':>7s} "
          f"{'RPC':>7s} {'ứng viên':>9s} {'Jaccard':>8s} {'XOR rank':>9s}")
    print('-' * 76)
    for sr in ('stable', 'exhaust'):
        for fs in ('all', 'topk'):
            v = diag.get((sr, fs), [])
            if not v:
                print(f'{sr:9s} {fs:8s} {"(chưa chạy)":>12s}'); continue
            # st.mean gãy trên generator rỗng. Trả nan khi không trường nào
            # có khoá đó — đúng lớp lỗi đã gặp ở analyze_axes.py.
            def a(k, _v=v):
                xs = [x[k] for x in _v if x.get(k) is not None]
                return st.mean(xs) if xs else float('nan')

            res[(sr, fs)] = {'r': a('recall_at_5'), 'j': a('jaccard_mean'),
                             'c': a('mean_unique_candidates'),
                             'x': a('xor_rank_mean')}
            print(f"{sr:9s} {fs:8s} {len(v):>2} {a('recall_at_5'):>8.1f}% "
                  f"{a('reachable_recall5'):>6.1f}% "
                  f"{a('disc_rpcs'):>7.0f} {res[(sr,fs)]['c']:>9.0f} "
                  f"{res[(sr,fs)]['j']:>7.3f} {res[(sr,fs)]['x']:>9.1f}")

    if len(res) == 4:
        base = res[('stable', 'all')]
        print()
        print('  TÁCH HAI TRỤC (so với cấu hình hiện dùng, stable/all):')
        print(f"    chỉ đổi ĐIỀU KIỆN DỪNG  "
              f"{res[('exhaust','all')]['r'] - base['r']:>+6.1f} điểm")
        print(f"    chỉ đổi PHẠM VI HỎI     "
              f"{res[('stable','topk')]['r'] - base['r']:>+6.1f} điểm")
        print(f"    đổi cả hai              "
              f"{res[('exhaust','topk')]['r'] - base['r']:>+6.1f} điểm")

        print()
        print('  GIẢ THUYẾT CỦA THẦY: Ripple Search hiện tại cho Jaccard THẤP HƠN')
        print('  (đa dạng cao hơn) + nhiều ứng viên hơn + reachable recall cao hơn.')
        print()
        import math
        js = sorted((res[k]['j'], res[k]['r'], f'{k[0]}/{k[1]}') for k in res
                    if not math.isnan(res[k]['j']) and not math.isnan(res[k]['r']))
        if len(js) < 2:
            print('    (chưa đủ dữ liệu Jaccard để so)')
            js = [(0, 0, '-'), (0, 0, '-')]
        lo_j, lo_r, lo_l = js[0]
        hi_j, hi_r, hi_l = js[-1]
        print(f"    Jaccard thấp nhất : {lo_l:14s} {lo_j:.3f}  ->  recall {lo_r:.1f}%")
        print(f"    Jaccard cao nhất  : {hi_l:14s} {hi_j:.3f}  ->  recall {hi_r:.1f}%")
        print()
        if lo_r > hi_r + 0.5:
            print('    => GIẢ THUYẾT ĐƯỢC ỦNG HỘ: Jaccard thấp đi kèm recall cao.')
            print('       Đa dạng peer set đúng là cơ chế. Đưa bảng này vào bài để')
            print('       biến "bug bị phát hiện" thành design choice có bằng chứng.')
        elif hi_r > lo_r + 0.5:
            print('    => GIẢ THUYẾT KHÔNG ĐƯỢC ỦNG HỘ: Jaccard cao lại đi kèm recall')
            print('       cao. Đa dạng peer set không phải cơ chế. Nên mô tả Ripple')
            print('       Search là xấp xỉ VÌ NGÂN SÁCH, không vì mục tiêu khác.')
        else:
            print('    => Recall gần như không phụ thuộc Jaccard. Chưa kết luận được.')

        xs = sorted((res[k]['x'], res[k]['r'], f'{k[0]}/{k[1]}') for k in res
                    if not math.isnan(res[k]['x']))
        print()
        print('    XOR rank trung bình (thấp = gần tập XOR-gần nhất toàn cục):')
        for x, r, lbl in xs:
            print(f'      {lbl:14s} rank {x:>6.1f}   recall {r:.1f}%')

# ------------------------------------------------------------------ C
print()
print('=' * 76)
print('C. MARGIN-RANKED PROBING ABLATION')
print('=' * 76)
pc = {}
for po in ('margin', 'random'):
    v = agg(glob.glob(f'r3_C_{po}_s*.txt'))
    if v:
        pc[po] = v
        print(f"  {po:8s} n={len(v):<2} Recall@5 {mean(v,'recall'):>5.1f}±{sd(v,'recall'):<4.1f}  "
              f"RPC {mean(v,'rpc'):>7.0f}  node {mean(v,'nodes'):>5.0f}")
if len(pc) == 2:
    dm = mean(pc['margin'], 'recall') - mean(pc['random'], 'recall')
    sdev = max(sd(pc['margin'], 'recall'), sd(pc['random'], 'recall'))
    print()
    print(f"  chênh margin - random: {dm:+.1f} điểm (σ ≈ {sdev:.1f})")
    if dm > 2 * sdev and dm > 0:
        print('  => Heuristic margin CÓ tác dụng, vượt hai lần độ lệch chuẩn.')
        print('     Giữ lập luận về margin trong bài.')
    elif abs(dm) <= 2 * sdev:
        print('  => Chênh KHÔNG vượt 2σ. Heuristic margin chưa chứng minh được là')
        print('     hơn lật bit ngẫu nhiên. Phải hạ giọng lập luận về margin, hoặc')
        print('     chạy thêm seed.')
    else:
        print('  => Lật bit NGẪU NHIÊN lại hơn. Lập luận về margin phải bỏ.')

# ------------------------------------------------------------------ D
print()
print('=' * 76)
print('D. BẢNG CHI PHÍ — ĐƯỜNG TỚI HẠN LÀ CHỈ SỐ CHÍNH')
print('=' * 76)
v = agg(glob.glob('r3_D_cost_s*.txt'))
if v:
    print(f"  n={len(v)} seed")
    for k, lbl, unit in [('rounds', 'vòng/query', ''), ('rpc', 'RPC/query', ''),
                         ('bytes', 'bytes/query', ''), ('nodes', 'peer chạm', ''),
                         ('p50', 'latency p50', 'ms')]:
        m_, s_ = mean(v, k), sd(v, k)
        print(f'  {lbl:14s} {m_:>10,.0f} ± {s_:>6,.0f} {unit}')
    print()
    print('  Theo hướng dẫn của thầy: rounds + RPC + bytes + peer chạm là chỉ số')
    print('  chính; latency tuyệt đối chỉ phụ trợ, vì mô hình mạng còn thiếu')
    print('  bandwidth, queueing và packet loss.')
else:
    print('  chưa có dữ liệu')