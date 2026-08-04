#!/usr/bin/env python3
"""
MC13: crossover r* khác nhau theo baseline mode.

    python3 analyze_mc13.py
"""
import glob
import math
import json
import statistics as st
from collections import defaultdict

MODES = [('random_slots', 'nominal slots', 'L*K*T node, oracle'),
         ('random_keys', 'keyed lookup', 'L*T khoá ngẫu nhiên + lookup thật'),
         ('random_unique', 'equal unique', 'khớp số node phân biệt của semantic')]


def load():
    d = defaultdict(list)
    for f in glob.glob('result_code_N10000_L*_K20_MA*_T8_m512*_nq500.json'):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if r.get('n_query', 0) < 500 or r.get('zipf', 0) > 0 or r.get('node_loss', 0) > 0:
            continue
        mode = r.get('routing_mode') or ('random_slots' if r.get('random_routing') else 'semantic')
        d[(r.get('num_tables', 5), r['meta_anchors'], mode)].append(r['recall5'])
    return d


def main():
    d = load()
    if not d:
        print('Chưa có dữ liệu.')
        return
    Ls = sorted({k[0] for k in d})
    rs = sorted({k[1] for k in d})

    print('=' * 100)
    print('MC13 — CROSSOVER r* PHỤ THUỘC BASELINE MODE')
    print('=' * 100)

    cross = {}
    for mode, lbl, desc in MODES:
        print()
        print(f'--- baseline: {lbl}  ({desc}) ---')
        print(f"{'L':>3s} " + ' '.join(f'r={r:<5}' for r in rs) + '   crossover L·r')
        print('-' * 76)
        for L in Ls:
            row, prev_lr, prev_ratio = [], None, None
            cr = None
            for r in rs:
                sem = d.get((L, r, 'semantic'), [])
                bas = d.get((L, r, mode), [])
                if not sem or not bas:
                    row.append('  --  '); continue
                ratio = st.mean(sem) / st.mean(bas)
                row.append(f'{ratio:6.2f}')
                # Nội suy theo log(L·r) để tìm chỗ ratio cắt 1.
                # prev_lr và prev_ratio luôn được gán cùng nhau, nhưng bộ kiểm
                # kiểu không thấy điều đó nên phải kiểm cả hai tường minh.
                if (prev_ratio is not None and prev_lr is not None
                        and (prev_ratio - 1) * (ratio - 1) < 0):
                    frac = (1 - prev_ratio) / (ratio - prev_ratio)
                    cr = math.exp(math.log(prev_lr)
                                  + frac * (math.log(L * r) - math.log(prev_lr)))
                prev_lr, prev_ratio = L * r, ratio
            crs = f'{cr:.0f}' if cr else ('>' + str(L*rs[-1]) if prev_ratio and prev_ratio > 1
                                          else '<' + str(L*rs[0]))
            cross.setdefault(mode, []).append((L, cr))
            print(f'{L:>3} ' + ' '.join(row) + f'   {crs:>10s}')

    print()
    print('=' * 100)
    print('CROSSOVER THEO MODE — con số phải đưa vào bài')
    print('=' * 100)
    print(f"{'baseline mode':18s} " + ' '.join(f'L={L:<7}' for L in Ls))
    print('-' * 76)
    for mode, lbl, _ in MODES:
        vals = dict(cross.get(mode, []))
        cells = []
        for L in Ls:
            v = vals.get(L)
            cells.append(f'{v:7.0f}' if v else '     --')
        print(f'{lbl:18s} ' + ' '.join(cells))

    print()
    print('  CÁCH VIẾT TRONG BÀI:')
    print('  Không gọi r* là ngưỡng chung. Với mỗi phát biểu phải nêu rõ baseline:')
    print('    "crossover under the nominal-slot baseline"')
    print('  Và thêm B vào hàm: r* = r*(N, L, K, T, W, D, B).')
    print()
    print('  Baseline nào cho M nhỏ hơn thì P_rand nhỏ hơn, nên crossover LÙI RA XA:')
    print('  equal-unique (M~492) < keyed lookup (M~770) < nominal slots (M=800).')


if __name__ == '__main__':
    main()