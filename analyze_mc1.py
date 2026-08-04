#!/usr/bin/env python3
"""
Phân tích MC1: headline dưới discrete-event, kèm CI như thầy yêu cầu.

    python3 analyze_mc1.py
"""
import glob
import random
import re
import statistics as st

MODES = [('semantic', 'Semantic'),
         ('keyed_lookup', 'Random, keyed lookup'),
         ('random_slots', 'Random, nominal slots'),
         ('random_unique', 'Random, equal unique')]

# Số từ recall-only simulator, để so upper bound với thực tế
IDEAL = {'semantic': 80.0, 'keyed_lookup': 33.4,
         'random_slots': 34.5, 'random_unique': 22.1}


def read(mode):
    out = []
    for f in sorted(glob.glob(f'mc1_{mode}_s*.txt')):
        t = open(f, encoding='utf-8', errors='ignore').read()
        g = lambda p: (float(m.group(1).replace(',', ''))
                       if (m := re.search(p, t)) else None)
        r = g(r'Recall@5\s*:\s*([\d.]+)%')
        # re.search trả Match|None. Gọi thẳng .group(1) vừa làm bộ kiểm kiểu báo
        # lỗi, vừa CRASH THẬT nếu tên file không khớp mẫu (ví dụ ai đổi tên tay).
        ms = re.search(r'_s(\d+)\.txt$', f)
        if r is not None and ms is not None:
            out.append({'seed': ms.group(1),
                        'recall': r,
                        'nodes': g(r'Unique nodes contacted\s+([\d,]+)'),
                        'rpc': g(r'RPC/query\s*:\s*([\d,.]+)')})
        elif r is not None:
            print(f'  [bỏ qua] {f}: tên file không có _s<seed>.txt')
    return out


def boot_ci(vals, n=10000, alpha=0.05):
    """CI bootstrap cho trung bình. Với 10 seed, t-CI cũng được, nhưng bootstrap
    không giả định phân bố chuẩn — an toàn hơn khi n nhỏ."""
    if len(vals) < 2:
        return (float('nan'), float('nan'))
    rnd = random.Random(0)
    means = sorted(st.mean(rnd.choices(vals, k=len(vals))) for _ in range(n))
    return (means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n)])


def main():
    data = {m: read(m) for m, _ in MODES}
    if not any(data.values()):
        print('Chưa có file mc1_*.txt. Chạy run_mc1_headline.sh trước.')
        return

    print('=' * 100)
    print('MC1 — HEADLINE DƯỚI DISCRETE-EVENT (walk Kademlia lặp thật)')
    print('=' * 100)
    print(f"{'chế độ':24s} {'n':>3s} {'Recall@5':>16s} {'CI 95%':>16s} "
          f"{'node':>7s} {'RPC':>8s} {'upper bound':>12s}")
    print('-' * 100)

    for mode, lbl in MODES:
        d = data[mode]
        if not d:
            print(f'{lbl:24s} {"(chưa chạy)":>3s}')
            continue
        v = [x['recall'] for x in d]
        m, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
        lo, hi = boot_ci(v)
        nd = st.mean(x['nodes'] for x in d if x['nodes'])
        rp = st.mean(x['rpc'] for x in d if x['rpc'])
        ub = IDEAL.get(mode)
        ubs = f'{ub:.1f} ({m-ub:+.1f})' if ub else '---'
        print(f'{lbl:24s} {len(v):>3} {m:>9.1f}±{sd:<5.1f} '
              f'[{lo:>5.1f},{hi:>5.1f}] {nd:>7.0f} {rp:>8.0f} {ubs:>12s}')

    # tỉ lệ với CI, dùng bootstrap ghép cặp theo seed
    sem = data['semantic']
    if sem:
        print()
        print('=' * 100)
        print('TỈ LỆ semantic / baseline, bootstrap ghép cặp theo seed')
        print('=' * 100)
        by_seed = {x['seed']: x['recall'] for x in sem}
        print(f"{'baseline':24s} {'tỉ lệ':>8s} {'CI 95%':>16s} {'lý tưởng':>10s}")
        print('-' * 100)
        for mode, lbl in MODES[1:]:
            d = data[mode]
            if not d:
                continue
            pairs = [(by_seed[x['seed']], x['recall'])
                     for x in d if x['seed'] in by_seed and x['recall'] > 0]
            if not pairs:
                continue
            ratios = [a / b for a, b in pairs]
            r_m = st.mean(ratios)
            lo, hi = boot_ci(ratios)
            ub = IDEAL['semantic'] / IDEAL[mode] if IDEAL.get(mode) else None
            print(f'{lbl:24s} {r_m:>7.2f}x [{lo:>6.2f},{hi:>6.2f}] '
                  f'{ub:>9.2f}x' if ub else '')

    print()
    print('  CÁCH DÙNG SỐ NÀY TRONG BÀI:')
    print('  Cột "upper bound" là số của recall-only simulator. Con số discrete-event')
    print('  mới là hiệu năng dưới định tuyến thật, và phải là số dùng ở Abstract,')
    print('  Main Result, Conclusion. Số upper bound giữ lại nhưng gọi đúng tên:')
    print('  "idealized closest-peer upper bound".')


if __name__ == '__main__':
    main()