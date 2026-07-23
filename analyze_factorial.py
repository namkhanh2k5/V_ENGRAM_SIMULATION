#!/usr/bin/env python3
"""
Phân tích factorial grid L × r (mục 14).

Câu hỏi: tỉ lệ semantic/random có phụ thuộc DUY NHẤT vào tích L·r không, hay
còn phụ thuộc riêng vào L và r?

Cách kiểm: gom các cấu hình có CÙNG L·r nhưng KHÁC cách phân tách, rồi xem tỉ lệ
của chúng có bằng nhau không. Nếu bằng -> tích L·r là biến giải thích đủ.
Nếu khác -> phải phát biểu lại.

Cũng in P_rand dự đoán từ công thức để đối chiếu với random recall đo được.

    python3 analyze_factorial.py
"""
import glob
import json
import statistics as st
from collections import defaultdict

N_NODES, K, T = 10000, 20, 8


def p_rand_formula(L, r, N=N_NODES, K=K, T=T):
    """P_rand = 1 - (1 - L·r/N)^(K·L·T). Lưu ý L có ở CẢ cơ số lẫn số mũ."""
    return 100.0 * (1 - (1 - L * r / N) ** (K * L * T))


def main():
    # gom kết quả: (L, r, mode) -> list recall
    data = defaultdict(list)
    nodes = defaultdict(list)
    for f in glob.glob('result_code_N10000_*_T8_m512*_nq500.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('n_query', 0) < 500 or d.get('zipf', 0) > 0:
            continue
        mode = d.get('routing_mode') or ('random_slots' if d.get('random_routing') else 'semantic')
        if mode not in ('semantic', 'random_slots'):
            continue
        L, r = d.get('num_tables', 5), d['meta_anchors']
        data[(L, r, mode)].append(d['recall5'])
        nodes[(L, r, mode)].append(d['pct_network_touched'])

    # bảng đầy đủ theo L·r
    print("=" * 86)
    print("FACTORIAL GRID L × r — sắp theo tích L·r")
    print("=" * 86)
    print(f"{'L':>3} {'r':>3} {'L·r':>4} {'n':>3} {'semantic':>12} {'random':>12} "
          f"{'tỉ lệ':>7} {'P_rand lt':>10} {'node%':>7}")
    print("-" * 86)

    rows = []
    for (L, r, mode), vals in sorted(data.items(), key=lambda x: (x[0][0] * x[0][1], x[0][0])):
        if mode != 'semantic':
            continue
        rnd = data.get((L, r, 'random_slots'), [])
        if not rnd:
            continue
        sem_m = st.mean(vals)
        rnd_m = st.mean(rnd)
        n = min(len(vals), len(rnd))
        ratio = sem_m / rnd_m if rnd_m > 0 else float('nan')
        sem_s = st.stdev(vals) if len(vals) > 1 else 0.0
        rnd_s = st.stdev(rnd) if len(rnd) > 1 else 0.0
        nd = st.mean(nodes.get((L, r, 'semantic'), [0]))
        rows.append({'L': L, 'r': r, 'Lr': L * r, 'n': n, 'sem': sem_m, 'sem_sd': sem_s,
                     'rnd': rnd_m, 'rnd_sd': rnd_s, 'ratio': ratio,
                     'pred': p_rand_formula(L, r), 'node': nd})
        print(f"{L:>3} {r:>3} {L*r:>4} {n:>3} {sem_m:>7.1f}±{sem_s:<4.1f} "
              f"{rnd_m:>7.1f}±{rnd_s:<4.1f} {ratio:>7.2f} {p_rand_formula(L, r):>9.1f}% {nd:>6.1f}%")

    # so các cách phân tách CÙNG L·r
    print()
    print("=" * 86)
    print("CÁC CÁCH PHÂN TÁCH CÙNG L·r — tỉ lệ có bằng nhau không?")
    print("=" * 86)
    by_lr = defaultdict(list)
    for row in rows:
        by_lr[row['Lr']].append(row)

    verdicts = []
    for Lr in sorted(by_lr):
        group = by_lr[Lr]
        if len(group) < 2:
            continue
        print(f"\nL·r = {Lr}:")
        print(f"  {'phân tách':>12} {'semantic':>10} {'random':>10} {'tỉ lệ':>7} {'P_rand lt':>10}")
        for g in sorted(group, key=lambda x: x['L']):
            print(f"  {'L=%d, r=%d' % (g['L'], g['r']):>12} {g['sem']:>9.1f}% "
                  f"{g['rnd']:>9.1f}% {g['ratio']:>7.2f} {g['pred']:>9.1f}%")
        ratios = [g['ratio'] for g in group]
        spread = max(ratios) - min(ratios)
        sem_spread = max(g['sem'] for g in group) - min(g['sem'] for g in group)
        rnd_spread = max(g['rnd'] for g in group) - min(g['rnd'] for g in group)
        print(f"  -> chênh lệch tỉ lệ: {spread:.2f}  "
              f"(semantic chênh {sem_spread:.1f}đ, random chênh {rnd_spread:.1f}đ)")
        verdicts.append((Lr, spread, sem_spread, rnd_spread))

    # kết luận
    print()
    print("=" * 86)
    print("KẾT LUẬN")
    print("=" * 86)
    if not verdicts:
        print("  Chưa đủ cặp cùng L·r để kết luận. Cần chạy thêm.")
        return
    max_spread = max(v[1] for v in verdicts)
    print(f"  Chênh lệch tỉ lệ LỚN NHẤT giữa các cách phân tách cùng L·r: {max_spread:.2f}")
    if max_spread < 0.15:
        print("  => Tỉ lệ gần như KHÔNG đổi khi phân tách khác nhau.")
        print("     Tích L·r là biến giải thích đủ. Luận điểm hiện tại ĐỨNG VỮNG.")
    else:
        print("  => Tỉ lệ THAY ĐỔI rõ theo cách phân tách.")
        print("     Tích L·r KHÔNG đủ giải thích. Phải phát biểu lại:")
        print("     P_rand phụ thuộc cả dấu vết metadata (L·r) LẪN ngân sách tiếp xúc")
        print("     (K·L·T), và cả hai đều tăng theo L, nên L không phải trục tự do.")
    print()
    print("  Đối chiếu công thức: cột 'P_rand lt' là dự đoán, cột 'random' là đo được.")
    print("  Lệch hệ thống là do công thức giả định K·L·T node UNIQUE, trong khi số")
    print("  node thực chạm nhỏ hơn (mục 13 — cần A, M thực).")


if __name__ == '__main__':
    main()