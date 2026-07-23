#!/usr/bin/env python3
"""
ĐO PHÂN BỐ SỐ BIT LỆCH giữa sketch của query và sketch của neighbor THẬT.

Trả lời câu hỏi: cơ chế probe hiện tại (lật MỘT bit mỗi lần) phủ được bao nhiêu
phần của khối lượng neighbor?

  - Probe lật 1 bit chỉ với tới neighbor lệch <= 1 bit trong c bit đầu.
  - Nếu mode của phân bố nằm ở 2-3 bit, cơ chế single-bit đang chạm trần
    và multi-bit schedule là hướng mở rộng có căn cứ.

Đây là phép đo THUẦN TRÊN VECTOR — không chạy mô phỏng mạng, không cần node.
Chạy vài phút.

    python3 measure_bitflips.py --dataset code
    python3 measure_bitflips.py --dataset scifact --c 16 --seeds 5

Ghi kết quả ra bitflips_{dataset}.json và in bảng.
"""
import argparse
import json
import random
from collections import Counter

import numpy as np


def build_projection(seed, d=1024, bits=63):
    """Sinh ma trận chiếu GIỐNG HỆT main_simulation_v2.py để số đo khớp mô phỏng.

    Achlioptas thưa: -1 với xác suất 1/6, 0 với 2/3, +1 với 1/6.
    """
    rnd = np.random.RandomState(seed)
    return rnd.choice([-1.0, 0.0, 1.0], size=(d, bits), p=[1 / 6, 2 / 3, 1 / 6])


def sketch_bits(vectors, proj, c):
    """Trả về ma trận bit (n, c) — dấu của c phép chiếu đầu."""
    return (np.asarray(vectors) @ proj[:, :c] > 0).astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='code', choices=['code', 'scifact', 'squad'])
    ap.add_argument('--c', type=int, default=16,
                    help='Số bit đầu được phép lật khi probe (mặc định 16, khớp code)')
    ap.add_argument('--seeds', type=int, default=5,
                    help='Số ma trận chiếu độc lập để lấy trung bình')
    ap.add_argument('--topk', type=int, default=5,
                    help='Đo với top-k neighbor thật (mặc định 5)')
    args = ap.parse_args()

    # Đường dẫn khớp main_simulation_v2.py: ./data/{dataset}_*
    D = f'./data/{args.dataset}'
    print(f"[*] Nạp dữ liệu {args.dataset} từ {D}_* ...")
    E = np.load(f'{D}_corpus_embeddings.npy')       # (N, 1024) đã normalize
    Qv = np.load(f'{D}_query_embeddings.npy')       # (nq, 1024) đã normalize
    gt = json.load(open(f'{D}_ground_truth.json', encoding='utf-8'))
    print(f"    corpus={len(E):,} | query={len(gt)} | c={args.c} | seeds={args.seeds}")

    # Gom (query_idx, neighbor_idx) cho mọi cặp query - neighbor thật
    pairs = []
    for qi, item in enumerate(gt):
        for r in item['top_5_results'][:args.topk]:
            pairs.append((qi, r['index']))
    print(f"    số cặp query-neighbor: {len(pairs):,}")

    q_idx = np.array([p[0] for p in pairs])
    n_idx = np.array([p[1] for p in pairs])

    # Góc trung bình để đối chiếu với dự đoán lý thuyết
    cos = np.sum(Qv[q_idx] * E[n_idx], axis=1).clip(-1, 1)
    ang = np.degrees(np.arccos(cos))
    print(f"    góc trung bình query-neighbor: {ang.mean():.1f}°")

    # Đếm số bit lệch, gộp qua nhiều ma trận chiếu
    counter = Counter()
    total = 0
    for s in range(args.seeds):
        proj = build_projection(20235956 + s)
        bq = sketch_bits(Qv[q_idx], proj, args.c)
        bn = sketch_bits(E[n_idx], proj, args.c)
        diff = np.sum(bq != bn, axis=1)      # số bit lệch trong c bit đầu
        counter.update(diff.tolist())
        total += len(diff)
        print(f"    seed {s}: bit lệch trung bình = {diff.mean():.2f}")

    # Dự đoán lý thuyết: Binomial(c, theta/180)
    p_flip = ang.mean() / 180.0
    print(f"\n    xác suất lệch mỗi bit (lý thuyết) = {p_flip:.3f}")
    print(f"    số bit lệch kỳ vọng = {args.c * p_flip:.2f}")

    from math import comb
    print("\n" + "=" * 72)
    print(f"PHÂN BỐ SỐ BIT LỆCH — {args.dataset}, c={args.c}, {args.seeds} ma trận chiếu")
    print("=" * 72)
    print(f"{'k bit lệch':>11} {'đo được':>10} {'lý thuyết':>10} {'tích luỹ':>10}   ghi chú")
    print("-" * 72)

    cum = 0.0
    rows = []
    for k in range(args.c + 1):
        emp = 100.0 * counter.get(k, 0) / total
        theo = 100.0 * comb(args.c, k) * (p_flip ** k) * ((1 - p_flip) ** (args.c - k))
        cum += emp
        note = ""
        if k == 0:
            note = "<- key gốc bắt được"
        elif k == 1:
            note = "<- single-bit probe bắt được"
        elif k == 2:
            note = "<- CẦN probe 2 bit"
        elif k == 3:
            note = "<- CẦN probe 3 bit"
        if emp >= 0.05 or k <= 4:
            print(f"{k:>11} {emp:>9.1f}% {theo:>9.1f}% {cum:>9.1f}%   {note}")
        rows.append({'k': k, 'empirical_pct': emp, 'theory_pct': theo, 'cumulative_pct': cum})

    cov1 = rows[0]['empirical_pct'] + rows[1]['empirical_pct']
    cov2 = cov1 + rows[2]['empirical_pct']
    cov3 = cov2 + rows[3]['empirical_pct']
    mode = max(range(args.c + 1), key=lambda k: counter.get(k, 0))

    print("-" * 72)
    print(f"  Mode (số bit lệch phổ biến nhất): k = {mode}")
    print(f"  Phủ bởi cơ chế HIỆN TẠI (k<=1)  : {cov1:.1f}%")
    print(f"  Nếu probe tới 2 bit (k<=2)      : {cov2:.1f}%  (+{cov2-cov1:.1f} điểm)")
    print(f"  Nếu probe tới 3 bit (k<=3)      : {cov3:.1f}%  (+{cov3-cov1:.1f} điểm)")
    print()
    print(f"  Số probe cần cho k<=1: {1 + args.c} (trần cứng của cơ chế hiện tại)")
    print(f"  Số probe cần cho k<=2: {1 + args.c + args.c*(args.c-1)//2}")
    print("=" * 72)
    print()
    print("LƯU Ý ĐỌC SỐ: đây là xác suất trúng ĐÚNG subtree của một neighbor qua")
    print("MỘT bảng. Recall thật cao hơn nhiều vì (a) có L bảng khuếch đại OR,")
    print("(b) K=20 node quanh mỗi target nên không cần trúng chính xác. Con số")
    print("dùng để so sánh TƯƠNG ĐỐI giữa các mức k, không phải dự đoán recall.")

    out = {
        'dataset': args.dataset, 'c': args.c, 'seeds': args.seeds, 'topk': args.topk,
        'num_pairs': len(pairs), 'mean_angle_deg': float(ang.mean()),
        'p_flip_per_bit': float(p_flip), 'mode_k': int(mode),
        'coverage_k_le_1': float(cov1), 'coverage_k_le_2': float(cov2),
        'coverage_k_le_3': float(cov3), 'distribution': rows,
    }
    fn = f'bitflips_{args.dataset}.json'
    json.dump(out, open(fn, 'w'), indent=2)
    print(f"-> Lưu: {fn}")


if __name__ == '__main__':
    main()