#!/usr/bin/env python3
"""
Phân tích weak scaling: giữ tỉ lệ doc/node cố định, tăng cả hai.

Phân biệt rõ với N-sweep:
  N-sweep      : corpus CỐ ĐỊNH, node tăng -> doc/node GIẢM -> kiểm công thức r*
  weak scaling : corpus VÀ node cùng tăng -> doc/node CỐ ĐỊNH -> kiểm scalability

    python3 analyze_weakscaling.py
"""
import glob
import json
import statistics as st
from collections import defaultdict

# (dataset, số node) theo tỉ lệ 2 doc/node
POINTS = [('code', 10000, 20000), ('code50k', 25000, 50000), ('code100k', 50000, 100000)]


def main():
    data = defaultdict(list)
    for f in glob.glob('result_*_L5_K20_MA1_T8_m512*_nq500.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('n_query', 0) < 500 or d.get('zipf', 0) > 0 or d.get('node_loss', 0) > 0:
            continue
        mode = d.get('routing_mode') or ('random_slots' if d.get('random_routing') else 'semantic')
        if mode not in ('semantic', 'random_slots'):
            continue
        data[(d['dataset'], d.get('nodes'), mode)].append(d)

    print("=" * 96)
    print("WEAK SCALING — tỉ lệ doc/node CỐ ĐỊNH ở 2, tăng cả corpus và mạng")
    print("=" * 96)
    print(f"{'nodes':>7s} {'docs':>8s} {'d/n':>4s} {'n':>3s} {'semantic':>13s} {'random':>13s} "
          f"{'tỉ lệ':>6s} {'node%':>7s} {'cand%':>7s}")
    print("-" * 96)

    rows = []
    for ds, nodes, docs in POINTS:
        sem = data.get((ds, nodes, 'semantic'), [])
        rnd = data.get((ds, nodes, 'random_slots'), [])
        if not sem:
            print(f"{nodes:>7,} {docs:>8,} {docs/nodes:>4.0f} {'--':>3s}  (chưa có dữ liệu)")
            continue
        s_m = st.mean(x['recall5'] for x in sem)
        s_s = st.stdev([x['recall5'] for x in sem]) if len(sem) > 1 else 0.0
        r_m = st.mean(x['recall5'] for x in rnd) if rnd else float('nan')
        r_s = st.stdev([x['recall5'] for x in rnd]) if len(rnd) > 1 else 0.0
        ratio = s_m / r_m if rnd and r_m > 0 else float('nan')
        nd = st.mean(x['pct_network_touched'] for x in sem)
        cd = st.mean(100.0 * x['mean_unique_candidates'] / docs for x in sem)
        rs = f"{ratio:>6.2f}" if ratio == ratio else f"{'--':>6}"
        rstr = f"{r_m:>7.1f}±{r_s:<4.1f}" if rnd else f"{'--':>13}"
        print(f"{nodes:>7,} {docs:>8,} {docs/nodes:>4.0f} {len(sem):>3} "
              f"{s_m:>7.1f}±{s_s:<4.1f} {rstr} {rs} {nd:>6.1f}% {cd:>6.1f}%")
        rows.append({'nodes': nodes, 'docs': docs, 'sem': s_m, 'rnd': r_m,
                     'ratio': ratio, 'node_pct': nd, 'cand_pct': cd})

    if len(rows) < 2:
        print("\nChưa đủ điểm để kết luận. Cần ít nhất 2 mức.")
        return

    print()
    print("=" * 96)
    print("KẾT LUẬN")
    print("=" * 96)
    a, b = rows[0], rows[-1]
    fold = b['docs'] / a['docs']
    print(f"  Từ {a['nodes']:,} node / {a['docs']:,} doc  →  "
          f"{b['nodes']:,} node / {b['docs']:,} doc  ({fold:.0f}× lớn hơn)")
    print(f"    semantic recall : {a['sem']:.1f}% → {b['sem']:.1f}%  ({b['sem']-a['sem']:+.1f}đ)")
    if b['rnd'] == b['rnd']:
        print(f"    random recall   : {a['rnd']:.1f}% → {b['rnd']:.1f}%  ({b['rnd']-a['rnd']:+.1f}đ)")
        print(f"    tỉ lệ sem/rand  : {a['ratio']:.2f}× → {b['ratio']:.2f}×")
    print(f"    node chạm       : {a['node_pct']:.1f}% → {b['node_pct']:.1f}% mạng")
    print(f"    ứng viên        : {a['cand_pct']:.1f}% → {b['cand_pct']:.1f}% corpus")
    print()
    drop = a['sem'] - b['sem']
    if abs(drop) < 3:
        print("  => Recall GIỮ ĐƯỢC khi hệ lớn lên ở tỉ lệ cố định. Đây là weak scaling ĐẠT.")
    elif drop > 0:
        print(f"  => Recall GIẢM {drop:.1f} điểm. Hệ KHÔNG giữ được chất lượng ở quy mô lớn hơn;")
        print("     phải nêu rõ và tìm nguyên nhân (nhiễu tăng? ngân sách node cố định?).")
    else:
        print(f"  => Recall TĂNG {-drop:.1f} điểm khi hệ lớn lên. Cần giải thích:")
        print("     có thể corpus lớn hơn cho codebook PQ tốt hơn.")
    print()
    print("  LƯU Ý ĐỌC SỐ: ground truth KHÁC NHAU giữa các mức, vì top-10 của một")
    print("  query trong corpus 100.000 khác trong corpus 20.000. Task khó dần theo")
    print("  thiết kế — đó chính là điều weak scaling kiểm tra.")


if __name__ == '__main__':
    main()
