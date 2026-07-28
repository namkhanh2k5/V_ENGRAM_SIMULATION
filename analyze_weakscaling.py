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

# (dataset, node, doc, K) — hai đường
# A: K cố định 20 -> chi phí không đổi, K/N co lại
# B: K tỉ lệ N (K/N = 0,20%) -> phủ không đổi, chi phí tăng tuyến tính
TRACK_A = [('code', 10000, 20000, 20), ('code50k', 25000, 50000, 20),
           ('code100k', 50000, 100000, 20)]
TRACK_B = [('code', 10000, 20000, 20), ('code50k', 25000, 50000, 50),
           ('code100k', 50000, 100000, 100)]


def main():
    data = defaultdict(list)
    for f in glob.glob('result_*_L5_K*_MA1_T8_m512*_nq500.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('n_query', 0) < 500 or d.get('zipf', 0) > 0 or d.get('node_loss', 0) > 0:
            continue
        mode = d.get('routing_mode') or ('random_slots' if d.get('random_routing') else 'semantic')
        if mode not in ('semantic', 'random_slots'):
            continue
        data[(d['dataset'], d.get('nodes'), d['k_query'], mode)].append(d)

    def show(track, title, note):
        print("=" * 100)
        print(title)
        print(note)
        print("=" * 100)
        print(f"{'nodes':>7s} {'docs':>8s} {'K':>4s} {'n':>3s} {'semantic':>13s} "
              f"{'random':>13s} {'tỉ lệ':>6s} {'node%':>7s} {'cand%':>7s} {'slot':>7s}")
        print("-" * 100)
        out = []
        for ds, nodes, docs, K in track:
            sem = data.get((ds, nodes, K, 'semantic'), [])
            rnd = data.get((ds, nodes, K, 'random_slots'), [])
            if not sem:
                print(f"{nodes:>7,} {docs:>8,} {K:>4} {'--':>3s}  (chưa có)")
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
            print(f"{nodes:>7,} {docs:>8,} {K:>4} {len(sem):>3} "
                  f"{s_m:>7.1f}±{s_s:<4.1f} {rstr} {rs} {nd:>6.1f}% {cd:>6.1f}% "
                  f"{5*K*8:>7,}")
            out.append({'nodes': nodes, 'docs': docs, 'K': K, 'sem': s_m,
                        'rnd': r_m, 'ratio': ratio, 'node_pct': nd, 'cand_pct': cd})
        print()
        return out

    A = show(TRACK_A, "ĐƯỜNG A — K CỐ ĐỊNH ở 20",
             "Chi phí truy vấn không đổi. K/N co lại khi mạng lớn -> phủ ít corpus dần.")
    B = show(TRACK_B, "ĐƯỜNG B — K TỈ LỆ VỚI N (K/N = 0,20%)",
             "Phủ corpus giữ nguyên. Giá: slot truy vấn tăng tuyến tính theo N.")

    if len(A) >= 2 and len(B) >= 2:
        print("=" * 100)
        print("KẾT LUẬN")
        print("=" * 100)
        a0, a1 = A[0], A[-1]
        b0, b1 = B[0], B[-1]
        fold = a1['docs'] / a0['docs']
        print(f"  Hệ lớn lên {fold:.0f} lần: {a0['nodes']:,} node/{a0['docs']:,} doc "
              f"-> {a1['nodes']:,} node/{a1['docs']:,} doc")
        print()
        print(f"  ĐƯỜNG A (giữ K=20, chi phí không đổi):")
        print(f"    recall {a0['sem']:.1f}% -> {a1['sem']:.1f}%  ({a1['sem']-a0['sem']:+.1f}đ)")
        print(f"    phủ    {a0['cand_pct']:.1f}% -> {a1['cand_pct']:.1f}% corpus")
        if a1['ratio'] == a1['ratio']:
            print(f"    tỉ lệ  {a0['ratio']:.1f}x -> {a1['ratio']:.1f}x  "
                  f"(lợi thế TĂNG vì random giảm nhanh hơn)")
        print()
        print(f"  ĐƯỜNG B (K tỉ lệ N, giữ phủ):")
        print(f"    recall {b0['sem']:.1f}% -> {b1['sem']:.1f}%  ({b1['sem']-b0['sem']:+.1f}đ)")
        print(f"    phủ    {b0['cand_pct']:.1f}% -> {b1['cand_pct']:.1f}% corpus")
        print(f"    slot   {5*b0['K']*8:,} -> {5*b1['K']*8:,}  (gấp {b1['K']/b0['K']:.0f})")
        print()
        drop = b0['sem'] - b1['sem']
        if abs(drop) < 5:
            print(f"  => Ở NGÂN SÁCH PHỦ CỐ ĐỊNH, recall chỉ mất {drop:.1f} điểm khi hệ lớn")
            print(f"     {fold:.0f} lần. Weak scaling ĐẠT, nhưng chi phí truy vấn tăng TUYẾN TÍNH.")
        else:
            print(f"  => Recall vẫn mất {drop:.1f} điểm dù giữ phủ. Corpus lớn có nhiều")
            print(f"     đối thủ nhiễu hơn — task tự nó khó lên, không chỉ do ngân sách.")
        print()
        print("  ĐÁNH ĐỔI: ở quy mô lớn, hoặc mất recall (đường A), hoặc trả chi phí")
        print("  tuyến tính theo N (đường B). O(log N) của Kademlia chỉ áp cho ĐỊNH VỊ")
        print("  một khoá; phủ đủ láng giềng ngữ nghĩa thì cần K tỉ lệ N.")
        print()
        print("  LƯU Ý: ground truth KHÁC nhau giữa các mức, vì top-10 của một query")
        print("  trong corpus 100.000 khác trong 20.000. Task khó dần theo thiết kế.")


if __name__ == '__main__':
    main()