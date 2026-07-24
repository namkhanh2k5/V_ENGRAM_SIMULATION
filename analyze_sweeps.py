#!/usr/bin/env python3
"""
Phân tích mục 17 (Zipf) và mục 21 (sweep T, K).

    python3 analyze_sweeps.py
"""
import glob
import json
import statistics as st
from collections import defaultdict


def load():
    rows = []
    for f in glob.glob('result_*_N10000_L5_*_MA1_T*_m512*_nq500.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('n_query', 0) < 500 or d.get('node_loss', 0) > 0:
            continue
        mode = d.get('routing_mode') or ('random_slots' if d.get('random_routing') else 'semantic')
        if mode != 'semantic':
            continue
        rows.append(d)
    return rows


def agg(rows, keyfn):
    g = defaultdict(list)
    for d in rows:
        g[keyfn(d)].append(d)
    return g


CORPUS = {'code': 20000, 'scifact': 5183, 'squad': 18891}


def m(vals):
    return st.mean(vals) if vals else float('nan')


def sd(vals):
    return st.stdev(vals) if len(vals) > 1 else 0.0


def main():
    rows = load()

    # ===== MỤC 17: Zipf =====
    print("=" * 96)
    print("MỤC 17 — WORKLOAD LỆCH (Zipf): tải TRUY VẤN có giống tải LƯU TRỮ không?")
    print("=" * 96)
    print(f"{'ds':8s} {'zipf':>5s} {'n':>2s} {'R@5':>7s} | {'RPC mean':>9s} {'RPC P99':>8s} "
          f"{'RPC max':>8s} {'RPC Gini':>9s} | {'store Gini':>10s} {'chênh':>7s}")
    print("-" * 96)
    g = agg([d for d in rows if d['k_query'] == 20 and d.get('multi_probe') == 8],
            lambda d: (d['dataset'], d.get('zipf', 0.0)))
    for k in sorted(g):
        v = g[k]
        rg = m([x.get('rpc_gini', 0) for x in v])
        sg = m([x['metadata_gini'] for x in v])
        print(f"{k[0]:8s} {k[1]:>5.1f} {len(v):>2} {m([x['recall5'] for x in v]):>6.1f}% | "
              f"{m([x.get('rpc_mean',0) for x in v]):>9.1f} {m([x.get('rpc_p99',0) for x in v]):>8.0f} "
              f"{m([x.get('rpc_max',0) for x in v]):>8.0f} {rg:>9.3f} | {sg:>10.3f} {sg-rg:>+7.3f}")
    print()
    print("  Cột 'chênh' = Gini lưu trữ trừ Gini RPC. DƯƠNG nghĩa là tải truy vấn")
    print("  NHẸ hơn tải lưu trữ: node giữ nhiều metadata không đồng nghĩa nhận nhiều query.")

    # ===== MỤC 21a: sweep T =====
    print()
    print("=" * 96)
    print("MỤC 21a — SWEEP T (K=20): thăm dò thêm prefix có đáng không?")
    print("=" * 96)
    print(f"{'T':>3s} {'n':>2s} {'Reach R@5':>11s} {'Final R@5':>11s} {'node%':>7s} "
          f"{'cand%':>7s} {'R@5/node%':>10s}")
    print("-" * 96)
    g = agg([d for d in rows if d['dataset'] == 'code' and d['k_query'] == 20
             and d.get('zipf', 0) == 0], lambda d: d.get('multi_probe', 1))
    prev = None
    for T in sorted(g):
        v = g[T]
        r5 = m([x['recall5'] for x in v])
        nd = m([x['pct_network_touched'] for x in v])
        eff = r5 / nd if nd > 0 else float('nan')
        delta = f"  (+{r5-prev:.1f}đ)" if prev is not None else ""
        print(f"{T:>3} {len(v):>2} {m([x['reachable_recall5'] for x in v]):>10.1f}% "
              f"{r5:>10.1f}% {nd:>6.1f}% {m([100*x['mean_unique_candidates']/CORPUS.get(x['dataset'],20000) for x in v]):>6.1f}% "
              f"{eff:>10.1f}{delta}")
        prev = r5

    # ===== MỤC 21b: sweep K =====
    print()
    print("=" * 96)
    print("MỤC 21b — SWEEP K (T=8): mở rộng biên có đáng không?")
    print("=" * 96)
    print(f"{'K':>3s} {'n':>2s} {'Reach R@5':>11s} {'Final R@5':>11s} {'node%':>7s} "
          f"{'cand%':>7s} {'R@5/node%':>10s}")
    print("-" * 96)
    g = agg([d for d in rows if d['dataset'] == 'code' and d.get('multi_probe') == 8
             and d.get('zipf', 0) == 0], lambda d: d['k_query'])
    prev = None
    for K in sorted(g):
        v = g[K]
        r5 = m([x['recall5'] for x in v])
        nd = m([x['pct_network_touched'] for x in v])
        eff = r5 / nd if nd > 0 else float('nan')
        delta = f"  (+{r5-prev:.1f}đ)" if prev is not None else ""
        print(f"{K:>3} {len(v):>2} {m([x['reachable_recall5'] for x in v]):>10.1f}% "
              f"{r5:>10.1f}% {nd:>6.1f}% {m([100*x['mean_unique_candidates']/CORPUS.get(x['dataset'],20000) for x in v]):>6.1f}% "
              f"{eff:>10.1f}{delta}")
        prev = r5

    print()
    print("  Cột R@5/node% là hiệu suất: recall thu được trên mỗi phần trăm mạng chạm.")
    print("  So T và K ở cùng cột này cho biết nên tiêu ngân sách vào đâu.")
    print()
    print("CHƯA ĐO ĐƯỢC Ở ĐÂY: R_max và % query chạm R_max.")
    print("  main_simulation_v2.py dùng lookup lý tưởng hoá (knn toàn cục), không có")
    print("  vòng lặp nào để cắt. Phải chạy main_simulation.py (SimPy) cho phần đó.")


if __name__ == '__main__':
    main()