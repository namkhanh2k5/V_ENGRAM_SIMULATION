"""Gộp result_*.json -> bảng mean±std, xuất summary.csv (dán thẳng vào .tex)."""
import glob, json, csv
from collections import defaultdict
import statistics as st

rows = defaultdict(list)
for f in glob.glob("result_*.json"):
    r = json.load(open(f))
    key = (r["dataset"], r["k_query"], r["meta_anchors"], r.get("multi_probe", 1),
           r["use_pq"], r.get("random_routing", False))
    rows[key].append(r)

def ms(vals):
    if not vals: return "-"
    if len(vals) == 1: return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f}±{st.stdev(vals):.1f}"

out = []
for key in sorted(rows):
    ds, K, r, T, pq, rand = key
    g = rows[key]
    out.append({
        "dataset": ds, "K": K, "r": r, "T": T,
        "PQ": "on" if pq else "off", "routing": "random" if rand else "semantic",
        "seeds": len(g),
        "recall@5": ms([x["recall5"] for x in g]),
        "recall@10": ms([x["recall10"] for x in g]),
        "hit@5": ms([x["final_hit5"] for x in g]),
        "reachable_r@5": ms([x.get("reachable_recall5", 0) for x in g]),
        "cand_pct": ms([100*x["mean_unique_candidates"]/(20000 if ds=="code" else 5183) for x in g]),
        "node_pct": ms([x["pct_network_touched"] for x in g]),
        "gini": ms([x["metadata_gini"] for x in g]),
    })

if out:
    with open("summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

hdr = f"{'ds':8s} {'K':>4s} {'r':>4s} {'T':>2s} {'PQ':>3s} {'routing':>8s} {'n':>2s} " \
      f"{'R@5':>12s} {'reach':>12s} {'cand%':>10s} {'node%':>10s}"
print("\n" + hdr); print("-"*len(hdr))
for o in out:
    print(f"{o['dataset']:8s} {o['K']:>4} {o['r']:>4} {o['T']:>2} {o['PQ']:>3s} "
          f"{o['routing']:>8s} {o['seeds']:>2} {o['recall@5']:>12s} "
          f"{o['reachable_r@5']:>12s} {o['cand_pct']:>10s} {o['node_pct']:>10s}")
print(f"\n-> summary.csv ({len(out)} cấu hình)")

# Bảng r* — contribution chính
print("\n=== NGƯỠNG r*: semantic vs random (code, K=20, T=3) ===")
print(f"{'r':>4s} {'semantic':>12s} {'random':>12s} {'gấp':>7s}")
for r in [1, 5, 10, 30, 150]:
    k_sem = ("code", 20, r, 3, True, False); k_rnd = ("code", 20, r, 3, True, True)
    if k_sem in rows and k_rnd in rows:
        s_ = st.mean([x["recall5"] for x in rows[k_sem]])
        n_ = st.mean([x["recall5"] for x in rows[k_rnd]])
        ratio = f"{s_/n_:.1f}x" if n_ > 0 else "inf"
        flag = "  <- RANDOM THẮNG" if n_ > s_ else ""
        print(f"{r:>4} {s_:>11.1f}% {n_:>11.1f}% {ratio:>7s}{flag}")