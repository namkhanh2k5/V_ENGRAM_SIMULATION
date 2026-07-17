"""Gộp result_*.json -> bảng mean±std, xuất summary.csv (dán thẳng vào .tex)."""
import glob, json, csv
from collections import defaultdict
import statistics as st

rows = defaultdict(list)
for f in glob.glob("result_*.json"):
    r = json.load(open(f))
    # File chạy trước bản vá không có trường pq_variant -> suy từ TÊN FILE
    # (tên có dạng result_{ds}_K{K}_MA{r}_{pq|m512|nopq}_L{L}_T{T}[_RANDOM].json)
    if "pq_variant" not in r:
        r["pq_variant"] = "m512" if "_m512" in f else "m256"
    key = (r["dataset"], r.get("num_tables", 5), r["k_query"], r["meta_anchors"],
           r.get("multi_probe", 1), r["use_pq"], r.get("random_routing", False),
           r.get("pq_variant", "m256"))
    rows[key].append(r)

def ms(vals):
    if not vals: return "-"
    if len(vals) == 1: return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f}±{st.stdev(vals):.1f}"

out = []
for key in sorted(rows):
    ds, L, K, r, T, pq, rand, pqv = key
    g = rows[key]
    out.append({
        "dataset": ds, "L": L, "K": K, "r": r, "T": T, "pq_var": pqv,
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

hdr = f"{'ds':8s} {'L':>3s} {'K':>4s} {'r':>3s} {'T':>2s} {'PQ':>3s} {'var':>5s} {'routing':>8s} {'n':>2s} " \
      f"{'R@5':>12s} {'reach':>12s} {'cand%':>9s} {'node%':>9s}"
print("\n" + hdr); print("-"*len(hdr))
for o in out:
    print(f"{o['dataset']:8s} {o['L']:>3} {o['K']:>4} {o['r']:>3} {o['T']:>2} {o['PQ']:>3s} "
          f"{o['pq_var']:>5s} {o['routing']:>8s} {o['seeds']:>2} {o['recall@5']:>12s} "
          f"{o['reachable_r@5']:>12s} {o['cand_pct']:>9s} {o['node_pct']:>9s}")
print(f"\n-> summary.csv ({len(out)} cấu hình)")

# ===== BẢNG CHỌN CẤU HÌNH: recall cao MÀ vẫn thắng random đậm =====
print("\n" + "="*88)
print("CHỌN CẤU HÌNH — sắp theo Recall@5. Chỉ nhận cấu hình có TỈ LỆ sem/rand lớn.")
print("  recall cao mà random cũng cao => VÔ NGHĨA (vd r=150: sem 70.0% vs rand 87.6%)")
print("="*88)
print(f"{'ds':8s} {'L':>3s} {'K':>4s} {'r':>3s} {'T':>2s} {'PQ':>5s} {'semantic':>10s} {'random':>10s} "
      f"{'tỉ lệ':>8s} {'node%':>7s}  đánh giá")
print("-"*88)

cand_rows = []
for key in rows:
    ds, L, K, r, T, pq, rand, pqv = key
    if rand or not pq:
        continue
    k_rnd = (ds, L, K, r, T, pq, True, pqv)
    if k_rnd not in rows:
        continue
    sem = st.mean([x["recall5"] for x in rows[key]])
    rnd = st.mean([x["recall5"] for x in rows[k_rnd]])
    node = st.mean([x["pct_network_touched"] for x in rows[key]])
    ratio = sem / rnd if rnd > 0 else 999
    cand_rows.append((sem, rnd, ratio, node, ds, L, K, r, T, pqv))

for sem, rnd, ratio, node, ds, L, K, r, T, pqv in sorted(cand_rows, reverse=True):
    if rnd > sem:
        verdict = "BỎ — random thắng"
    elif ratio < 1.5:
        verdict = "yếu — sem/rand < 1.5x"
    elif sem >= 80:
        verdict = "*** ĐẠT >80% và thắng đậm ***"
    elif sem >= 70:
        verdict = "tốt"
    else:
        verdict = ""
    rs = f"{ratio:.1f}x" if ratio < 900 else "inf"
    print(f"{ds:8s} {L:>3} {K:>4} {r:>3} {T:>2} {pqv:>5s} {sem:>9.1f}% {rnd:>9.1f}% "
          f"{rs:>8s} {node:>6.1f}%  {verdict}")

# Trần no-PQ: cho biết PQ đang ăn mất bao nhiêu
print("\n=== TRẦN no-PQ (PQ đang ăn mất bao nhiêu điểm) ===")
print(f"{'ds':8s} {'L':>3s} {'T':>2s} {'var':>5s} {'PQ on':>9s} {'PQ off':>9s} {'mất':>7s}")
for key in sorted(rows):
    ds, L, K, r, T, pq, rand, pqv = key
    if not pq or rand:
        continue
    k_off = (ds, L, K, r, T, False, False, pqv)
    if k_off in rows:
        on_ = st.mean([x["recall5"] for x in rows[key]])
        off = st.mean([x["recall5"] for x in rows[k_off]])
        print(f"{ds:8s} {L:>3} {T:>2} {pqv:>5s} {on_:>8.1f}% {off:>8.1f}% {off-on_:>6.1f}đ")