"""V-Engram — mô phỏng đầy đủ (SimPy): discovery + payload + latency.

Chạy:
  python main_simulation.py --dataset code --nq 100
  python main_simulation.py --dataset code --nq 100 --random-routing   # baseline
  python main_simulation.py --dataset scifact --k-query 20 --multi-probe 3

Sửa so với bản cũ:
  1. Đường dẫn data mới (file cũ embeddings_20k.npy/faiss_absolute_baseline.json đã bị xoá)
  2. Dùng query_embeddings.npy PRECOMPUTED thay vì model.encode()
     -> bản cũ KHÔNG normalize query, trong khi corpus embeddings và PQ codebook
        đều train trên vector norm=1 => khoảng cách ADC bị lệch. Đây là lỗi ngầm.
     -> đồng thời khỏi nạp model bge 1.3GB mỗi lần chạy.
  3. Tham số hoá r / K / T qua dòng lệnh để sweep
  4. Báo cáo Recall@5/10 thay vì chỉ Success@5 (bão hoà ở 100%)
"""
import argparse
import json

import numpy as np
import simpy

from src.network import (
    bootstrap_network,
    data_ingestion_process,
    query_pipeline_process,
    reset_global_metadata_dht,
)
from src.evaluation import stage5_metrics_collection, export_comparison_report, test_data_integrity
import src.network as netmod
from src.routing import initialize_lsh_projections

DATASETS = {
    "code":    {"emb": "./data/code_corpus_embeddings.npy",
                "q":   "./data/code_query_embeddings.npy",
                "pq":  "./data/code_pq_codes.npy",
                "cb":  "./data/code_pq_codebook.npy",
                "gt":  "./data/code_ground_truth.json",
                "n":   20000},
    "scifact": {"emb": "./data/scifact_corpus_embeddings.npy",
                "q":   "./data/scifact_query_embeddings.npy",
                "pq":  "./data/scifact_pq_codes.npy",
                "cb":  "./data/scifact_pq_codebook.npy",
                "gt":  "./data/scifact_ground_truth.json",
                "n":   5183},
    "squad":   {"emb": "./data/squad_corpus_embeddings.npy",
                "q":   "./data/squad_query_embeddings.npy",
                "pq":  "./data/squad_pq_codes.npy",
                "cb":  "./data/squad_pq_codebook.npy",
                "gt":  "./data/squad_ground_truth.json",
                "n":   18891},
}

SHARDS_PER_FILE = 30
K_SIZE = 20


def run_simulation(env, args, cfg):
    codebook = np.load(cfg["cb"])
    Q = np.load(cfg["q"])                      # (500, 1024) ĐÃ normalize
    with open(cfg["gt"], "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"[*] {args.dataset}: corpus={cfg['n']:,} | codebook={codebook.shape} | "
          f"query={Q.shape}")
    print(f"[*] Cấu hình: r={netmod.METADATA_ANCHORS} K={args.k_query} "
          f"T={args.multi_probe} L=5 seed={args.seed}"
          f"{' [RANDOM ROUTING]' if args.random_routing else ''}")

    network_nodes = yield env.process(bootstrap_network(env, args.nodes, K_SIZE))

    yield env.process(data_ingestion_process(
        env, network_nodes, args.num_files, SHARDS_PER_FILE,
        embeddings_path=cfg["emb"], pq_codes_path=cfg["pq"],
        data_label=args.dataset.upper()))

    # Metadata load — số cho mục 4.x Metadata Load Distribution
    meta = np.array([len(n.RAM_Index) for n in network_nodes])
    x = np.sort(meta.astype(float))
    gini = float((2 * np.arange(1, len(x) + 1) - len(x) - 1) @ x / (len(x) * x.sum())) \
        if x.sum() > 0 else 0.0
    print(f"\n[METADATA LOAD] mean={meta.mean():.1f} P95={np.percentile(meta,95):.0f} "
          f"P99={np.percentile(meta,99):.0f} max={meta.max()} Gini={gini:.3f} "
          f"| tổng={meta.sum():,} bản")

    for doc_id in [12, 99, 256, 512, 1024, 2048, 4096, 8192, 12345, 19999]:
        if doc_id < args.num_files:
            test_data_integrity(network_nodes, test_doc_id=doc_id)

    n_run = min(args.nq, len(ground_truth))
    test_results, uniq_cands, all_stats = [], [], []
    for item in ground_truth[:n_run]:
        q_id = item["query_id"]
        query_vector = Q[q_id - 1]             # precomputed, đã normalize

        retrieved_tags, hops, n_uniq, stats = yield env.process(
            query_pipeline_process(env, network_nodes, query_vector, codebook,
                                   target_k=args.target_k,
                                   k_query=args.k_query,
                                   multi_probe=args.multi_probe,
                                   random_routing=args.random_routing,
                                   verbose=args.verbose))
        uniq_cands.append(n_uniq)
        all_stats.append(stats)
        test_results.append({"query_id": q_id, "retrieved": retrieved_tags, "hops": hops})
        if not args.verbose and len(test_results) % 10 == 0:
            print(f"    {len(test_results)}/{n_run} query xong")

    m = stage5_metrics_collection(env, network_nodes, test_results,
                                  ground_truth_path=cfg["gt"])

    print("\n" + "=" * 64)
    print(f"KẾT QUẢ | {args.dataset} | r={netmod.METADATA_ANCHORS} K={args.k_query} "
          f"T={args.multi_probe}{' [RANDOM]' if args.random_routing else ''}")
    print("=" * 64)
    if m:
        print(f"  Hit@5      : {m.get('hit_at_5', 0):.1f}%   (bão hoà — không dùng để so cấu hình)")
        print(f"  Recall@5   : {m.get('recall_at_5', 0):.1f}%   <- metric chính")
        print(f"  Recall@10  : {m.get('recall_at_10', 0):.1f}%")
        print(f"  GT thu hồi : {m.get('mean_gt_recovered', 0):.2f}/5 mỗi query")
        print(f"  MRR@5      : {m.get('mrr_score', 0):.4f}")
    print(f"  Rounds/query   : {np.mean([s['rounds'] for s in all_stats]):.1f}")
    print(f"  RPC/query      : {np.mean([s['rpcs'] for s in all_stats]):.1f}")

    # ===== MỤC 5: BẢNG CHI PHÍ ĐẦY ĐỦ =====
    def _q(key, p):
        return np.percentile([s.get(key, 0) for s in all_stats], p)

    def _m(key):
        return np.mean([s.get(key, 0) for s in all_stats])

    print()
    print("=" * 78)
    print("BẢNG CHI PHÍ (mục 5) — tách Discovery / Payload / Total")
    print("=" * 78)
    print(f"{'Đại lượng':28s} {'Discovery':>13s} {'Payload':>13s} {'Total':>13s}")
    print("-" * 78)
    print(f"{'Routing rounds/query':28s} {_m('disc_rounds'):>13.1f} "
          f"{_m('pay_rounds'):>13.1f} {_m('disc_rounds')+_m('pay_rounds'):>13.1f}")
    print(f"{'RPC/query':28s} {_m('disc_rpcs'):>13.1f} "
          f"{_m('pay_rpcs'):>13.1f} {_m('disc_rpcs')+_m('pay_rpcs'):>13.1f}")
    print(f"{'Bytes/query':28s} {_m('disc_bytes'):>13,.0f} "
          f"{_m('pay_bytes'):>13,.0f} {_m('disc_bytes')+_m('pay_bytes'):>13,.0f}")
    print(f"{'Unique nodes contacted':28s} {_m('contacted_nodes'):>13.1f} "
          f"{'--':>13s} {_m('contacted_nodes'):>13.1f}")
    print(f"{'Candidate tags moved':28s} {_m('candidate_tags'):>13.1f} "
          f"{'--':>13s} {_m('candidate_tags'):>13.1f}")
    print(f"{'Unique candidates':28s} {_m('unique_candidates'):>13.1f} "
          f"{'--':>13s} {_m('unique_candidates'):>13.1f}")
    print("-" * 78)
    print(f"{'Latency p50 (ms)':28s} {'':>13s} {'':>13s} {_q('latency_ms',50):>13.0f}")
    print(f"{'Latency p95 (ms)':28s} {'':>13s} {'':>13s} {_q('latency_ms',95):>13.0f}")
    print(f"{'Latency p99 (ms)':28s} {'':>13s} {'':>13s} {_q('latency_ms',99):>13.0f}")
    print("=" * 78)
    print("  Lưu ý: 'rounds' cộng dồn qua L×T lookup CHẠY SONG SONG, nên nó là tổng")
    print("  công, KHÔNG phải đường tới hạn. Latency mới phản ánh thời gian thực tế.")

    # ===== MỤC 21: % LOOKUP CHẠM R_max =====
    tot = sum(s.get('lookups_total', 0) for s in all_stats)
    cap = sum(s.get('lookups_at_cap', 0) for s in all_stats)
    rmax = all_stats[0].get('r_max', '?') if all_stats else '?'
    q_any_cap = sum(1 for s in all_stats if s.get('lookups_at_cap', 0) > 0)
    print()
    print(f"[MỤC 21] R_max = {rmax}")
    print(f"  Lookup chạm trần : {cap:,}/{tot:,} ({100.0*cap/max(1,tot):.1f}%)")
    print(f"  Query có >=1 lookup chạm trần: {q_any_cap}/{len(all_stats)} "
          f"({100.0*q_any_cap/max(1,len(all_stats)):.1f}%)")
    print(f"  => Nếu tỉ lệ cao, R_max đang CẮT lookup trước khi hội tụ; recall báo")
    print(f"     cáo là dưới trần đó, không phải của thuật toán hội tụ đầy đủ.")
    print(f"  Node chạm/query: {np.mean([s['contacted_nodes'] for s in all_stats]):.1f} "
          f"({100*np.mean([s['contacted_nodes'] for s in all_stats])/args.nodes:.1f}% mạng)")
    print(f"  Candidate/query: {np.mean(uniq_cands):.0f} "
          f"({100*np.mean(uniq_cands)/args.num_files:.2f}% corpus)")
    print(f"  Metadata Gini  : {gini:.3f}")

    out = {"dataset": args.dataset, "seed": args.seed, "nodes": args.nodes,
           "r": netmod.METADATA_ANCHORS, "k_query": args.k_query,
           "multi_probe": args.multi_probe, "random_routing": args.random_routing,
           "n_query": n_run,
           "hit_at_5": m.get("hit_at_5") if m else None,
           "recall_at_5": m.get("recall_at_5") if m else None,
           "recall_at_10": m.get("recall_at_10") if m else None,
           "mrr": m.get("mrr_score") if m else None,
           "mean_rounds": float(np.mean([s["rounds"] for s in all_stats])),
           "mean_rpcs": float(np.mean([s["rpcs"] for s in all_stats])),
           "disc_rounds": float(np.mean([s.get("disc_rounds",0) for s in all_stats])),
           "disc_rpcs": float(np.mean([s.get("disc_rpcs",0) for s in all_stats])),
           "disc_bytes": float(np.mean([s.get("disc_bytes",0) for s in all_stats])),
           "pay_rounds": float(np.mean([s.get("pay_rounds",0) for s in all_stats])),
           "pay_rpcs": float(np.mean([s.get("pay_rpcs",0) for s in all_stats])),
           "pay_bytes": float(np.mean([s.get("pay_bytes",0) for s in all_stats])),
           "candidate_tags": float(np.mean([s.get("candidate_tags",0) for s in all_stats])),
           "latency_p50": float(np.percentile([s.get("latency_ms",0) for s in all_stats], 50)),
           "latency_p95": float(np.percentile([s.get("latency_ms",0) for s in all_stats], 95)),
           "latency_p99": float(np.percentile([s.get("latency_ms",0) for s in all_stats], 99)),
           "r_max": all_stats[0].get("r_max", 15) if all_stats else 15,
           "lookups_total": int(sum(s.get("lookups_total",0) for s in all_stats)),
           "lookups_at_cap": int(sum(s.get("lookups_at_cap",0) for s in all_stats)),
           "mean_contacted": float(np.mean([s["contacted_nodes"] for s in all_stats])),
           "mean_candidates": float(np.mean(uniq_cands)),
           "metadata_gini": gini, "metadata_total": int(meta.sum())}
    fn = args.out or (f"result_full_{args.dataset}_r{netmod.METADATA_ANCHORS}"
                      f"_K{args.k_query}_T{args.multi_probe}"
                      f"{'_RANDOM' if args.random_routing else ''}_s{args.seed}.json")
    json.dump(out, open(fn, "w"), indent=2)
    print(f"\n→ Lưu: {fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="code", choices=list(DATASETS))
    ap.add_argument("--nodes", type=int, default=10000)
    ap.add_argument("--num-files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20235956)
    ap.add_argument("--k-query", type=int, default=20, help="K — node mỗi bảng")
    ap.add_argument("--multi-probe", type=int, default=3, help="T — prefix mỗi bảng")
    ap.add_argument("--meta-anchors", type=int, default=None, help="r — ghi đè METADATA_ANCHORS")
    ap.add_argument("--target-k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--random-routing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    if args.num_files is None:
        args.num_files = cfg["n"]
    if args.meta_anchors is not None:
        netmod.METADATA_ANCHORS = args.meta_anchors

    import random as _r
    _r.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    initialize_lsh_projections(args.seed)
    reset_global_metadata_dht()

    env = simpy.Environment()
    env.process(run_simulation(env, args, cfg))
    env.run()