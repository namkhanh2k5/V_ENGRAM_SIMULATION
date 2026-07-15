"""
V-Engram simulation v2 — chạy với data mới (code/scifact/squad).

Thay cho main_simulation.py cũ (đang crash vì file data cũ đã bị xoá).

Sửa so với bản cũ:
  1. Đường dẫn data mới, tham số hoá --dataset
  2. Dùng query_embeddings precomputed (đã normalize) thay vì model.encode()
     -> bản cũ KHÔNG normalize query trong khi codebook train trên vector norm=1 => ADC lệch
  3. Sửa bug ngân sách: ngân sách theo TỪNG bảng, không cộng dồn (bản cũ: bảng 1 ghé 14 node,
     bảng 2-5 mỗi bảng ghé đúng 1 node)
  4. Đo 3 tầng recall: reachable / returned / final -> tách discovery vs PQ filtering
  5. Chế độ --no-pq: rerank bằng cosine vector gốc (oracle upper bound)
  6. Tham số hoá K_QUERY, METADATA_ANCHORS để sweep

Cách chạy:
  python main_simulation_v2.py --dataset code --seed 20235956 --k-query 300 --meta-anchors 30 --use-pq --nq 100
  python main_simulation_v2.py --dataset code --seed 20235956 --k-query 20  --meta-anchors 1  --no-pq  --nq 100
"""
import argparse, json, random, time
import numpy as np

from src.routing import initialize_lsh_projections, generate_multi_semantic_keys


# ------------------------------------------------------------------ #
# Mô hình mạng tối giản (không SimPy): chỉ đo RECALL, không đo latency.
# Nhanh hơn nhiều -> sweep được nhiều cấu hình. Latency đo riêng sau.
# ------------------------------------------------------------------ #
class FastNetwork:
    def __init__(self, num_nodes, seed):
        rnd = random.Random(seed)
        # 63-bit thay vì 160-bit: đủ để phân biệt 10k node, cho phép vector hoá numpy
        self.node_ids = np.array([rnd.getrandbits(63) for _ in range(num_nodes)], dtype=np.int64)
        self.num_nodes = num_nodes
        self.ram = [dict() for _ in range(num_nodes)]   # node_idx -> {tag: pq_code}

    @staticmethod
    def key63(vector, proj):
        bits = (np.asarray(vector).flatten() @ proj > 0).astype(int)[:63]
        return np.int64(int(''.join(map(str, bits)), 2))

    def knn(self, key, k):
        d = np.bitwise_xor(self.node_ids, key)
        k = min(k, self.num_nodes)
        idx = np.argpartition(d, k - 1)[:k]
        return idx

    @staticmethod
    def probe_keys63(vector, proj, T, c=16):
        """Sinh T prefix: key gốc + (T-1) biến thể lật bit YẾU nhất.

        Bit 'yếu' = |v . r_i| nhỏ => vector nằm sát siêu phẳng chiếu => bit dễ lật
        ở neighbor đúng (boundary effect, mục 3.2 của paper). Lật đúng những bit
        đó = đi tới subtree mà neighbor nhiều khả năng rơi vào.
        Chỉ lật trong c bit đầu vì chỉ prefix mới quyết định vùng node.
        """
        pr = np.asarray(vector).flatten() @ proj
        bits = (pr > 0).astype(int)[:63]
        base = int(''.join(map(str, bits)), 2)
        keys = [np.int64(base)]
        if T <= 1:
            return keys
        c = min(c, 63)
        weak = np.argsort(np.abs(pr[:c]))          # bit yếu nhất trước
        for j in weak[:T - 1]:
            keys.append(np.int64(base ^ (1 << (62 - int(j)))))
        return keys

    def metadata_count(self):
        return np.array([len(r) for r in self.ram])


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='code', choices=['code', 'scifact', 'squad'])
    ap.add_argument('--nodes', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20235956)
    ap.add_argument('--k-query', type=int, default=300, help='số node mỗi bảng ghé (K_QUERY)')
    ap.add_argument('--meta-anchors', type=int, default=30, help='số node neo metadata (MA)')
    ap.add_argument('--local-topk', type=int, default=30, help='mỗi node trả top-k')
    ap.add_argument('--target-k', type=int, default=10, help='top-k cuối trả về client')
    ap.add_argument('--nq', type=int, default=100, help='số query chạy (500 = full)')
    ap.add_argument('--use-pq', dest='use_pq', action='store_true', default=True)
    ap.add_argument('--no-pq', dest='use_pq', action='store_false')
    ap.add_argument('--multi-probe', type=int, default=1, metavar='T',
                    help='Số prefix probe mỗi bảng (T=1 là single-probe như hiện tại). '
                         'Lật T-1 bit YẾU nhất (|proj| nhỏ = vector sát siêu phẳng). '
                         'Tăng node chạm nhưng KHÔNG tăng nhân bản metadata.')
    ap.add_argument('--probe-bits', type=int, default=16, metavar='C',
                    help='Chỉ xét lật bit trong C bit đầu (prefix hiệu dụng)')
    ap.add_argument('--random-routing', action='store_true',
                    help='BASELINE: chạm cùng số node nhưng chọn NGẪU NHIÊN '
                         '(không dùng semantic key). So với bản thường để biết '
                         'routing có đóng góp gì không.')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    D = f'./data/{args.dataset}'
    print(f"[*] Nạp data: {args.dataset} | PQ={'BẬT' if args.use_pq else 'TẮT (oracle cosine)'}")
    E  = np.load(f'{D}_corpus_embeddings.npy')      # (N,1024) đã normalize
    Qv = np.load(f'{D}_query_embeddings.npy')       # (500,1024) đã normalize
    codes = np.load(f'{D}_pq_codes.npy')            # (N,256) uint8
    codebook = np.load(f'{D}_pq_codebook.npy')      # (256,256,4)
    gt = json.load(open(f'{D}_ground_truth.json', encoding='utf-8'))
    N_DOCS = len(E)
    print(f"    corpus={N_DOCS:,} | query={len(gt)} | codebook={codebook.shape}")

    # LSH projections — dùng đúng hàm của repo để không lệch với code gốc
    initialize_lsh_projections(args.seed)
    from src.routing import PROJECTION_MATRICES as P
    L = len(P)
    print(f"[*] L={L} bảng chiếu, seed={args.seed}")

    net = FastNetwork(args.nodes, args.seed)

    # -------------------- INGESTION: neo metadata -------------------- #
    t0 = time.time()
    print(f"[*] Ingest {N_DOCS:,} doc (MA={args.meta_anchors})...")
    for i in range(N_DOCS):
        for proj in P:
            skey = net.key63(E[i], proj)
            for nidx in net.knn(skey, args.meta_anchors):
                net.ram[nidx][i] = codes[i]
        if (i + 1) % 5000 == 0:
            print(f"    ... {i+1:,}/{N_DOCS:,}")
    mc = net.metadata_count()
    print(f"✓ Ingest xong ({time.time()-t0:.0f}s). Metadata/node: mean={mc.mean():.1f} "
          f"P95={np.percentile(mc,95):.0f} P99={np.percentile(mc,99):.0f} max={mc.max()} "
          f"Gini={gini(mc):.3f} | tổng bản sao={mc.sum():,}")

    # -------------------- QUERY -------------------- #
    m, d_sub = 256, 4
    n_run = min(args.nq, len(gt))
    reach_hit = ret_hit = fin_hit = 0
    reach_r5 = ret_r5 = 0.0          # Recall@5 tầng 1, tầng 2
    rec5_sum = rec10_sum = 0.0
    uniq_list, touched_list = [], []

    rnd_route = random.Random(args.seed + 777)
    print(f"\n[*] Chạy {n_run} query... "
          f"{'[ROUTING NGẪU NHIÊN - baseline]' if args.random_routing else ''}")
    for qi in range(n_run):
        item = gt[qi]
        gt5  = [r['index'] for r in item['top_5_results']]
        gt10 = [r['index'] for r in item['top_10_results']]
        q = Qv[qi]

        # LUT cho ADC (chỉ cần khi dùng PQ)
        if args.use_pq:
            qsub = np.asarray(q, dtype=np.float32).reshape(m, d_sub)
            LUT = np.sum((qsub[:, np.newaxis, :] - codebook) ** 2, axis=2)   # (256,256)

        # --- Ripple Search: ngân sách THEO TỪNG BẢNG (sửa bug cộng dồn) ---
        touched = set()
        if args.random_routing:
            # BASELINE: chạm ĐÚNG cùng số node, nhưng chọn ngẫu nhiên
            n_touch = args.k_query * L * args.multi_probe
            touched = set(rnd_route.sample(range(args.nodes), min(n_touch, args.nodes)))
        else:
            for proj in P:
                for qkey in net.probe_keys63(q, proj, args.multi_probe, args.probe_bits):
                    touched.update(int(x) for x in net.knn(qkey, args.k_query))
        touched_list.append(len(touched))

        # Tầng 1: reachable — GT có nằm trong RAM của node được ghé không
        reachable_tags = set()
        for nidx in touched:
            reachable_tags.update(net.ram[nidx].keys())
        if set(gt5) & reachable_tags:
            reach_hit += 1
        reach_r5 += len(set(gt5) & reachable_tags) / 5.0

        # Tầng 2: returned — mỗi node trả local top-k
        cand = {}
        for nidx in touched:
            ram = net.ram[nidx]
            if not ram:
                continue
            tags = np.fromiter(ram.keys(), dtype=np.int64, count=len(ram))
            if args.use_pq:
                cm = np.stack([ram[int(t)] for t in tags])            # (n,256)
                dist = LUT[np.arange(m)[None, :], cm].sum(axis=1)     # ADC
            else:
                dist = -(E[tags] @ q)                                 # cosine (oracle)
            top = np.argsort(dist)[:args.local_topk]
            for j in top:
                t = int(tags[j]); dv = float(dist[j])
                if t not in cand or dv < cand[t]:
                    cand[t] = dv
        uniq_list.append(len(cand))
        if set(gt5) & set(cand.keys()):
            ret_hit += 1
        ret_r5 += len(set(gt5) & set(cand.keys())) / 5.0

        # Tầng 3: final — global rerank
        final = [t for t, _ in sorted(cand.items(), key=lambda x: x[1])[:args.target_k]]
        f5 = set(final[:5])
        if f5 & set(gt5):
            fin_hit += 1
        rec5_sum  += len(f5 & set(gt5)) / 5.0
        rec10_sum += len(set(final) & set(gt10)) / 10.0

        if (qi + 1) % 25 == 0:
            print(f"    {qi+1}/{n_run} | Hit@5={100*fin_hit/(qi+1):.1f}%")

    # -------------------- KẾT QUẢ -------------------- #
    res = {
        'dataset': args.dataset, 'nodes': args.nodes, 'seed': args.seed,
        'k_query': args.k_query, 'meta_anchors': args.meta_anchors,
        'random_routing': args.random_routing,
        'multi_probe': args.multi_probe,
        'local_topk': args.local_topk, 'use_pq': args.use_pq, 'n_query': n_run,
        'reachable_hit5': 100 * reach_hit / n_run,
        'returned_hit5':  100 * ret_hit / n_run,
        'final_hit5':     100 * fin_hit / n_run,
        'reachable_recall5': 100 * reach_r5 / n_run,
        'returned_recall5':  100 * ret_r5 / n_run,
        'recall5':  100 * rec5_sum / n_run,
        'recall10': 100 * rec10_sum / n_run,
        'mean_unique_candidates': float(np.mean(uniq_list)),
        'mean_nodes_touched': float(np.mean(touched_list)),
        'pct_network_touched': 100 * float(np.mean(touched_list)) / args.nodes,
        'metadata_total': int(mc.sum()),
        'metadata_mean_per_node': float(mc.mean()),
        'metadata_gini': gini(mc),
    }

    print("\n" + "=" * 62)
    print(f"KẾT QUẢ | {args.dataset} | K={args.k_query} MA={args.meta_anchors} "
          f"T={args.multi_probe} PQ={'ON' if args.use_pq else 'OFF'}"
          f"{' [RANDOM]' if args.random_routing else ''}")
    print("=" * 62)
    print(f"  {'':34s}  Hit@5    Recall@5")
    print(f"  Tầng 1 reachable (thuần discovery) : {res['reachable_hit5']:6.1f}%  {res['reachable_recall5']:6.1f}%")
    print(f"  Tầng 2 returned  (sau node lọc)    : {res['returned_hit5']:6.1f}%  {res['returned_recall5']:6.1f}%")
    print(f"  Tầng 3 final     (sau rerank)      : {res['final_hit5']:6.1f}%  {res['recall5']:6.1f}%")
    print(f"  --- MẤT Recall@5: do node lọc top-{args.local_topk} = "
          f"{res['reachable_recall5']-res['returned_recall5']:.1f}đ | "
          f"do global rerank = {res['returned_recall5']-res['recall5']:.1f}đ")
    print(f"  Recall@10={res['recall10']:.1f}%")
    print(f"  Candidate unique/query: {res['mean_unique_candidates']:.0f} "
          f"({100*res['mean_unique_candidates']/N_DOCS:.2f}% corpus)")
    print(f"  Node chạm/query: {res['mean_nodes_touched']:.0f} "
          f"({res['pct_network_touched']:.1f}% mạng)")
    print(f"  Metadata: {res['metadata_total']:,} bản ({res['metadata_mean_per_node']:.0f}/node, "
          f"Gini={res['metadata_gini']:.3f})")

    out = args.out or (f"result_{args.dataset}_K{args.k_query}_MA{args.meta_anchors}_"
                       f"{'pq' if args.use_pq else 'nopq'}"
                       f"_T{args.multi_probe}"
                       f"{'_RANDOM' if args.random_routing else ''}.json")
    json.dump(res, open(out, 'w'), indent=2)
    print(f"\n→ Lưu: {out}")


if __name__ == '__main__':
    main()