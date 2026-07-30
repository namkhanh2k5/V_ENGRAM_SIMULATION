"""Đo pool size THẬT của Crypto-DHT so với LSH ceiling, cùng POOL_PER_TABLE.
Câu hỏi: 3.505 trong Bảng 6 có phải pool đo được của Crypto-DHT, hay là số
copy từ hàng LSH? Và Recall 22,5% nằm trên hay dưới null model?"""
import hashlib, json, sys
import numpy as np
sys.path.insert(0, '.')
from src.routing import initialize_lsh_projections, PROJECTION_MATRICES
import src.routing as routing

L, N_CORPUS = 5, 20000
E  = np.load('data/code_corpus_embeddings.npy').astype(np.float32)
Qv = np.load('data/code_query_embeddings.npy').astype(np.float32)
gt = json.load(open('data/code_ground_truth.json', encoding='utf-8'))
gt_sets = [set(r['index'] for r in g['top_5_results']) for g in gt]
Q = len(gt_sets)

initialize_lsh_projections(20235956)
projs = routing.PROJECTION_MATRICES
doc_bits = [(E @ projs[t]) > 0 for t in range(L)]
q_bits   = [(Qv @ projs[t]) > 0 for t in range(L)]

def sha160(s): return int(hashlib.sha256(s.encode()).hexdigest(), 16) & ((1 << 160) - 1)
doc_ck = {t: np.array([sha160(f"doc_{i}_tbl_{t}") >> 96 for i in range(N_CORPUS)], dtype=np.uint64)
          for t in range(L)}

def measure(P):
    lsh_pool, cry_pool, lsh_rec, cry_rec = [], [], 0.0, 0.0
    for q in range(Q):
        c1 = set()
        for t in range(L):
            ham = (doc_bits[t] ^ q_bits[t][q]).sum(axis=1)
            c1.update(int(i) for i in np.argpartition(ham, P)[:P])
        c2 = set()
        for t in range(L):
            qk = np.uint64(sha160(f"query_{q}_tbl_{t}") >> 96)
            d = doc_ck[t] ^ qk
            c2.update(int(i) for i in np.argpartition(d, P)[:P])
        lsh_pool.append(len(c1)); cry_pool.append(len(c2))
        # rerank CHÍNH XÁC: recall = |top5 ∩ pool| / 5 (tách khỏi lỗi PQ)
        lsh_rec += len(gt_sets[q] & c1) / 5.0
        cry_rec += len(gt_sets[q] & c2) / 5.0
    return (np.mean(lsh_pool), 100*lsh_rec/Q, np.mean(cry_pool), 100*cry_rec/Q)

print(f"{'P/bảng':>7s} | {'LSH pool':>9s} {'LSH R@5':>8s} | {'CRYPTO pool':>12s} "
      f"{'CRYPTO R@5':>11s} {'null=pool/N':>12s} {'lệch':>6s}")
print("-"*78)
for P in (701, 902, 1000):
    lp, lr, cp, cr = measure(P)
    null = 100*cp/N_CORPUS
    print(f"{P:>7d} | {lp:>9.0f} {lr:>7.1f}% | {cp:>12.0f} {cr:>10.1f}% "
          f"{null:>11.1f}% {cr-null:>+6.1f}")
