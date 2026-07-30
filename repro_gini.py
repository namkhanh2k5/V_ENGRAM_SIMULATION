"""Tái tạo metadata_gini trực tiếp từ embeddings, không cần simulator.
metadata_gini phụ thuộc (seed, dataset, N, L, r) -- KHÔNG phụ thuộc zipf.
Mục tiêu: xem con số nào trong Bảng 15 / muc 4.4 / muc 4.13 là đúng."""
import json, random, sys
import numpy as np

N_NODES, L, R_ANCHORS = 10000, 5, 1

def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    if x.sum() == 0: return 0.0
    return float((2*np.arange(1, n+1) - n - 1) @ x / (n * x.sum()))

def projections(seed, dim=1024):
    rng = np.random.RandomState(seed)
    return [rng.choice([0,1,-1], size=(dim,160), p=[2/3,1/6,1/6]) for _ in range(L)]

def node_ids(seed, n=N_NODES):
    rnd = random.Random(seed)
    return np.array([rnd.getrandbits(63) for _ in range(n)], dtype=np.int64)

POW = (np.int64(1) << np.arange(62, -1, -1, dtype=np.int64))   # bit i -> 2^(62-i)

def keys63(E, proj):
    """vector hoá key63: bits = (v@proj > 0)[:63] -> int64"""
    bits = (E @ proj > 0)[:, :63].astype(np.int64)
    return bits @ POW

def nearest(keys, nids, chunk=512):
    out = np.empty(len(keys), dtype=np.int64)
    for i in range(0, len(keys), chunk):
        k = keys[i:i+chunk]
        d = np.bitwise_xor(nids[None, :], k[:, None])
        out[i:i+chunk] = np.argmin(d, axis=1)
    return out

def run(ds, seed):
    E = np.load(f'data/{ds}_corpus_embeddings.npy').astype(np.float32)
    P = projections(seed, E.shape[1]); nids = node_ids(seed)
    # ram[node] = set(doc) -> đếm doc PHÂN BIỆT, khớp dict trong code gốc
    counts = np.zeros(N_NODES, dtype=np.int64)
    seen = [set() for _ in range(N_NODES)]
    for proj in P:
        anchors = nearest(keys63(E, proj), nids)
        for doc, nidx in enumerate(anchors):
            s = seen[nidx]
            if doc not in s:
                s.add(doc); counts[nidx] += 1
    return counts

if __name__ == '__main__':
    ds = sys.argv[1]
    seeds = [20235956, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    res = {}
    for s in seeds:
        c = run(ds, s); res[s] = gini(c)
        print(f"  {ds} seed={s:<9} Gini={res[s]:.4f}  total={c.sum():,}  "
              f"mean={c.mean():.2f} P95={np.percentile(c,95):.0f} "
              f"P99={np.percentile(c,99):.0f} max={c.max()}", flush=True)
    json.dump({str(k): v for k, v in res.items()}, open(f'/home/claude/gini_{ds}.json','w'))
    z5  = [res[s] for s in [20235956,1,2,3,4]]
    z3  = [res[s] for s in [20235956,1,2]]
    all10 = [res[s] for s in seeds]
    print(f"\n  {ds}: mean 3 seed  (20235956,1,2)   = {np.mean(z3):.3f}")
    print(f"  {ds}: mean 5 seed  (20235956,1..4)  = {np.mean(z5):.3f}   <- hàng z>0 của Bảng 15")
    print(f"  {ds}: mean 10 seed (20235956,1..9)  = {np.mean(all10):.3f}   <- hàng z=0 của Bảng 15")
