"""
baselines.py — Sinh Bảng 4 (Baseline Comparison) một cách TRUNG THỰC, TÁI LẬP ĐƯỢC.

Đặt file này ở THƯ MỤC GỐC của repo V_ENGRAM_SIMULATION (cùng cấp với src/, ./data/).
Chạy:  python baselines.py
Phụ thuộc: numpy, faiss-cpu, sentence-transformers  (đã có sẵn vì V-Engram dùng).

Mục tiêu: mọi baseline dùng ĐÚNG ground truth, ĐÚNG metric (Success@5/MRR@5 như
src/evaluation.py), và ĐÚNG ma trận SRP như V-Engram (import từ src/routing.py),
để con số so sánh được với V-Engram một cách công bằng.

Năm baseline:
  (0) Exact Brute-Force (IndexFlatIP)  -> kiểm tra harness = 100% so với GT.
  (1) FAISS-HNSW (M=32, efSearch=200)  -> chuẩn ANN tập trung.
  (2) Centralized Multi-Table LSH      -> TRẦN tín hiệu SRP (gom theo Hamming,
                                          global view, KHÔNG mất mát do routing XOR).
  (3) Crypto-DHT cùng ngân sách        -> SÀN: đúng pipeline (gom ~POOL candidate +
                                          rerank ADC) nhưng key = SHA-256 (mất ngữ nghĩa).
  (4) Random-5 (pure chance)           -> bốc đại 5 doc, KHÔNG rerank (đây là nguồn
                                          của con số ~0.1% trong bản nháp cũ).
"""

import os
import json
import hashlib
import numpy as np

# ======================= CẤU HÌNH =======================
CORPUS = "scifact"            # "code" hoặc "scifact"

PATHS = {
    "code": {
        "emb":      "./data/embeddings_20k.npy",
        "gt":       "./data/faiss_absolute_baseline.json",
        "codebook": "./data/pq_codebook.npy",
        "pq_codes": "./data/pq_codes.npy",        # nếu thiếu -> tự encode từ emb+codebook
    },
    "scifact": {
        "emb":      "./data/scifact_embeddings.npy",
        "gt":       "./data/scifact_faiss_absolute_baseline.json",
        "codebook": "./data/scifact_pq_codebook.npy",
        "pq_codes": "./data/scifact_pq_codes.npy",
    },
}[CORPUS]

MODEL_NAME       = "BAAI/bge-large-en-v1.5"
LSH_SEED         = 20235956   # PHẢI trùng DEFAULT_LSH_SEED trong src/routing.py
NUM_PROJECTIONS  = 5          # L
POOL_PER_TABLE   = 100        # ngân sách gom mỗi bảng (×L ~ 500 ứng viên, khớp V-Engram)
RERANK           = "adc"      # "adc" (giống V-Engram) hoặc "exact" (cosine chính xác)
INDEXED_COUNT    = None       # None = index toàn bộ corpus (khớp main_simulation NUM_FILES).
                              # Đặt 16000 nếu run chính thức dùng 80/20 (nhớ GT phải tính
                              # trên đúng tập index thì mới đạt được).
HNSW_M, HNSW_EF  = 32, 200
PQ_M, PQ_DSUB    = 256, 4     # 256 subquantizer × 4 chiều = 1024 (khớp node.adc_search)
REPORT_PATH      = "baseline_report.txt"
RNG_SEED         = 42         # cho random-5
# ========================================================

import faiss
from sentence_transformers import SentenceTransformer
# Import ĐÚNG ma trận chiếu của V-Engram để key hoàn toàn trùng khớp
from src.routing import initialize_lsh_projections
import src.routing as routing


def log(msg): print(msg, flush=True)


# ---------- METRIC: sao y src/evaluation.py ----------
def success_and_mrr(all_retrieved, gt_sets):
    """all_retrieved[q] = list index top-5 (theo thứ tự hạng). gt_sets[q] = set top-5 GT."""
    hit, mrr = 0, 0.0
    for retrieved, gt in zip(all_retrieved, gt_sets):
        if set(retrieved) & gt:
            hit += 1
        for rank, idx in enumerate(retrieved, 1):   # MRR theo hạng đầu tiên trúng
            if idx in gt:
                mrr += 1.0 / rank
                break
    n = len(all_retrieved)
    return 100.0 * hit / n, mrr / n


# ---------- PQ: encode + ADC (sao y node.adc_search) ----------
def pq_encode(E, codebook):
    """E (N,1024) -> codes (N,256) uint8, gán mỗi subvector về centroid gần nhất (L2)."""
    N = E.shape[0]
    sub = E.reshape(N, PQ_M, PQ_DSUB)
    codes = np.empty((N, PQ_M), dtype=np.uint8)
    for m in range(PQ_M):
        d = ((sub[:, m, :][:, None, :] - codebook[m][None, :, :]) ** 2).sum(axis=2)  # (N,256)
        codes[:, m] = np.argmin(d, axis=1)
    return codes

def adc_rerank(query_vec, cand_idx, pq_codes, codebook, top_k=5):
    """Trả về top_k index trong cand_idx theo khoảng cách ADC (L2 xấp xỉ) tới query."""
    q_sub = query_vec.reshape(PQ_M, PQ_DSUB)
    diff = q_sub[:, None, :] - codebook            # (256,256,4)
    LUT = (diff ** 2).sum(axis=2)                  # (256,256)
    codes = pq_codes[cand_idx]                     # (C,256)
    dists = LUT[np.arange(PQ_M), codes].sum(axis=1)  # (C,)
    order = np.argsort(dists)[:top_k]
    return [int(cand_idx[i]) for i in order]

def exact_rerank(query_vec, cand_idx, E, top_k=5):
    sims = E[cand_idx] @ query_vec                 # cosine (vector đã chuẩn hoá)
    order = np.argsort(-sims)[:top_k]
    return [int(cand_idx[i]) for i in order]

def rerank(query_vec, cand_idx, E, pq_codes, codebook):
    cand_idx = np.asarray(list(cand_idx), dtype=np.int64)
    if cand_idx.size == 0:
        return []
    if RERANK == "adc" and pq_codes is not None:
        return adc_rerank(query_vec, cand_idx, pq_codes, codebook)
    return exact_rerank(query_vec, cand_idx, E)


# ---------- SRP keys (trùng V-Engram) ----------
def srp_bits(E, proj):
    """proj (1024,160) -> bits (N,160) bool, đúng quy ước (E·M > 0) của routing.py."""
    return (E @ proj) > 0

def sha160(s):
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) & ((1 << 160) - 1)


def main():
    assert os.path.isdir("src"), "Hãy chạy ở thư mục gốc repo (cần thấy src/)."
    log(f"[*] CORPUS = {CORPUS} | RERANK = {RERANK} | L = {NUM_PROJECTIONS} | POOL/table = {POOL_PER_TABLE}")

    # 1) Nạp dữ liệu
    E = np.load(PATHS["emb"]).astype(np.float32)
    faiss.normalize_L2(E)                          # phòng hờ; emb gốc đã chuẩn hoá
    if INDEXED_COUNT:
        E = E[:INDEXED_COUNT]
    N, dim = E.shape
    log(f"[*] Corpus: {N:,} vectors × {dim} chiều")

    gt_raw = json.load(open(PATHS["gt"], "r", encoding="utf-8"))
    queries = [it["query_text"] for it in gt_raw]
    gt_sets = [set(r["index"] for r in it["top_5_results"]) for it in gt_raw]
    Q = len(queries)
    log(f"[*] Queries: {Q}")

    codebook = np.load(PATHS["codebook"]).astype(np.float32)
    if codebook.shape != (PQ_M, 256, PQ_DSUB):
        log(f"[!] Cảnh báo: codebook shape {codebook.shape} ≠ (256,256,4). Kiểm tra lại.")
    if os.path.exists(PATHS["pq_codes"]):
        pq_codes = np.load(PATHS["pq_codes"])
        if pq_codes.shape[0] != N:
            log("[!] pq_codes lệch số dòng so với corpus -> encode lại.")
            pq_codes = pq_encode(E, codebook)
    else:
        log("[*] Không thấy pq_codes -> tự encode từ emb + codebook ...")
        pq_codes = pq_encode(E, codebook)
    log(f"[*] PQ codes: {pq_codes.shape}")

    # 2) Encode query bằng đúng model của V-Engram
    log("[*] Nạp model & encode query ...")
    model = SentenceTransformer(MODEL_NAME)
    Qv = np.asarray(model.encode(queries, normalize_embeddings=True, show_progress_bar=False),
                    dtype=np.float32)

    # 3) Sanity: brute-force exact PHẢI = 100% so với GT (nếu không -> lệch model/normalize)
    flat = faiss.IndexFlatIP(dim); flat.add(E)
    _, I = flat.search(Qv, 5)
    bf_ret = [[int(i) for i in row] for row in I]
    bf_s, bf_m = success_and_mrr(bf_ret, gt_sets)
    log(f"[SANITY] Exact Brute-Force vs GT: Success@5={bf_s:.1f}%  MRR@5={bf_m:.3f}  (mong đợi 100% / 1.000)")
    if bf_s < 99.0:
        log("    ⚠️  < 100% nghĩa là query encode lại KHÁC lúc tạo GT (sai model/version/normalize). "
            "Số baseline vẫn chạy nhưng nên xem lại tính nhất quán.")

    results = {}
    results["Exact Brute-Force (IndexFlatIP)"] = (bf_s, bf_m, "Centralized reference = Ground Truth")

    # 4) HNSW
    log("[*] Baseline HNSW ...")
    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M)         # METRIC_L2; với vector chuẩn hoá ~ cosine
    hnsw.hnsw.efConstruction = HNSW_EF
    hnsw.add(E)
    hnsw.hnsw.efSearch = HNSW_EF
    _, Ih = hnsw.search(Qv, 5)
    hnsw_ret = [[int(i) for i in row if i >= 0] for row in Ih]
    s, m = success_and_mrr(hnsw_ret, gt_sets)
    results[f"FAISS-HNSW (M={HNSW_M}, ef={HNSW_EF})"] = (s, m, "Centralized graph ANN")
    log(f"    Success@5={s:.1f}%  MRR@5={m:.3f}")

    # 5) Chuẩn bị SRP keys (trùng V-Engram) + crypto keys
    initialize_lsh_projections(LSH_SEED)
    projs = routing.PROJECTION_MATRICES            # list L ma trận (1024,160)
    doc_bits = [srp_bits(E, projs[t]) for t in range(NUM_PROJECTIONS)]   # mỗi cái (N,160) bool
    q_bits   = [srp_bits(Qv, projs[t]) for t in range(NUM_PROJECTIONS)]  # (Q,160) bool

    # crypto: top-64 bit của SHA-256 (XOR bị MSB chi phối — đúng tinh thần Kademlia)
    doc_ck = {t: np.array([sha160(f"doc_{i}_tbl_{t}") >> 96 for i in range(N)], dtype=np.uint64)
              for t in range(NUM_PROJECTIONS)}

    # 6) Centralized Multi-Table LSH (TRẦN): gom theo HAMMING, global view, rồi rerank
    log("[*] Baseline Centralized Multi-Table LSH (Hamming gather) ...")
    lsh_ret, pool_sizes = [], []
    for q in range(Q):
        cand = set()
        for t in range(NUM_PROJECTIONS):
            ham = (doc_bits[t] ^ q_bits[t][q]).sum(axis=1)        # (N,) Hamming
            top = np.argpartition(ham, POOL_PER_TABLE)[:POOL_PER_TABLE]
            cand.update(int(i) for i in top)
        pool_sizes.append(len(cand))
        lsh_ret.append(rerank(Qv[q], cand, E, pq_codes, codebook))
    s, m = success_and_mrr(lsh_ret, gt_sets)
    results["Centralized Multi-Table LSH"] = (s, m, f"LSH signal ceiling, no DHT hops (pool~{int(np.mean(pool_sizes))})")
    log(f"    Success@5={s:.1f}%  MRR@5={m:.3f}  pool TB={np.mean(pool_sizes):.0f}")

    # 7) Crypto-DHT cùng ngân sách (SÀN có rerank): gom theo XOR trên key ngẫu nhiên
    log("[*] Baseline Crypto-DHT (XOR gather, key SHA-256, cùng ngân sách) ...")
    crypto_ret = []
    for q in range(Q):
        cand = set()
        for t in range(NUM_PROJECTIONS):
            qk = np.uint64(sha160(f"query_{q}_tbl_{t}") >> 96)
            d = doc_ck[t] ^ qk
            top = np.argpartition(d, POOL_PER_TABLE)[:POOL_PER_TABLE]
            cand.update(int(i) for i in top)
        crypto_ret.append(rerank(Qv[q], cand, E, pq_codes, codebook))
    s, m = success_and_mrr(crypto_ret, gt_sets)
    results["Crypto-DHT (same budget + rerank)"] = (s, m, "DHT floor: random keys, semantics removed")
    log(f"    Success@5={s:.1f}%  MRR@5={m:.3f}")

    # 8) Random-5 (pure chance, KHÔNG rerank) — nguồn của con số ~0.1% cũ
    log("[*] Baseline Random-5 (pure chance) ...")
    rng = np.random.RandomState(RNG_SEED)
    rand_ret = [[int(i) for i in rng.choice(N, size=5, replace=False)] for _ in range(Q)]
    s, m = success_and_mrr(rand_ret, gt_sets)
    results["Random-5 (no rerank)"] = (s, m, "Pure chance ~ 1-(1-5/N)^5")
    log(f"    Success@5={s:.2f}%  MRR@5={m:.3f}")

    # 9) In bảng + ghi báo cáo
    header = f"{'Method':40s} {'Success@5':>10s} {'MRR@5':>8s}   Note"
    lines = [header, "-" * len(header)]
    order = ["Exact Brute-Force (IndexFlatIP)",
             f"FAISS-HNSW (M={HNSW_M}, ef={HNSW_EF})",
             "Centralized Multi-Table LSH",
             "Crypto-DHT (same budget + rerank)",
             "Random-5 (no rerank)"]
    for name in order:
        s, m, note = results[name]
        lines.append(f"{name:40s} {s:9.2f}% {m:8.3f}   {note}")
    lines.append(f"{'V-Engram (P2P, từ comparison_report)':40s} {'~97.2%':>10s} {'~0.97':>8s}   "
                 f"Distributed rendezvous (chạy từ simulation)")
    table = "\n".join(lines)

    log("\n" + "=" * len(header))
    log(table)
    log("=" * len(header))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"BÁO CÁO BASELINE — corpus={CORPUS}, N={N}, Q={Q}, L={NUM_PROJECTIONS}, "
                f"pool/table={POOL_PER_TABLE}, rerank={RERANK}\n\n")
        f.write(table + "\n")
    log(f"\n[*] Đã ghi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
