"""
QUÉT L-ABLATION (L=1..5) CHO BẢNG consolidated_l — PHIÊN BẢN PAYLOAD-ONCE.

Chạy lại pipeline two-tier đã sửa (metadata nhân L lần, payload đặt 1 lần) trên
Code corpus, với từng L và (tuỳ chọn) nhiều seed, rồi ghi TẤT CẢ số liệu ra log.

Số kỳ vọng SAU khi sửa:
  - Mean Load (shards/node) ~ NUM_FILES*30/NUM_NODES  -> KHÔNG đổi theo L (payload-once)
  - Success@5 / MRR@5 / Avg Hops ~ GIỮ NGUYÊN so với bản cũ (discovery qua metadata anchor)
  - Unique-candidate/query: số object thực sự được rerank (phép đo mới cho Q1)

>>> ĐIỀN ĐÚNG đường dẫn data Code-corpus của anh ở phần CONFIG bên dưới <<<
"""
import sys, time, logging, random
import numpy as np
import simpy

import src.routing as routing
from src.network import (
    bootstrap_network, data_ingestion_process, query_pipeline_process,
    reset_global_metadata_dht,
)

# ============================ CONFIG ============================
NUM_NODES   = 10000
NUM_FILES   = 20000
SHARDS      = 30
L_VALUES    = [1, 2, 3, 4, 5]
SEEDS       = [20235956, 2026, 11, 12, 18]   # chạy HẾT 5 seed trong 1 lần -> mean±std cho consolidated_l
MODEL_NAME  = "BAAI/bge-large-en-v1.5"

# --- ĐƯỜNG DẪN DATA CODE-CORPUS (sửa cho khớp máy anh) ---
EMBEDDINGS_PATH   = "./data/code_corpus_embeddings.npy"
PQ_CODES_PATH     = "./data/code_pq_codes.npy"
PQ_CODEBOOK_PATH  = "./data/code_pq_codebook.npy"
GROUND_TRUTH_PATH = "./data/code_ground_truth.json"

LOG_PATH = f"logs/L_ablation_{time.strftime('%Y%m%d_%H%M%S')}.log"
# ================================================================

import os, json
os.makedirs("logs", exist_ok=True)
log = logging.getLogger("Lablation")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding="utf-8")):
    h.setFormatter(_fmt); log.addHandler(h)


def run_one(L, seed, model, codebook, ground_truth, embeddings):
    """Một lần chạy đầy đủ cho (L, seed). Trả về dict số liệu."""
    random.seed(seed); np.random.seed(seed)
    routing.NUM_PROJECTIONS = L
    routing.initialize_lsh_projections(seed)
    reset_global_metadata_dht()

    num_files = min(NUM_FILES, embeddings.shape[0])
    result = {}

    def proc(env):
        nodes = yield env.process(bootstrap_network(env, NUM_NODES, 50, 50))
        yield env.process(data_ingestion_process(
            env, nodes, num_files, SHARDS,
            embeddings_path=EMBEDDINGS_PATH, pq_codes_path=PQ_CODES_PATH,
            data_label=f"CODE(L={L},seed={seed})"))

        hit = 0; mrr = 0.0; hops_all = 0; uniqs = []
        for item in ground_truth:
            qv = model.encode(item["query_text"])
            tags, hops, n_uniq, _stats = yield env.process(
                query_pipeline_process(env, nodes, qv, codebook, target_k=5))
            hops_all += hops; uniqs.append(n_uniq)
            gt = set(r["index"] for r in item["top_5_results"])
            ridx = []
            for t in tags:
                try: ridx.append(int(t.split("_")[1]))
                except: pass
            if set(ridx) & gt: hit += 1
            for rank, idx in enumerate(ridx, 1):
                if idx in gt: mrr += 1.0/rank; break

        q = len(ground_truth) or 1
        loads = [len(n.SSD_Storage) for n in nodes]
        result.update(
            L=L, seed=seed,
            success=100.0*hit/q, mrr=mrr/q, hops=hops_all/q,
            load_mean=float(np.mean(loads)), load_std=float(np.std(loads)),
            load_max=int(np.max(loads)),
            uniq_mean=float(np.mean(uniqs)), uniq_pct=100.0*float(np.mean(uniqs))/num_files,
        )
    env = simpy.Environment(); env.process(proc(env)); env.run()
    return result


def main():
    log.info("="*78)
    log.info("L-ABLATION (PAYLOAD-ONCE) | NUM_NODES=%d NUM_FILES=%d SHARDS=%d", NUM_NODES, NUM_FILES, SHARDS)
    log.info("Kỳ vọng Mean Load ~ %.1f (KHÔNG đổi theo L) | Success/MRR/Hops phải GIỮ NGUYÊN",
             NUM_FILES*SHARDS/NUM_NODES)
    log.info("="*78)

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
        codebook = np.load(PQ_CODEBOOK_PATH)
        ground_truth = json.load(open(GROUND_TRUTH_PATH, encoding="utf-8"))
        embeddings = np.load(EMBEDDINGS_PATH)
    except Exception as e:
        log.error("Lỗi nạp data/model: %s — kiểm tra lại CONFIG đường dẫn.", e); sys.exit(1)

    rows = []
    total = len(L_VALUES) * len(SEEDS)
    log.warning("SẼ CHẠY %d lần ingest đầy đủ (%d L × %d seed) trên %d node × %d doc — RẤT LÂU.",
                total, len(L_VALUES), len(SEEDS), NUM_NODES, NUM_FILES)
    log.warning("Có thể chạy nền: nohup bash run_all.sh &   (kết quả vẫn ghi vào logs/).")
    done = 0
    for L in L_VALUES:
        per_seed = []
        for s in SEEDS:
            t0 = time.time()
            r = run_one(L, s, model, codebook, ground_truth, embeddings)
            done += 1
            log.info("  [%d/%d] L=%d seed=%d | %.0fs | Success=%.1f%% Load_mean=%.1f Std=%.1f Max=%d UniqCand=%.1f",
                     done, total, L, s, time.time()-t0,
                     r["success"], r["load_mean"], r["load_std"], r["load_max"], r["uniq_mean"])
            per_seed.append(r)
        agg = {k: float(np.mean([r[k] for r in per_seed])) for k in
               ("success","mrr","hops","load_mean","load_std","load_max","uniq_mean","uniq_pct")}
        agg_std_succ = float(np.std([r["success"] for r in per_seed]))
        rows.append((L, agg, agg_std_succ))
        log.info("== L=%d (TB %d seed): Success=%.1f%%±%.1f | MRR=%.3f | Hops=%.1f | Load(mean/std/max)=%.1f/%.1f/%d | UniqCand=%.1f (~%.2f%%)",
                 L, len(SEEDS), agg["success"], agg_std_succ, agg["mrr"], agg["hops"],
                 agg["load_mean"], agg["load_std"], agg["load_max"], agg["uniq_mean"], agg["uniq_pct"])

    log.info("\n" + "="*78 + "\nBẢNG TỔNG KẾT (dán vào consolidated_l):\n" + "="*78)
    log.info("%-3s %-12s %-8s %-8s %-22s %-16s", "L", "Success@5", "MRR", "Hops", "Load mean/std/max", "UniqCand(%corpus)")
    for L, a, _ in rows:
        log.info("%-3d %-12.1f %-8.3f %-8.1f %-22s %-16s",
                 L, a["success"], a["mrr"], a["hops"],
                 f"{a['load_mean']:.1f}/{a['load_std']:.1f}/{int(a['load_max'])}",
                 f"{a['uniq_mean']:.1f} (~{a['uniq_pct']:.2f}%)")
    log.info("\n✓ Log đầy đủ đã lưu tại: %s", LOG_PATH)


if __name__ == "__main__":
    main()