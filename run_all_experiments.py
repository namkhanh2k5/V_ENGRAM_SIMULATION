"""
============================================================================
 V-ENGRAM — CHẠY TOÀN BỘ SIMULATION TRONG MỘT LỆNH (bản PAYLOAD-ONCE)
============================================================================
Một lệnh:
    python run_all_experiments.py
Chạy nền (khuyên dùng vì rất lâu):
    bash run_all.sh --bg

MỌI kết quả (cả 5 khối) ghi vào CÙNG MỘT file: logs/v_engram_all_<timestamp>.log

Gồm 5 khối:
  1) L-ABLATION (Code corpus, L=1..5)  -> bảng consolidated_l   [nặng nhất]
  2) SCIFACT  (L=5)                    -> bảng scifact
  3) CHURN    (Code corpus, 10/20/30%) -> §4.9
  4) SCALABILITY (Code corpus)          -> bảng scalability
  5) PLACEMENT BREADTH                  -> bảng placement

Số kỳ vọng sau khi sửa payload-once:
  - Mean Load (shards/node) ~ files*30/nodes, KHÔNG đổi theo L; Std/Max thấp (hết hotspot payload).
  - Metric chinh la Recall@5 (Hit@5 bao hoa o 100%).
  - KHONG tang METADATA_ANCHORS de 'cuu' recall: r lon lam random routing THANG
    semantic routing (r=30: random 73.8% vs semantic 48.2%). Dung multi-probe (T).
  - Unique-candidate/query: phép đo mới cho Q1.

>>> Kiểm CONFIG đường dẫn ./data/ bên dưới cho khớp server <<<
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENV_PYTHON = os.path.join(_HERE, "venv", "bin", "python")
if __name__ == "__main__" and os.path.exists(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import time, json, random, logging, contextlib
from dataclasses import replace, dataclass
from typing import Any
import numpy as np
import simpy

import src.routing as routing
from src.network import (
    bootstrap_network, data_ingestion_process, query_pipeline_process,
    reset_global_metadata_dht, GLOBAL_METADATA_DHT, PLACEMENT_CANDIDATES,
)
from src.routing import generate_placement_key, iterative_find_k_closest_nodes
import src.network as net   # để override net.PLACEMENT_CANDIDATES khi quét placement breadth

# ============================ CONFIG ============================
SEEDS  = [20235956, 2026, 11, 12, 18]      # chạy HẾT 5 seed cho cả 3 khối
SHARDS = 30
MODEL_NAME = "BAAI/bge-large-en-v1.5"


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    emb: str
    pq: str
    cb: str
    gt: str
    num_nodes: int
    num_files: int | None


# --- Code corpus (m=256, do prepare/02+05 sinh) ---
CODE = CorpusConfig(
    name="CODE",
    emb="./data/code_corpus_embeddings.npy",
    pq="./data/code_pq_codes.npy",
    cb="./data/code_pq_codebook.npy",
    gt="./data/code_ground_truth.json",
    num_nodes=10000, num_files=20000,
)
L_VALUES_CODE = [1, 2, 3, 4, 5]

# --- SciFact corpus (bộ đã xác nhận chạy được) ---
SCIFACT = CorpusConfig(
    name="SCIFACT",
    emb="./data/scifact_corpus_embeddings.npy",
    pq="./data/scifact_pq_codes.npy",
    cb="./data/scifact_pq_codebook.npy",
    gt="./data/scifact_ground_truth.json",
    num_nodes=10000, num_files=None,        # None -> dùng embeddings.shape[0]
)
L_SCIFACT = 5

# --- Churn ---
CHURN_STEPS = [0.10, 0.20, 0.30]
RECOVERY_SAMPLE = 500
K_REQUIRED = 20

# --- Scalability (BLOCK 4) & Placement breadth (BLOCK 5): chạy 1 seed như bài gốc ---
L_DEFAULT       = 5                              # cấu hình đầy đủ (5 bảng) cho scalability/placement
SWEEP_SEED      = SEEDS[0]                       # 1 seed cho 2 khối này (đỡ lâu)
NODE_SWEEP      = [10000, 15000, 20000, 25000]   # scalability: corpus 20k cố định, đổi số node
PLACEMENT_SWEEP = [50, 100, 150, 200, 250]       # placement breadth B_place (= PLACEMENT_CANDIDATES)
# ================================================================

os.makedirs("logs", exist_ok=True)
ONE_LOG = f"logs/v_engram_all_{time.strftime('%Y%m%d_%H%M%S')}.log"
log = logging.getLogger("ALL")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(ONE_LOG, encoding="utf-8")):
    h.setFormatter(_fmt); log.addHandler(h)


@contextlib.contextmanager
def quiet():
    """Nuốt log dài (ingest/query) của src để file log chỉ chứa KẾT QUẢ sạch."""
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL_NAME, local_files_only=True)
    return _MODEL


def _seed(seed, L):
    random.seed(seed); np.random.seed(seed)
    routing.NUM_PROJECTIONS = L
    routing.initialize_lsh_projections(seed)
    reset_global_metadata_dht()


def run_retrieval(
    corpus: CorpusConfig,
    L: int,
    seed: int,
    model: Any,
    codebook: np.ndarray,
    ground_truth: list[dict[str, Any]],
    num_files: int,
):
    """Một lần chạy truy hồi đầy đủ (dùng chung Code & SciFact). Trả dict số liệu."""
    _seed(seed, L)
    res = {}

    def proc(env):
        nodes = yield env.process(bootstrap_network(env, corpus.num_nodes, 50, 50))
        yield env.process(data_ingestion_process(
            env, nodes, num_files, SHARDS,
            embeddings_path=corpus.emb, pq_codes_path=corpus.pq,
            data_label=f"{corpus.name}-L{L}-s{seed}"))

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
                if idx in gt: mrr += 1.0 / rank; break

        q = len(ground_truth) or 1
        loads = [len(n.SSD_Storage) for n in nodes]
        res.update(
            success=100.0*hit/q, mrr=mrr/q, hops=hops_all/q,
            load_mean=float(np.mean(loads)), load_std=float(np.std(loads)),
            load_max=int(np.max(loads)),
            uniq_mean=float(np.mean(uniqs)), uniq_pct=100.0*float(np.mean(uniqs))/num_files,
        )
    env = simpy.Environment(); env.process(proc(env)); env.run()
    return res


def run_ingest_only(corpus: CorpusConfig, L: int, seed: int, num_files: int):
    """Chỉ ingest + đo phân bố load (KHÔNG query) — cho placement-breadth ablation."""
    _seed(seed, L)
    res = {}

    def proc(env):
        nodes = yield env.process(bootstrap_network(env, corpus.num_nodes, 50, 50))
        yield env.process(data_ingestion_process(
            env, nodes, num_files, SHARDS,
            embeddings_path=corpus.emb, pq_codes_path=corpus.pq,
            data_label=f"{corpus.name}-ingest"))
        loads = [len(n.SSD_Storage) for n in nodes]
        res.update(load_mean=float(np.mean(loads)), load_std=float(np.std(loads)), load_max=int(np.max(loads)))
    env = simpy.Environment(); env.process(proc(env)); env.run()
    return res


def _can_recover(tag, nodes):
    """Khôi phục payload-once: 30 placement key từ tag, cần >=20/30 shard sống."""
    if not GLOBAL_METADATA_DHT.get(tag):
        return False
    got = 0
    for s_id in range(SHARDS):
        if not nodes:
            break
        p_key = generate_placement_key(tag, s_id)
        cand, _, _ = iterative_find_k_closest_nodes(p_key, random.choice(nodes), alpha=3, k=PLACEMENT_CANDIDATES)
        for n in cand:
            if f"{tag}_shard_{s_id}" in n.SSD_Storage:
                got += 1; break
        if got >= K_REQUIRED:
            break
    return got >= K_REQUIRED


def run_churn(seed, codebook, num_files):
    """Ingest Code corpus rồi giết node theo từng mức churn, đo % khôi phục."""
    _seed(seed, L=5)
    out = {}

    def proc(env):
        nodes = yield env.process(bootstrap_network(env, CODE.num_nodes, 50, 50))
        yield env.process(data_ingestion_process(
            env, nodes, num_files, SHARDS,
            embeddings_path=CODE.emb, pq_codes_path=CODE.pq, data_label=f"CHURN-s{seed}"))

        rnd = random.Random(seed)
        doc_ids = rnd.sample(range(num_files), min(RECOVERY_SAMPLE, num_files))
        total = len(nodes)
        for ratio in CHURN_STEPS:
            desired = int(total * (1 - ratio))
            kill = max(0, len(nodes) - desired)
            if kill:
                killed = set(rnd.sample(nodes, min(kill, len(nodes))))
                nodes[:] = [n for n in nodes if n not in killed]
                for n in nodes:
                    n.routing_table.difference_update(killed)
            rec = sum(1 for d in doc_ids if _can_recover(f"doc_{d}", nodes))
            out[ratio] = 100.0 * rec / len(doc_ids)
    env = simpy.Environment(); env.process(proc(env)); env.run()
    return out


def _mean(rows, k):
    return float(np.mean([r[k] for r in rows]))
def _std(rows, k):
    return float(np.std([r[k] for r in rows]))


def main():
    t_all = time.time()
    n_runs = (len(L_VALUES_CODE)*len(SEEDS) + len(SEEDS) + len(SEEDS)
              + len(NODE_SWEEP) + len(PLACEMENT_SWEEP))
    log.info("="*84)
    log.info("V-ENGRAM RUN-ALL (PAYLOAD-ONCE) | seeds=%s | log: %s", SEEDS, ONE_LOG)
    log.info("Tổng ~%d lần ingest: L-ablation %d + SciFact %d + Churn %d + Scalability %d + Placement %d. RẤT LÂU.",
             n_runs, len(L_VALUES_CODE)*len(SEEDS), len(SEEDS), len(SEEDS),
             len(NODE_SWEEP), len(PLACEMENT_SWEEP))
    log.info("="*84)

    log.info("[*] Nạp model %s ...", MODEL_NAME)
    try:
        model = get_model()
        code_cb = np.load(CODE.cb); code_gt = json.load(open(CODE.gt, encoding="utf-8"))
        code_nf = min(CODE.num_files or 0, np.load(CODE.emb, mmap_mode="r").shape[0])
        sci_cb = np.load(SCIFACT.cb); sci_gt = json.load(open(SCIFACT.gt, encoding="utf-8"))
        sci_nf = int(np.load(SCIFACT.emb, mmap_mode="r").shape[0])
    except Exception as e:
        log.error("Lỗi nạp model/data: %s — kiểm CONFIG đường dẫn ./data/.", e); sys.exit(1)

    done = 0

    # ---------------- BLOCK 1: L-ABLATION (Code) ----------------
    log.info("\n" + "#"*84 + "\n# BLOCK 1 — L-ABLATION (Code corpus) -> consolidated_l\n" + "#"*84)
    code_rows = []
    for L in L_VALUES_CODE:
        per = []
        for s in SEEDS:
            t0 = time.time()
            with quiet():
                r = run_retrieval(CODE, L, s, model, code_cb, code_gt, code_nf)
            done += 1
            log.info("  [%d/%d] L=%d seed=%-9d | %5.0fs | Succ=%.1f%% MRR=%.3f Hops=%.1f Load=%.1f/%.1f/%d Uniq=%.1f",
                     done, n_runs, L, s, time.time()-t0, r["success"], r["mrr"], r["hops"],
                     r["load_mean"], r["load_std"], r["load_max"], r["uniq_mean"])
            per.append(r)
        code_rows.append((L, per))
        log.info("  == L=%d (TB %d seed): Succ=%.1f%%±%.1f | MRR=%.3f±%.3f | Hops=%.1f | Load=%.1f/%.1f/%d | Uniq=%.1f (~%.2f%%)",
                 L, len(SEEDS), _mean(per,"success"), _std(per,"success"),
                 _mean(per,"mrr"), _std(per,"mrr"), _mean(per,"hops"),
                 _mean(per,"load_mean"), _mean(per,"load_std"), int(_mean(per,"load_max")),
                 _mean(per,"uniq_mean"), _mean(per,"uniq_pct"))

    # ---------------- BLOCK 2: SCIFACT (5 seeds) ----------------
    log.info("\n%s\n# BLOCK 2 — SCIFACT (L=%d, 5 seeds) -> scifact\n%s", "#"*84, L_SCIFACT, "#"*84)
    sci_per = []
    for s in SEEDS:
        t0 = time.time()
        with quiet():
            r = run_retrieval(SCIFACT, L_SCIFACT, s, model, sci_cb, sci_gt, sci_nf)
        done += 1
        log.info("  [%d/%d] SciFact seed=%-9d | %5.0fs | Succ=%.1f%% MRR=%.3f Hops=%.1f Load=%.1f/%.1f/%d Uniq=%.1f",
                 done, n_runs, s, time.time()-t0, r["success"], r["mrr"], r["hops"],
                 r["load_mean"], r["load_std"], r["load_max"], r["uniq_mean"])
        sci_per.append(r)
    log.info("  == SciFact (TB %d seed): Succ=%.1f%%±%.1f | MRR=%.3f±%.3f | Hops=%.1f | Load=%.1f/%.1f/%d | Uniq=%.1f (~%.2f%%)",
             len(SEEDS), _mean(sci_per,"success"), _std(sci_per,"success"),
             _mean(sci_per,"mrr"), _std(sci_per,"mrr"), _mean(sci_per,"hops"),
             _mean(sci_per,"load_mean"), _mean(sci_per,"load_std"), int(_mean(sci_per,"load_max")),
             _mean(sci_per,"uniq_mean"), _mean(sci_per,"uniq_pct"))

    # ---------------- BLOCK 3: CHURN (5 seeds) ----------------
    log.info("\n%s\n# BLOCK 3 — CHURN (Code corpus, %d seeds) -> §4.9\n%s", "#"*84, len(SEEDS), "#"*84)
    churn_per = []
    for s in SEEDS:
        t0 = time.time()
        with quiet():
            cr = run_churn(s, code_cb, code_nf)
        done += 1
        churn_str = " ".join(f"{int(r*100)}%={cr[r]:.1f}%" for r in CHURN_STEPS)
        log.info("  [%d/%d] Churn seed=%-9d | %5.0fs | %s", done, n_runs, s, time.time()-t0, churn_str)
        churn_per.append(cr)

    # ---------------- BLOCK 4: SCALABILITY (Code, 1 seed) ----------------
    log.info("\n%s\n# BLOCK 4 — SCALABILITY (Code, corpus 20k cố định, 1 seed) -> scalability\n%s", "#"*84, "#"*84)
    scal_rows = []
    for N in NODE_SWEEP:
        t0 = time.time()
        corpus_N = replace(CODE, num_nodes=N)
        with quiet():
            r = run_retrieval(corpus_N, L_DEFAULT, SWEEP_SEED, model, code_cb, code_gt, code_nf)
        done += 1
        log.info("  [%d/%d] Nodes=%-6d | %5.0fs | Succ=%.1f%% MRR=%.3f Hops=%.1f Load=%.1f/%.1f/%d",
                 done, n_runs, N, time.time()-t0, r["success"], r["mrr"], r["hops"],
                 r["load_mean"], r["load_std"], r["load_max"])
        scal_rows.append((N, r))

    # ---------------- BLOCK 5: PLACEMENT BREADTH B_place (Code, 1 seed) ----------------
    log.info("\n%s\n# BLOCK 5 — PLACEMENT BREADTH B_place (Code, 1 seed, chỉ đo load) -> placement\n%s", "#"*84, "#"*84)
    plc_rows = []
    _orig_pc = net.PLACEMENT_CANDIDATES
    for B in PLACEMENT_SWEEP:
        net.PLACEMENT_CANDIDATES = B          # đổi breadth lúc đặt shard
        t0 = time.time()
        with quiet():
            r = run_ingest_only(CODE, L_DEFAULT, SWEEP_SEED, code_nf)
        done += 1
        log.info("  [%d/%d] B_place=%-4d | %5.0fs | Load mean/std/max = %.1f / %.1f / %d",
                 done, n_runs, B, time.time()-t0, r["load_mean"], r["load_std"], r["load_max"])
        plc_rows.append((B, r))
    net.PLACEMENT_CANDIDATES = _orig_pc       # khôi phục

    # ---------------- TỔNG KẾT ----------------
    log.info("\n" + "="*84 + "\nTỔNG KẾT — DÁN VÀO BÀI\n" + "="*84)

    log.info("\n[consolidated_l] (Code, %d seed)", len(SEEDS))
    log.info("%-3s %-14s %-14s %-7s %-22s %-16s", "L", "Success@5", "MRR@5", "Hops", "Load mean/std/max", "UniqCand(%corpus)")
    for L, per in code_rows:
        log.info("%-3d %-14s %-14s %-7.1f %-22s %-16s",
                 L, f"{_mean(per,'success'):.1f}±{_std(per,'success'):.1f}",
                 f"{_mean(per,'mrr'):.3f}±{_std(per,'mrr'):.3f}", _mean(per,"hops"),
                 f"{_mean(per,'load_mean'):.1f}/{_mean(per,'load_std'):.1f}/{int(_mean(per,'load_max'))}",
                 f"{_mean(per,'uniq_mean'):.1f} (~{_mean(per,'uniq_pct'):.2f}%)")

    log.info("\n[scifact] (L=%d, %d seed): Success=%.1f%%±%.1f | MRR=%.3f±%.3f | Hops=%.1f | Load mean/std/max=%.1f/%.1f/%d | UniqCand=%.1f (~%.2f%%)",
             L_SCIFACT, len(SEEDS), _mean(sci_per,"success"), _std(sci_per,"success"),
             _mean(sci_per,"mrr"), _std(sci_per,"mrr"), _mean(sci_per,"hops"),
             _mean(sci_per,"load_mean"), _mean(sci_per,"load_std"), int(_mean(sci_per,"load_max")),
             _mean(sci_per,"uniq_mean"), _mean(sci_per,"uniq_pct"))

    log.info("\n[churn §4.9] (Code, %d seed) — %% khôi phục:", len(SEEDS))
    for r in CHURN_STEPS:
        vals = [cr[r] for cr in churn_per]
        log.info("  Mất %d%%: %.1f%% ± %.1f%%", int(r*100), float(np.mean(vals)), float(np.std(vals)))

    log.info("\n[scalability] (Code, 1 seed, L=%d, corpus 20k):", L_DEFAULT)
    log.info("%-8s %-12s %-8s %-22s", "Nodes", "Success@5", "Hops", "Load mean/std/max")
    for N, r in scal_rows:
        log.info("%-8d %-12.1f %-8.1f %-22s", N, r["success"], r["hops"],
                 f"{r['load_mean']:.1f}/{r['load_std']:.1f}/{int(r['load_max'])}")

    log.info("\n[placement] (Code, 1 seed, L=%d) — B_place vs phân bố load:", L_DEFAULT)
    log.info("%-9s %-22s", "B_place", "Load mean/std/max")
    for B, r in plc_rows:
        log.info("%-9d %-22s", B, f"{r['load_mean']:.1f}/{r['load_std']:.1f}/{int(r['load_max'])}")

    log.info("\n✓ XONG sau %.1f phút. TẤT CẢ kết quả ở: %s", (time.time()-t_all)/60, ONE_LOG)


if __name__ == "__main__":
    main()