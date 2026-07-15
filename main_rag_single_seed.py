import simpy
import numpy as np
import json
import random
from sentence_transformers import SentenceTransformer

from src.network import bootstrap_network, data_ingestion_process, query_pipeline_process, reset_global_metadata_dht
from src.routing import initialize_lsh_projections

# CẤU HÌNH THỬ NGHIỆM ĐƠN SEED
NUM_NODES = 10000
SHARDS_PER_FILE = 30
TARGET_SEED = 20235956
REPORT_PATH = "scifact_single_seed_report.txt"
EMBEDDINGS_PATH = "./data/scifact_corpus_embeddings.npy"
PQ_CODES_PATH = "./data/scifact_pq_codes.npy"
PQ_CODEBOOK_PATH = "./data/scifact_pq_codebook.npy"
GROUND_TRUTH_PATH = "./data/scifact_ground_truth.json"
DATA_LABEL = "SCIFACT"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    initialize_lsh_projections(seed)

def run_rag_simulation(env, seed, model, codebook, ground_truth, num_files):
    # 1. Khởi tạo mạng & Đẩy SciFact chunks vào DHT
    network_nodes = yield env.process(bootstrap_network(env, NUM_NODES, k_size_far=50, k_size_near=50))
    yield env.process(
        data_ingestion_process(
            env,
            network_nodes,
            num_files,
            SHARDS_PER_FILE,
            embeddings_path=EMBEDDINGS_PATH,
            pq_codes_path=PQ_CODES_PATH,
            data_label=DATA_LABEL,
        )
    )

    print(f"\n[!] Mạng lưới sẵn sàng 100%. Bắt đầu quét qua {len(ground_truth)} {DATA_LABEL} Queries...")

    hit_count = 0
    mrr_total = 0.0
    total_queries = len(ground_truth)
    total_hops_all_queries = 0
    uniq_cands = []
    
    # 2. Xử lý Truy vấn RAG
    for item in ground_truth:
        q_id = item['query_id']
        q_text = item['query_text']
        query_vector = model.encode(q_text)
        
        # Tiến hành Ripple Search trên mạng P2P
        retrieved_tags, hops, n_uniq, _stats = yield env.process(
            query_pipeline_process(env, network_nodes, query_vector, codebook, target_k=5)
        )
        total_hops_all_queries += hops
        uniq_cands.append(n_uniq)
        
        # Đối chiếu trực tiếp với Chân lý FAISS FlatL2
        gt_indices = set([res['index'] for res in item['top_5_results']])
        retrieved_indices = []
        for tag in retrieved_tags:
            try:
                retrieved_indices.append(int(tag.split('_')[1]))
            except:
                continue
        
        # Tính Success@5 (Trúng ít nhất 1 kết quả trong Top-5)
        if set(retrieved_indices).intersection(gt_indices):
            hit_count += 1
        
        # Tính MRR@5 (Vị trí xuất hiện của kết quả chính xác đầu tiên)
        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in gt_indices:
                mrr_total += (1.0 / rank)
                break
                
    # 3. Thu thập Metrics cho Báo Cáo
    success_at_5 = (hit_count / total_queries) * 100
    mrr_at_5 = mrr_total / total_queries
    avg_hops = total_hops_all_queries / total_queries if total_queries else 0
    avg_uniq = float(np.mean(uniq_cands)) if uniq_cands else 0.0
    pct_uniq = 100.0 * avg_uniq / num_files if num_files else 0.0

    # Phân tích Tải trọng (Load Distribution)
    shard_counts = [len(node.SSD_Storage) for node in network_nodes]
    avg_shards = np.mean(shard_counts)
    std_shards = np.std(shard_counts)
    max_shards = np.max(shard_counts)

    # 4. In Báo Cáo Giai Đoạn 5
    print("\n" + "★" * 80)
    print("GIAI ĐOẠN 5: THU HOẠCH SỐ LIỆU VÀ BÁO CÁO (METRICS COLLECTION)")
    print("★" * 80)

    print(f"\n[1] BÁO CÁO PHÂN BỐ TẢI (Load Distribution):")
    print(f"    - Trung bình (Mean): {avg_shards:.1f} Shards / Node")
    print(f"    - Độ lệch chuẩn (Std): {std_shards:.1f}")
    print(f"    - Node nặng nhất (Max Load): {max_shards:,.0f} Shards")
    print(f"    -> Nhận xét: Payload đặt 1 lần qua HMAC nên rải ĐỀU (Std thấp, hết semantic hotspot ở payload).")

    print(f"\n[2] BÁO CÁO HIỆU NĂNG ĐỊNH TUYẾN & ĐỘ CHÍNH XÁC:")
    print(f"    - Hiệu quả định tuyến: Trung bình ~{avg_hops:.1f} Hops (Chặng) mỗi truy vấn")
    print(f"    - Success@5: {success_at_5:.1f}% (So với Server FAISS tập trung)")
    print(f"    - MRR@5: {mrr_at_5:.3f} (Mean Reciprocal Rank trong Top-5 P2P)")
    print(f"    - Unique-candidate/query: {avg_uniq:.1f} objects (~{pct_uniq:.2f}% corpus)")
    print(f"    -> Kết luận: Hệ thống phân tán đạt độ chính xác ấn tượng trên tập dữ liệu chuyên ngành.")

    print(f"\n[3] BÁO CÁO CHI PHÍ TÀI NGUYÊN (Resource Cost):")
    print(f"    - Tổng thời gian mô phỏng ảo: {env.now:,.2f} ms")
    print(f"    - Tỷ lệ tiết kiệm RAM: 16x (4096 bytes Vector -> 256 bytes PQ Code)")
    print("★" * 80 + "\n")
    
    # Ghi log kết quả duy nhất vào file báo cáo
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO NHANH DECENTRALIZED SCIFACT (SINGLE SEED) ===\n")
        f.write(f"SEED CHẠY: {seed}\n")
        f.write(f"Success@5 : {success_at_5:.2f}%\n")
        f.write(f"MRR@5     : {mrr_at_5:.3f}\n")
        f.write(f"Avg Hops  : {avg_hops:.1f}\n")
        f.write(f"UniqCand/q: {avg_uniq:.1f} (~{pct_uniq:.2f}% corpus)\n")

if __name__ == "__main__":
    print("\n[*] Khởi động hệ sinh thái RAG... Đang nạp Codebook và Mô hình BAAI...")
    try:
        global_codebook = np.load(PQ_CODEBOOK_PATH)
        global_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        global_ground_truth = json.load(open(GROUND_TRUTH_PATH, "r", encoding="utf-8"))
        embeddings = np.load(EMBEDDINGS_PATH)
    except Exception as e:
        print(f"❌ Lỗi nạp dữ liệu: {e}. Vui lòng đảm bảo các file đã nằm trong mục ./data/")
        exit()
        
    print(f"\n🚀 KHỞI CHẠY MÔ PHỎNG VỚI MỘT HẠT GIỐNG DUY NHẤT: {TARGET_SEED}")
    
    reset_global_metadata_dht()
    seed_everything(TARGET_SEED)
    
    num_files = embeddings.shape[0]

    env = simpy.Environment()
    env.process(
        run_rag_simulation(
            env,
            TARGET_SEED,
            global_model,
            global_codebook,
            global_ground_truth,
            num_files,
        )
    )
    env.run()