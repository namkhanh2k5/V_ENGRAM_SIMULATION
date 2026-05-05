import simpy
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from src.network import bootstrap_network, data_ingestion_process, query_pipeline_process
from src.evaluation import stage5_metrics_collection, export_comparison_report

NUM_NODES = 10000
NUM_FILES = 20000
SHARDS_PER_FILE = 30
K_SIZE = 20

def run_simulation(env):
    # CHỈ CẦN NẠP BẢNG MÃ CODEBOOK VÀO HỆ THỐNG
    print("[*] Đang nạp PQ Codebook (chuẩn Web3)...")
    try:
        codebook = np.load("./data/pq_codebook.npy")
    except Exception as e:
        print(f"❌ Lỗi nạp data: {e}. Vui lòng kiểm tra lại thư mục ./data/")
        return

    # Khởi tạo mạng lưới
    network_nodes = yield env.process(bootstrap_network(env, NUM_NODES, K_SIZE))
    
    # Đẩy 20k files vào mạng
    yield env.process(data_ingestion_process(env, network_nodes, NUM_FILES, SHARDS_PER_FILE))
    
    # Truy vấn và khôi phục
    try:
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        with open("./data/faiss_absolute_baseline.json", "r", encoding="utf-8") as f:
            ground_truth = json.load(f)
            
        test_results = []
        for item in ground_truth:
            q_id = item['query_id']
            q_text = item['query_text']
            
            query_vector = model.encode(q_text)
            
            print(f"\n>>> TRUY VẤN #{q_id}: '{q_text}'")
            retrieved_tags, hops = yield env.process(
                query_pipeline_process(env, network_nodes, query_vector, codebook, target_k=5)
            )
            test_results.append({
                "query_id": q_id,
                "retrieved": retrieved_tags,
                "hops": hops
            })
            
        # Thu hoạch số liệu
        stage5_metrics_collection(env, network_nodes, test_results)
        export_comparison_report(test_results, "./data/faiss_absolute_baseline.json", "comparison_report.txt")

    except Exception as e:
        print(f"❌ Lỗi trong quá trình chạy: {e}")

# ĐÂY LÀ NÚT CÔNG TẮC - THIẾU NÓ LÀ CỖ MÁY IM LẶNG!
if __name__ == "__main__":
    env = simpy.Environment()
    env.process(run_simulation(env))
    env.run()