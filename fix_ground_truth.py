import numpy as np
import json
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 1. Tải bộ dữ liệu đã được làm sạch
dataset = load_dataset("squad", split="train")
contexts = list(set(dataset['context']))  # Hàm set sẽ chỉ để lại 18,891 đoạn
test_samples = dataset.shuffle(seed=42).select(range(500))
queries = test_samples['question']

print("[*] Đang Load 18,891 Vector cũ (Không cần nhúng lại)...")
chunk_embeddings = np.load("./data/rag_embeddings_20k.npy")

print("[*] Đang tạo Oracle FAISS mới cho 18,891 chunks...")
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
query_embeddings = np.array(model.encode(queries, normalize_embeddings=True), dtype=np.float32)

index_flat = faiss.IndexFlatL2(1024)
index_flat.add(chunk_embeddings)
distances, indices = index_flat.search(query_embeddings, 5)

ground_truth = []
for i in range(len(queries)):
    top_5 = [{"rank": r+1, "index": int(idx)} for r, idx in enumerate(indices[i])]
    ground_truth.append({
        "query_id": i,
        "query_text": queries[i],
        "top_5_results": top_5
    })

with open("./data/rag_faiss_absolute_baseline.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, ensure_ascii=False, indent=4)
print("✅ Đã vá xong lỗ hổng Ground Truth!")