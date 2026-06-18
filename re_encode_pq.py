import numpy as np
import faiss
import os

print("[*] Đang tải Embeddings 1024 chiều gốc...")
# Load lại vector chưa bị nén
chunk_embeddings = np.load("./data/rag_embeddings_20k.npy")

print("[*] Đang huấn luyện PQ Codebook với độ phân giải CAO (512 bytes)...")
d = 1024  # Số chiều của BAAI
m = 512   # TĂNG GẤP ĐÔI TỪ 256 LÊN 512!
pq = faiss.IndexPQ(d, m, 8) 
pq.train(chunk_embeddings)

print("[*] Đang nén Vector thành PQ Codes (uint8)...")
pq_codes = pq.sa_encode(chunk_embeddings)
np.save("./data/rag_pq_codes_512.npy", pq_codes)

# Trích xuất Centroids làm Codebook mới
centroids = faiss.vector_to_array(pq.pq.centroids).reshape(m, 256, d // m)
np.save("./data/rag_pq_codebook_512.npy", centroids)

print("✅ ĐÃ XONG! TẠO THÀNH CÔNG rag_pq_codes_512.npy VÀ rag_pq_codebook_512.npy")