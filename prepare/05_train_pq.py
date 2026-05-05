import time
import numpy as np
import faiss


def train_pq(save_dir):
    print("[Prepare] Bat dau huan luyen Product Quantization...")

    vectors = np.load(f"{save_dir}/embeddings_20k.npy")
    d = 1024
    m = 256
    nbits = 8

    print(f"[Prepare] Cau hinh: {d} chieu -> {m} sub-vectors x 256 centroids")
    pq = faiss.IndexPQ(d, m, nbits)

    start_time = time.time()
    print("[Prepare] Dang train PQ codebook...")
    pq.train(vectors)
    print(f"[Prepare] Thoi gian train: {time.time() - start_time:.2f}s")

    pq_codes = pq.sa_encode(vectors)

    centroids_1d = faiss.vector_to_array(pq.pq.centroids)
    pq_codebook = centroids_1d.reshape(m, 2**nbits, d // m)

    np.save(f"{save_dir}/pq_codes.npy", pq_codes)
    np.save(f"{save_dir}/pq_codebook.npy", pq_codebook)

    print("[Prepare] Da luu:")
    print(f"[Prepare] - {save_dir}/pq_codes.npy")
    print(f"[Prepare] - {save_dir}/pq_codebook.npy")


if __name__ == "__main__":
    train_pq(save_dir="./data")
