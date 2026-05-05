import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


def load_queries_from_baseline(save_dir, fallback_queries):
    baseline_path = f"{save_dir}/faiss_absolute_baseline.json"
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        queries = [item["query_text"] for item in baseline if "query_text" in item]
        if queries:
            print(f"[Prepare] Dung {len(queries)} query tu baseline hien co.")
            return queries
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return fallback_queries


def build_ground_truth(save_dir, num_items=20000):
    print("=" * 80)
    print("[Prepare] GIAI DOAN 1: TAO GROUND TRUTH BANG FAISS (LOCAL)")
    print("=" * 80)

    print("\n[Prepare] [1/7] Dang tai dataset CodeSearchNet...")
    start_time = time.time()
    dataset = load_dataset("code_search_net", "python", split=f"train[:{num_items}]")
    codes = dataset["func_code_string"]
    print(f"[Prepare] Da tai {len(codes):,} doan code ({time.time() - start_time:.2f}s)")

    print("\n[Prepare] [2/7] Dang tai model BGE-Large (1024 chieu)...")
    start_time = time.time()
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    print(f"[Prepare] Model san sang. Dim: {model.get_sentence_embedding_dimension()}")
    print(f"[Prepare] Thoi gian tai: {time.time() - start_time:.2f}s")

    print("\n[Prepare] [3/7] Dang vector hoa...")
    start_time = time.time()
    vectors = model.encode(codes, show_progress_bar=True, batch_size=64)
    vectors = np.asarray(vectors, dtype=np.float32)
    print(f"[Prepare] Kich thuoc vector: {vectors.shape}")
    print(f"[Prepare] Thoi gian encode: {time.time() - start_time:.2f}s")

    print("\n[Prepare] [4/7] Chuan hoa vector (L2)...")
    faiss.normalize_L2(vectors)

    print("\n[Prepare] [5/7] Luu du lieu dau ra...")
    np.save(f"{save_dir}/embeddings_20k.npy", vectors)
    with open(f"{save_dir}/v_engram_dataset_20k.json", "w", encoding="utf-8") as f:
        json.dump(list(codes), f, ensure_ascii=False)

    print("\n[Prepare] [6/7] Xay dung FAISS baseline...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(vectors, dtype=np.float32))
    print(f"[Prepare] Da index {index.ntotal:,} vectors")

    default_queries = [
        "def get_logger(name):",
        "def _merge_dicts(self, dict1, dict2, path=[]):         \"merges dict2 into dict1\"",
        "def max(self):         \"\"\" -> #float :func:numpy.max of the timing intervals \"\"\"",
        "def _strip_text(text):         \"\"\"Returns text with spaces and inserts removed.\"",
        "def unique_list(seq):     \"\"\" Removes duplicate elements from given @seq",
        "def _radixPass(a, b, r, n, K):     \"\"\"     Stable sort of the sequence a accordi",
        "def is_email():     \"\"\"     Validates that a fields value is a valid email addre",
        "def flatten_list(nested_list):",
        "def parse_yaml_config(file_path):",
        "def get_manifest_from_meta(metaurl, name):",
        "def get_value_at_percentile(self, percentile):",
        "def delete_async(blob_key, **options):",
        "def _insert_missing_rows(self, indexes):",
        "def set_log_level_format(level, format):",
        "def pack_pointer(self, name):",
    ]

    test_queries = load_queries_from_baseline(save_dir, default_queries)

    ground_truth_db = []
    for query_idx, query_text in enumerate(test_queries, 1):
        query_vector = model.encode([query_text])
        query_vector = np.asarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)
        k = 5
        distances, indices = index.search(np.asarray(query_vector, dtype=np.float32), k)
        top_results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            snippet = codes[idx].replace("\n", " ")[:80]
            top_results.append(
                {
                    "rank": rank,
                    "index": int(idx),
                    "cosine_similarity": float(dist),
                    "code_snippet": snippet,
                }
            )
        ground_truth_db.append(
            {"query_id": query_idx, "query_text": query_text, "top_5_results": top_results}
        )

    print("\n[Prepare] [7/7] Luu ground truth + metadata...")
    with open(f"{save_dir}/faiss_absolute_baseline.json", "w", encoding="utf-8") as f:
        json.dump(ground_truth_db, f, indent=2, ensure_ascii=False)

    metadata = {
        "stage": "1",
        "dataset": "CodeSearchNet Python",
        "num_vectors": len(vectors),
        "vector_dimension": dimension,
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "index_type": "IndexFlatIP (Cosine)",
        "queries_tested": len(test_queries),
        "top_k": 5,
    }

    with open(f"{save_dir}/pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Prepare] Hoan tat. File da luu:")
    print(f"[Prepare] - {save_dir}/embeddings_20k.npy")
    print(f"[Prepare] - {save_dir}/v_engram_dataset_20k.json")
    print(f"[Prepare] - {save_dir}/faiss_absolute_baseline.json")
    print(f"[Prepare] - {save_dir}/pipeline_metadata.json")


if __name__ == "__main__":
    build_ground_truth(save_dir="./data")
