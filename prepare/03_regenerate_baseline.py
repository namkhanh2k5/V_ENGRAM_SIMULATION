import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


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


def fix_query_baseline(save_dir):
    print("[Prepare] Khoi dong tai vector va chay FAISS cuc bo...")

    with open(f"{save_dir}/v_engram_dataset_20k.json", "r", encoding="utf-8") as f:
        codes = json.load(f)

    vectors = np.load(f"{save_dir}/embeddings_20k.npy")
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

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
        distances, indices = index.search(query_vector, k)
        top_results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            idx_int = int(idx)
            snippet = codes[idx_int].replace("\n", " ")[:80]
            top_results.append(
                {
                    "rank": rank,
                    "index": idx_int,
                    "cosine_similarity": float(dist),
                    "code_snippet": snippet,
                }
            )
        ground_truth_db.append(
            {"query_id": query_idx, "query_text": query_text, "top_5_results": top_results}
        )

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

    print("[Prepare] Da cap nhat baseline va metadata.")


if __name__ == "__main__":
    fix_query_baseline(save_dir="./data")
