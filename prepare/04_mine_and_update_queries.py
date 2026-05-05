import json
import random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_queries_from_baseline(save_dir):
    baseline_path = f"{save_dir}/faiss_absolute_baseline.json"
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        return [item["query_text"] for item in baseline if "query_text" in item]
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def mine_and_overwrite_baseline(codes, vectors, save_dir):
    print("[Prepare] [1/2] Dao query moi (cosine > 0.88)...")

    old_queries = load_queries_from_baseline(save_dir)
    if old_queries:
        print(f"[Prepare] Da tai {len(old_queries)} query tu baseline.")
    else:
        old_queries = [
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

    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.asarray(vectors, dtype=np.float32))

    new_queries = []
    attempts = 0

    max_new_needed = max(0, 15 - len(old_queries))
    if max_new_needed == 0:
        print("[Prepare] Da co du 15 query. Bo qua dao them.")
    while len(new_queries) < max_new_needed and attempts < 3000:
        attempts += 1
        idx = random.randint(0, len(codes) - 1)
        try:
            lines = str(codes[idx]).strip().split("\n")
            first_line = lines[0].strip()[:100]

            if not first_line.startswith("def ") or first_line in old_queries:
                continue

            query_vector = model.encode([first_line])
            query_vector = np.asarray(query_vector, dtype=np.float32)
            faiss.normalize_L2(query_vector)
            distances, indices = index.search(query_vector, 1)

            if float(distances[0][0]) > 0.88:
                new_queries.append(first_line)
                print(f"[Prepare] Tim thay: {first_line[:50]}... (Score: {float(distances[0][0]):.4f})")
        except Exception:
            continue

    final_15_queries = old_queries + new_queries

    print("[Prepare] [2/2] Tinh top-5 va ghi de JSON...")

    ground_truth_db = []
    for query_idx, query_text in enumerate(final_15_queries, 1):
        query_vector = model.encode([query_text])
        query_vector = np.asarray(query_vector, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        k = 5
        distances, indices = index.search(query_vector, k)

        top_results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            idx_int = int(idx)
            if 0 <= idx_int < len(codes):
                snippet = str(codes[idx_int]).replace("\n", " ")[:80]
            else:
                snippet = f"Index {idx_int} out of range"

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

    output_file = f"{save_dir}/faiss_absolute_baseline.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth_db, f, indent=2, ensure_ascii=False)

    print("[Prepare] Da luu:")
    print(f"[Prepare] {output_file}")


def main():
    save_dir = "./data"
    with open(f"{save_dir}/v_engram_dataset_20k.json", "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    codes = []
    if isinstance(raw_dataset, list):
        for item in raw_dataset:
            if isinstance(item, str):
                codes.append(item)
            elif isinstance(item, dict):
                for key in ["content", "code", "text", "snippet", "function"]:
                    if key in item:
                        codes.append(item[key])
                        break
                else:
                    longest_str = max([str(v) for v in item.values()], key=len, default="")
                    codes.append(longest_str)

    vectors = np.load(f"{save_dir}/embeddings_20k.npy")
    vectors = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    if not codes:
        raise RuntimeError("No code extracted from dataset.")

    print(f"[Prepare] Da tai {len(codes)} doan code va {vectors.shape[0]} vector")
    mine_and_overwrite_baseline(codes, vectors, save_dir)


if __name__ == "__main__":
    main()