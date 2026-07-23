import simpy
import random
from src.network import (
    bootstrap_network,
    data_ingestion_process,
    PLACEMENT_CANDIDATES,
    METADATA_ANCHORS,
    K_QUERY,
)
from src.routing import (
    generate_placement_key,
    generate_multi_semantic_keys,
    iterative_find_k_closest_nodes,
)
import numpy as np

import os as _os
# Tham số hoá qua biến môi trường để script chạy nhiều cấu hình:
#   DATASET=scifact NUM_FILES=5183 python3 main_churn_test.py
DATASET = _os.environ.get("DATASET", "code")          # code | scifact | squad
EMBEDDINGS_PATH = f"./data/{DATASET}_corpus_embeddings.npy"
PQ_CODES_PATH = f"./data/{DATASET}_pq_codes.npy"

NUM_NODES = int(_os.environ.get("NUM_NODES", "10000"))
NUM_FILES = int(_os.environ.get("NUM_FILES", "20000"))  # code=20000, scifact=5183
SHARDS_PER_FILE = 30
K_SIZE = 20
K_REQUIRED = 20
RECOVERY_SAMPLE = int(_os.environ.get("RECOVERY_SAMPLE", "500"))
CHURN_STEPS = [float(x) for x in
               _os.environ.get("CHURN_STEPS", "0,0.10,0.20,0.30").split(",")]
SAMPLE_SEED = int(_os.environ.get("SAMPLE_SEED", "2026"))


def kill_random_nodes(network_nodes, target_ratio, total_nodes):
    desired_remaining = int(total_nodes * (1 - target_ratio))
    kill_count = max(0, len(network_nodes) - desired_remaining)
    if kill_count == 0:
        return 0

    killed = set(random.sample(network_nodes, min(kill_count, len(network_nodes))))
    network_nodes[:] = [node for node in network_nodes if node not in killed]
    for node in network_nodes:
        node.routing_table.difference_update(killed)
    return len(killed)


def metadata_alive(vector, tag, network_nodes):
    """Metadata của object còn tìm được không, sau khi node chết?

    Bản cũ tra GLOBAL_METADATA_DHT (dict toàn cục) => metadata KHÔNG BAO GIỜ chết,
    nên thí nghiệm chỉ đo ngưỡng erasure code chứ không đo discovery. Nay client
    phải ĐỊNH TUYẾN THẬT tới các anchor còn sống, đúng như lúc query.

    Với r nhỏ (r=1 => 5 anchor/object trên toàn mạng), đây là phép đo có ý nghĩa:
    mất hết anchor là mất khả năng tìm thấy object dù payload còn nguyên.
    """
    if not network_nodes:
        return False
    for s_key in generate_multi_semantic_keys(vector):
        bootstrap_node = random.choice(network_nodes)
        anchors, _, _ = iterative_find_k_closest_nodes(
            s_key, bootstrap_node, alpha=3, k=K_QUERY
        )
        for node in anchors:
            if tag in node.RAM_Index:
                return True          # còn ít nhất 1 anchor sống ở bảng này
    return False


def can_recover_payload(tag, network_nodes):
    """Payload đặt 1 lần: tái tạo 30 placement key CHỈ từ tag, cần >=20/30 shard sống sót."""
    shards_collected = 0
    for s_id in range(SHARDS_PER_FILE):
        if not network_nodes:
            break
        p_key = generate_placement_key(tag, s_id)
        bootstrap_node = random.choice(network_nodes)
        candidate_nodes, _, _ = iterative_find_k_closest_nodes(
            p_key, bootstrap_node, alpha=3, k=PLACEMENT_CANDIDATES
        )
        for target_node in candidate_nodes:
            if f"{tag}_shard_{s_id}" in target_node.SSD_Storage:
                shards_collected += 1
                break
        if shards_collected >= K_REQUIRED:
            break

    return shards_collected >= K_REQUIRED


def count_recovered_files(network_nodes, doc_ids, vectors):
    """Đo TÁCH BẠCH 3 tầng, vì chúng hỏng vì lý do khác nhau:
      - metadata availability: routing còn tìm được discovery record không (phụ thuộc r)
      - payload decode      : còn >=20/30 shard không (phụ thuộc erasure code)
      - end-to-end          : cần CẢ HAI
    Bản cũ gộp tất cả và bỏ qua tầng metadata, nên chỉ đo được erasure code.
    """
    meta_ok = payload_ok = e2e_ok = 0
    for doc_id in doc_ids:
        tag = f"doc_{doc_id}"
        m = metadata_alive(vectors[doc_id], tag, network_nodes)
        p = can_recover_payload(tag, network_nodes)
        meta_ok += m
        payload_ok += p
        e2e_ok += (m and p)
    return meta_ok, payload_ok, e2e_ok


def run_churn_test(env):
    print("[*] Khoi tao mang P2P...")
    network_nodes = yield env.process(bootstrap_network(env, NUM_NODES, K_SIZE))

    print("[*] Dang phan bo du lieu...")
    yield env.process(data_ingestion_process(
        env, network_nodes, NUM_FILES, SHARDS_PER_FILE,
        embeddings_path=EMBEDDINGS_PATH, pq_codes_path=PQ_CODES_PATH,
        data_label=DATASET.upper()))

    vectors = np.load(EMBEDDINGS_PATH)

    random.seed(SAMPLE_SEED)
    doc_ids = random.sample(range(NUM_FILES), min(RECOVERY_SAMPLE, NUM_FILES))

    print("\n" + "=" * 70)
    print(f"THI NGHIEM: NODE FAILURE TINH (r={METADATA_ANCHORS}, {len(doc_ids)} object mau)")
    print("LUU Y: day KHONG phai churn. Node bi tat truoc khi query va khong quay lai;")
    print("       khong co bucket refresh, khong re-anchor, khong shard repair.")
    print("=" * 70)
    print(f"{'Node loss':>10s} {'Metadata':>12s} {'Payload':>12s} {'End-to-end':>12s}")

    total_nodes = len(network_nodes)
    n = len(doc_ids)
    for churn_ratio in CHURN_STEPS:
        kill_random_nodes(network_nodes, churn_ratio, total_nodes)
        meta_ok, payload_ok, e2e_ok = count_recovered_files(network_nodes, doc_ids, vectors)
        print(f"{churn_ratio:>9.0%} {100*meta_ok/n:>11.1f}% {100*payload_ok/n:>11.1f}% "
              f"{100*e2e_ok/n:>11.1f}%")


if __name__ == "__main__":
    env = simpy.Environment()
    env.process(run_churn_test(env))
    env.run()