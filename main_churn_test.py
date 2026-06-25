import simpy
import random
from src.network import (
    bootstrap_network,
    data_ingestion_process,
    GLOBAL_METADATA_DHT,
    PLACEMENT_CANDIDATES,
)
from src.routing import generate_placement_key, iterative_find_k_closest_nodes

NUM_NODES = 10000
NUM_FILES = 20000
SHARDS_PER_FILE = 30
K_SIZE = 20
K_REQUIRED = 20
RECOVERY_SAMPLE = 500
CHURN_STEPS = [0.10, 0.20, 0.30]
SAMPLE_SEED = 2026


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


def can_recover_file(tag, network_nodes):
    if not GLOBAL_METADATA_DHT.get(tag):
        return False

    # Payload đặt 1 lần: tái tạo 30 placement key CHỈ từ tag, cần >=20/30 shard sống sót.
    shards_collected = 0
    for s_id in range(SHARDS_PER_FILE):
        if not network_nodes:
            break
        p_key = generate_placement_key(tag, s_id)
        bootstrap_node = random.choice(network_nodes)
        candidate_nodes, _ = iterative_find_k_closest_nodes(
            p_key, bootstrap_node, alpha=3, k=PLACEMENT_CANDIDATES
        )
        for target_node in candidate_nodes:
            if f"{tag}_shard_{s_id}" in target_node.SSD_Storage:
                shards_collected += 1
                break
        if shards_collected >= K_REQUIRED:
            break

    return shards_collected >= K_REQUIRED


def count_recovered_files(network_nodes, doc_ids):
    recovered = 0
    for doc_id in doc_ids:
        tag = f"doc_{doc_id}"
        if can_recover_file(tag, network_nodes):
            recovered += 1
    return recovered


def run_churn_test(env):
    print("[*] Khoi tao mang P2P...")
    network_nodes = yield env.process(bootstrap_network(env, NUM_NODES, K_SIZE))

    print("[*] Dang phan bo du lieu...")
    yield env.process(data_ingestion_process(env, network_nodes, NUM_FILES, SHARDS_PER_FILE))

    random.seed(SAMPLE_SEED)
    doc_ids = random.sample(range(NUM_FILES), RECOVERY_SAMPLE)

    total_nodes = len(network_nodes)
    for churn_ratio in CHURN_STEPS:
        killed = kill_random_nodes(network_nodes, churn_ratio, total_nodes)
        recovered = count_recovered_files(network_nodes, doc_ids)
        print(
            "[Churn] Mat {ratio:.0%} | Kill {killed:,} nodes | Recovered {rec}/{total}"
            .format(ratio=churn_ratio, killed=killed, rec=recovered, total=len(doc_ids))
        )


if __name__ == "__main__":
    env = simpy.Environment()
    env.process(run_churn_test(env))
    env.run()
