import random
import time
import numpy as np
from src.node import VEngramNode
from src.routing import (
    generate_multi_semantic_keys, 
    generate_placement_key, 
    find_closest_node, 
    find_k_closest_nodes
)

# [MỚI] BẢNG BĂM HAI TẦNG - LƯU DANH BẠ METADATA CỦA TOÀN MẠNG
GLOBAL_METADATA_DHT = {}

# [MỚI] So luong ung vien dong bo giua luu va doc
PLACEMENT_CANDIDATES = 40

# [MỚI] Gioi han shard tren moi node
MAX_SHARDS_PER_NODE = 2500

def bootstrap_network(env, num_nodes, k_size):
    start_time = time.time()
    print(f"[*] Đang sinh ra {num_nodes:,} Nodes...")
    network_nodes = [VEngramNode(env, random.getrandbits(160)) for _ in range(num_nodes)]
    for node in network_nodes:
        targets = random.sample(network_nodes, k_size)
        for target in targets:
            if target.node_id != node.node_id:
                env.process(node.ping(target))
    yield env.timeout(0)
    print(f"✓ Mạng lưới sẵn sàng ({time.time() - start_time:.2f}s)")
    return network_nodes

def data_ingestion_process(env, network_nodes, num_files, shards_per_file):
    print("\n" + "=" * 60)
    print("GIAI ĐOẠN 2: PHÂN BỔ DỮ LIỆU (8-BIT + LOAD BALANCING + ANTI-AFFINITY)")
    print("=" * 60)

    vectors = np.load("./data/embeddings_20k.npy")
    pq_codes = np.load("./data/pq_codes.npy")  # Load bản thu gọn uint8
    total_shards = 0

    for i in range(num_files):
        vector = vectors[i]
        pq_code = pq_codes[i]  # 256 Bytes
        tag = f"doc_{i}"

        s_keys = generate_multi_semantic_keys(vector)

        # [MỚI] Lưu Sổ đỏ (Danh bạ) vào DHT
        GLOBAL_METADATA_DHT[tag] = s_keys

        # [MỚI] Anti-affinity: Node nao da cam shard cua doc nay thi tranh
        nodes_used_for_this_doc = set()

        for idx, s_key in enumerate(s_keys):
            for s_id in range(shards_per_file):
                p_key = generate_placement_key(s_key, tag, s_id)

                # Optimistic routing: thu node gan nhat truoc, fail moi fallback k ung vien
                primary = find_closest_node(p_key, network_nodes)
                current_load = len(primary.SSD_Storage)

                if current_load < MAX_SHARDS_PER_NODE and primary.node_id not in nodes_used_for_this_doc:
                    payload = {"shard_id": f"{idx}_{s_id}", "is_aes": True}
                    primary.store_shard(tag, s_id, pq_code, payload)
                    total_shards += 1
                    nodes_used_for_this_doc.add(primary.node_id)
                    continue

                # Fallback: chi khi can moi lay k ung vien
                candidates = find_k_closest_nodes(p_key, network_nodes, k=PLACEMENT_CANDIDATES)

                for target in candidates:
                    current_load = len(target.SSD_Storage)

                    if current_load < MAX_SHARDS_PER_NODE and target.node_id not in nodes_used_for_this_doc:
                        payload = {"shard_id": f"{idx}_{s_id}", "is_aes": True}
                        target.store_shard(tag, s_id, pq_code, payload)
                        total_shards += 1
                        nodes_used_for_this_doc.add(target.node_id)
                        break

        yield env.timeout(random.uniform(10, 30))
        if (i + 1) % 5000 == 0:
            print(f"  ... Đã phân bổ {i + 1:,}/{num_files:,} files.")

def query_pipeline_process(env, network_nodes, query_vector, codebook, target_k=5):
    print("\n" + "=" * 60)
    print("GIAI ĐOẠN 3: RIPPLE SEARCH BẰNG PQ (KHÔNG DÙNG ORACLE)")
    print("=" * 60)
    
    q_s_keys = generate_multi_semantic_keys(query_vector)
    all_candidates = []
    nodes_contacted = 0
    
    for idx, s_key in enumerate(q_s_keys):
        search_radius_nodes = find_k_closest_nodes(s_key, network_nodes, k=150)
        for node in search_radius_nodes:
            nodes_contacted += 1
            yield env.timeout(random.uniform(5, 15))
            
            # Tính bằng Codebook chuẩn PQ
            local_candidates = node.adc_search(query_vector, codebook, top_k=10)
            all_candidates.extend(local_candidates)
            
            if len(all_candidates) >= target_k * 40: 
                break

    print("\n" + "=" * 60)
    print("GIAI ĐOẠN 4: KHÔI PHỤC THEO KIẾN TRÚC TWO-TIER DHT")
    print("=" * 60)
    
    unique_candidates = {}
    for tag, score in all_candidates:
        if tag not in unique_candidates or score < unique_candidates[tag]:
            unique_candidates[tag] = score
    reranked_top = sorted(unique_candidates.items(), key=lambda x: x[1])[:target_k]
    
    k_required = 20  
    for rank, (tag, score) in enumerate(reranked_top, 1):
        shards_collected = 0
        
        # [MỚI] Tra sổ đỏ từ DHT Network, KHÔNG ngó vào Vector
        actual_doc_keys = GLOBAL_METADATA_DHT.get(tag)
        if not actual_doc_keys:
            continue
            
        main_doc_key = actual_doc_keys[0]
        
        for s_id in range(30):
            # Client tự "múa" ra toạ độ và phi thẳng đến đích
            p_key = generate_placement_key(main_doc_key, tag, s_id)

            # [MỚI] Đồng bộ tìm ứng viên y hệt lúc lưu
            candidate_nodes = find_k_closest_nodes(p_key, network_nodes, k=PLACEMENT_CANDIDATES)

            for target_node in candidate_nodes:
                yield env.timeout(random.uniform(2, 5))
                shard_data = yield env.process(target_node.get_shard(f"{tag}_shard_{s_id}"))
                if shard_data:
                    shards_collected += 1
                    break

            if shards_collected >= k_required:
                break
                
        status = "THÀNH CÔNG" if shards_collected >= k_required else "THẤT BẠI"
        print(f"  - Top {rank} ({tag}): Khôi phục {status}! ({shards_collected}/30 Shards)")

    retrieved_tags = [tag for tag, score in reranked_top]
    return retrieved_tags, nodes_contacted