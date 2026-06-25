import random
import time
import numpy as np
from src.node import VEngramNode
from src.routing import (
    generate_multi_semantic_keys,
    generate_placement_key,
    iterative_find_k_closest_nodes,
)

# [MỚI] BẢNG BĂM HAI TẦNG - LƯU DANH BẠ METADATA CỦA TOÀN MẠNG
GLOBAL_METADATA_DHT = {}

def reset_global_metadata_dht():
    GLOBAL_METADATA_DHT.clear()

# [MỚI] So luong ung vien dong bo giua luu va doc (TẦNG PAYLOAD)
PLACEMENT_CANDIDATES = 300

# [MỚI] So node neo METADATA (PQ code) quanh MOI semantic key — TẦNG METADATA.
# Nhan ban L lan (mot lan moi semantic key) de discovery ben va giu Success@5.
# Khong anh huong "Mean Load (shards/node)" vi load chi dem SSD_Storage (payload).
METADATA_ANCHORS = 30

# [MỚI] Gioi han shard tren moi node
MAX_SHARDS_PER_NODE = 2500


def bootstrap_network(env, num_nodes, k_size_far=50, k_size_near=50):
    start_time = time.time()
    print(f"[*] Đang sinh ra {num_nodes:,} Nodes...")
    network_nodes = [VEngramNode(env, random.getrandbits(160)) for _ in range(num_nodes)]

    # Sort once to build a ring-style adjacency list
    network_nodes.sort(key=lambda n: n.node_id)

    print("[*] Đan cấu trúc Small-World (Ring-Adjacency: 50 xa + 50 gần)...")
    for i, node in enumerate(network_nodes):
        # A. Long-distance random links
        targets = random.sample(network_nodes, min(k_size_far, num_nodes))
        node.routing_table.update(t for t in targets if t.node_id != node.node_id)

        # B. Short-distance neighbors (left/right in sorted ring)
        half_near = k_size_near // 2
        left_neighbors = [network_nodes[(i - j) % num_nodes] for j in range(1, half_near + 1)]
        right_neighbors = [network_nodes[(i + j) % num_nodes] for j in range(1, half_near + 1)]
        node.routing_table.update(left_neighbors)
        node.routing_table.update(right_neighbors)

    yield env.timeout(0)
    print(f"✓ Mạng lưới sẵn sàng ({time.time() - start_time:.2f}s)")
    return network_nodes


def data_ingestion_process(
    env,
    network_nodes,
    num_files,
    shards_per_file,
    embeddings_path="./data/scifact_embeddings.npy", # <-- Cập nhật SciFact
    pq_codes_path="./data/scifact_pq_codes.npy",     # <-- Cập nhật SciFact
    data_label="SciFact",
):
    print("\n" + "=" * 60)
    print(
        f"GIAI ĐOẠN 2: PHÂN BỔ DỮ LIỆU {data_label} (TWO-TIER: METADATA L-replica + PAYLOAD-ONCE)"
    )
    print("=" * 60)

    vectors = np.load(embeddings_path)
    pq_codes = np.load(pq_codes_path)
    total_shards = 0       # tong so payload shard da dat (ky vong = num_files * shards_per_file)
    total_anchors = 0      # tong so ban sao metadata (ky vong ~ num_files * L * METADATA_ANCHORS)

    for i in range(num_files):
        vector = vectors[i]
        pq_code = pq_codes[i]  # 256 Bytes
        tag = f"doc_{i}"

        s_keys = generate_multi_semantic_keys(vector)

        # Sổ đỏ (Danh bạ) tag -> L semantic keys
        GLOBAL_METADATA_DHT[tag] = s_keys

        # ============================================================
        # TẦNG 1 — METADATA: neo PQ code tại L semantic key (nhân L lần)
        # Day la be mat discovery: query dinh tuyen toi vung ngu nghia se thay PQ code.
        # ============================================================
        for s_key in s_keys:
            bootstrap_node = random.choice(network_nodes)
            anchors, _ = iterative_find_k_closest_nodes(
                s_key, bootstrap_node, alpha=3, k=METADATA_ANCHORS
            )
            for anchor in anchors[:METADATA_ANCHORS]:
                anchor.store_metadata(tag, pq_code)
                total_anchors += 1

        # ============================================================
        # TẦNG 2 — PAYLOAD: đặt 30 shard MỘT lần ở K_place(s)=HMAC(tag,s)
        # Doc lap ngu nghia -> rai DEU (het semantic hotspot o payload).
        # ============================================================
        nodes_used_for_this_doc = set()
        for s_id in range(shards_per_file):
            p_key = generate_placement_key(tag, s_id)

            bootstrap_node = random.choice(network_nodes)
            candidates, _ = iterative_find_k_closest_nodes(
                p_key, bootstrap_node, alpha=3, k=PLACEMENT_CANDIDATES
            )
            if not candidates:
                candidates = [bootstrap_node]

            # Dat tren node gan nhat con cho & chua giu shard khac cua doc nay (anti-affinity)
            for target in candidates:
                if (
                    len(target.SSD_Storage) < MAX_SHARDS_PER_NODE
                    and target.node_id not in nodes_used_for_this_doc
                ):
                    target.store_payload_shard(tag, s_id, {"shard_id": f"{s_id}", "is_aes": True})
                    total_shards += 1
                    nodes_used_for_this_doc.add(target.node_id)
                    break

        yield env.timeout(random.uniform(10, 30))
        if (i + 1) % 5000 == 0:
            print(f"  ... Đã phân bổ {i + 1:,}/{num_files:,} files.")

    print(
        f"✓ Hoàn tất: {total_shards:,} payload shard (đặt 1 lần) "
        f"| {total_anchors:,} bản sao metadata (nhân L lần)."
    )


def query_pipeline_process(env, network_nodes, query_vector, codebook, target_k=5):
    print("\n" + "=" * 60)
    print("GIAI ĐOẠN 3: RIPPLE SEARCH BẰNG PQ (KHÔNG DÙNG ORACLE)")
    print("=" * 60)

    q_s_keys = generate_multi_semantic_keys(query_vector)
    all_candidates = []
    total_hops = 0

    for idx, s_key in enumerate(q_s_keys):
        bootstrap_node = random.choice(network_nodes)
        search_radius_nodes, hops = iterative_find_k_closest_nodes(
            s_key, bootstrap_node, alpha=3, k=300
        )
        total_hops += hops
        for node in search_radius_nodes:
            yield env.timeout(random.uniform(5, 15))

            # Tính bằng Codebook chuẩn PQ
            local_candidates = node.adc_search(query_vector, codebook, top_k=30)
            all_candidates.extend(local_candidates)

            if len(all_candidates) >= target_k * 80:
                break

    print("\n" + "=" * 60)
    print("GIAI ĐOẠN 4: KHÔI PHỤC THEO KIẾN TRÚC TWO-TIER DHT")
    print("=" * 60)

    unique_candidates = {}
    for tag, score in all_candidates:
        if tag not in unique_candidates or score < unique_candidates[tag]:
            unique_candidates[tag] = score
    # [MỚI] So object UNIQUE thuc su duoc rerank cho query nay (phep do Q1:
    # bac bo gia thuyet "recall den tu rerank rong" — bao nhieu % corpus duoc xet)
    num_unique_candidates = len(unique_candidates)
    reranked_top = sorted(unique_candidates.items(), key=lambda x: x[1])[:target_k]

    # --- KHÔI PHỤC: payload lấy MỘT lần từ HMAC key tái tạo từ tag ---
    k_required = 20
    for rank, (tag, score) in enumerate(reranked_top, 1):
        # Tra sổ đỏ chỉ để xác nhận object tồn tại (placement KHÔNG cần semantic key)
        if not GLOBAL_METADATA_DHT.get(tag):
            continue

        shards_collected = 0
        for s_id in range(30):
            # Client tự tái tạo toạ độ shard CHỈ từ tag
            p_key = generate_placement_key(tag, s_id)
            bootstrap_node = random.choice(network_nodes)
            candidate_nodes, _ = iterative_find_k_closest_nodes(
                p_key, bootstrap_node, alpha=3, k=PLACEMENT_CANDIDATES
            )
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
    return retrieved_tags, total_hops, num_unique_candidates