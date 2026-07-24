import random
import time
import numpy as np
from src.node import VEngramNode
from src.routing import (
    generate_multi_semantic_keys,
    generate_probe_keys,
    generate_placement_key,
    iterative_find_k_closest_nodes,
    NUM_PROJECTIONS,
    DEFAULT_ALPHA,
    DEFAULT_R_MAX,
    DEFAULT_MULTI_PROBE,
    DEFAULT_PROBE_BITS,
)

# [MỚI] BẢNG BĂM HAI TẦNG - LƯU DANH BẠ METADATA CỦA TOÀN MẠNG
GLOBAL_METADATA_DHT = {}

def reset_global_metadata_dht():
    GLOBAL_METADATA_DHT.clear()

# [MỚI] So luong ung vien dong bo giua luu va doc (TẦNG PAYLOAD)
PLACEMENT_CANDIDATES = 300

def _rtt():
    """RTT mỗi message ~ N(50ms, sigma=15ms), khớp mục 4.1 trong paper.
    Bản cũ dùng uniform(5,15)/(2,5)/(10,30) — lệch với con số đã in trong paper."""
    return max(1.0, random.normalvariate(50, 15))

# ============================================================================
# THAM SỐ GIAO THỨC — khớp Table 2 trong paper
# ============================================================================
# r — số node neo metadata quanh MỖI semantic key (tầng METADATA).
# Mỗi object có L*r bản metadata.
#
# QUAN TRỌNG: r KHÔNG phải tham số độ bền tự do. Nó quyết định semantic key có
# giá trị hay không. Mỗi object phủ L*r / N mạng; khi tỉ lệ này đủ lớn, một client
# chạm cùng số node NGẪU NHIÊN cũng giao với anchor set, và semantic routing không
# còn đóng góp gì (mục 3.6 + 4.x ngưỡng r*).
# Số đo: r=1 -> semantic 43.8% vs random 3.6% (thắng 12.2x)
#        r=30 -> semantic 48.2% vs random 73.8% (random THẮNG!)
import os as _os
# Mục 16: quét r ∈ {1,2,3} qua biến môi trường thay vì sửa code
METADATA_ANCHORS = int(_os.environ.get("META_ANCHORS", "1"))
# Fetch payload song song (mặc định BẬT). PARALLEL_FETCH=0 để đối chiếu.
PARALLEL_FETCH = _os.environ.get("PARALLEL_FETCH", "1") != "0"

# K — ngân sách node mỗi bảng (số node chạy ADC cho mỗi prefix)
K_QUERY = 20

# T — số prefix probe mỗi bảng (multi-probe, mục 3.5)
MULTI_PROBE = DEFAULT_MULTI_PROBE

# kappa — số tag mỗi node trả về
LOCAL_TOP_K = 30

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
    embeddings_path="./data/code_corpus_embeddings.npy",
    pq_codes_path="./data/code_pq_codes.npy",
    data_label="CODE",
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
            anchors, _, _ = iterative_find_k_closest_nodes(
                s_key, bootstrap_node, alpha=DEFAULT_ALPHA, k=METADATA_ANCHORS
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
            candidates, _, _ = iterative_find_k_closest_nodes(
                p_key, bootstrap_node, alpha=DEFAULT_ALPHA, k=PLACEMENT_CANDIDATES
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

        yield env.timeout(_rtt())
        if (i + 1) % 5000 == 0:
            print(f"  ... Đã phân bổ {i + 1:,}/{num_files:,} files.")

    print(
        f"✓ Hoàn tất: {total_shards:,} payload shard (đặt 1 lần) "
        f"| {total_anchors:,} bản sao metadata (nhân L lần)."
    )


def _fetch_one_shard(env, tag, s_id, network_nodes, acc):
    """Lấy MỘT shard. Viết thành generator riêng để 30 shard đi SONG SONG.

    Ghi chi phí vào dict acc dùng chung (rounds/rpcs/bytes/ok).
    """
    p_key = generate_placement_key(tag, s_id)
    bootstrap_node = random.choice(network_nodes)
    candidate_nodes, ph, pr = iterative_find_k_closest_nodes(
        p_key, bootstrap_node, alpha=DEFAULT_ALPHA, k=PLACEMENT_CANDIDATES
    )
    acc["rounds"] += ph
    acc["rpcs"] += pr
    acc["bytes"] += pr * 8 * 20
    for target_node in candidate_nodes:
        yield env.timeout(_rtt())
        shard_data = yield env.process(target_node.get_shard(f"{tag}_shard_{s_id}"))
        acc["rpcs"] += 1
        if shard_data:
            acc["ok"] += 1
            acc["bytes"] += 4096 // 20
            return


def _fetch_one_object(env, tag, network_nodes, acc, k_required=20):
    """Lấy payload của MỘT object: phóng k_required shard song song, thiếu thì bù.

    Viết riêng để bản thân các object cũng chạy song song với nhau — client thật
    không đợi giải mã xong object 1 rồi mới đi tìm object 2.
    """
    procs = [env.process(_fetch_one_shard(env, tag, s_id, network_nodes, acc))
             for s_id in range(k_required)]
    yield env.all_of(procs)
    if acc["ok"] < k_required:            # có shard chết -> bù nốt, vẫn song song
        procs = [env.process(_fetch_one_shard(env, tag, s_id, network_nodes, acc))
                 for s_id in range(k_required, 30)]
        yield env.all_of(procs)


def query_pipeline_process(env, network_nodes, query_vector, codebook, target_k=5,
                           k_query=None, multi_probe=None, random_routing=False,
                           verbose=True):
    """Đường đọc: Ripple Search đa bảng + multi-probe -> ADC -> merge -> fetch payload.

    Sửa so với bản cũ:
      1. BUG NGÂN SÁCH: bản cũ `if len(all_candidates) >= target_k*80: break` cộng dồn
         qua CẢ 5 bảng, nên bảng 1 ghé ~14 node còn bảng 2-5 mỗi bảng ghé ĐÚNG 1 node.
         Paper mô tả 5 bảng đối xứng. Nay ngân sách áp theo TỪNG bảng (k_query).
      2. MULTI-PROBE (mục 3.5): mỗi bảng probe T prefix thay vì 1.
      3. Đếm TÁCH BẠCH rounds / RPC / contacted nodes (mục 3.9).
      4. random_routing=True: baseline chạm CÙNG số node nhưng chọn ngẫu nhiên,
         bỏ qua semantic key — phép so duy nhất trả lời "semantic key có đáng không".
    """
    if k_query is None:
        k_query = K_QUERY
    if multi_probe is None:
        multi_probe = MULTI_PROBE

    if verbose:
        print("\n" + "=" * 60)
        print(f"GIAI ĐOẠN 3: RIPPLE SEARCH (L={NUM_PROJECTIONS}, K={k_query}, "
              f"T={multi_probe}{', RANDOM' if random_routing else ''})")
        print("=" * 60)

    all_candidates = []
    total_hops = 0          # routing rounds
    total_rpcs = 0          # RPC count
    contacted = set()       # node chạy ADC
    # --- Mục 5: tách chi phí DISCOVERY và PAYLOAD ---
    disc_rounds = disc_rpcs = disc_bytes = 0
    pay_rounds = pay_rpcs = pay_bytes = 0
    lookups_total = lookups_at_cap = 0        # mục 21: % lookup chạm R_max
    t_query_start = env.now

    if random_routing:
        # BASELINE: chạm đúng cùng số node nhưng chọn ngẫu nhiên
        n_touch = min(k_query * NUM_PROJECTIONS * multi_probe, len(network_nodes))
        for node in random.sample(network_nodes, n_touch):
            contacted.add(node.node_id)
            yield env.timeout(_rtt())
            all_candidates.extend(node.adc_search(query_vector, codebook, top_k=LOCAL_TOP_K))
    else:
        for t in range(NUM_PROJECTIONS):
            # Multi-probe: T prefix cho bảng này (gốc + T-1 biến thể lật bit yếu)
            for p_key in generate_probe_keys(query_vector, t, T=multi_probe,
                                             c=DEFAULT_PROBE_BITS):
                bootstrap_node = random.choice(network_nodes)
                nodes, hops, rpcs = iterative_find_k_closest_nodes(
                    p_key, bootstrap_node, alpha=DEFAULT_ALPHA,
                    k=k_query, max_rounds=DEFAULT_R_MAX
                )
                total_hops += hops
                total_rpcs += rpcs
                disc_rounds += hops
                disc_rpcs += rpcs
                disc_bytes += rpcs * 8 * 20        # FIND_NODE trả ~8 contact × 20B
                lookups_total += 1
                if hops >= DEFAULT_R_MAX:
                    lookups_at_cap += 1            # không hội tụ, chạm trần
                # NGÂN SÁCH THEO TỪNG PREFIX — không cộng dồn qua các bảng
                for node in nodes:
                    if node.node_id in contacted:
                        continue          # node đã chạy ADC cho prefix khác
                    contacted.add(node.node_id)
                    yield env.timeout(_rtt())
                    total_rpcs += 1       # ADC request
                    disc_rpcs += 1
                    disc_bytes += 512 + LOCAL_TOP_K * 24   # query PQ gửi đi + tag trả về
                    all_candidates.extend(
                        node.adc_search(query_vector, codebook, top_k=LOCAL_TOP_K)
                    )

    # --- Merge: giữ khoảng cách nhỏ nhất cho mỗi tag, dedup giữa các bảng ---
    unique_candidates = {}
    for tag, score in all_candidates:
        if tag not in unique_candidates or score < unique_candidates[tag]:
            unique_candidates[tag] = score
    num_unique_candidates = len(unique_candidates)
    reranked_top = sorted(unique_candidates.items(), key=lambda x: x[1])[:target_k]
    retrieved_tags = [tag for tag, _ in reranked_top]

    if verbose:
        print("\n" + "=" * 60)
        print("GIAI ĐOẠN 4: KHÔI PHỤC PAYLOAD (HMAC key tái tạo từ tag)")
        print("=" * 60)

    # --- Payload: client tự tính lại toạ độ shard CHỈ từ tag (stateless) ---
    # Bỏ lối tắt tra GLOBAL_METADATA_DHT: trong thí nghiệm churn nó khiến metadata
    # KHÔNG BAO GIỜ chết, nên metadata availability không đo được (mục Threats).
    k_required = 20
    if PARALLEL_FETCH:
        # SONG SONG: phóng k_required shard cùng lúc, thiếu thì phóng tiếp phần còn
        # lại. Client thật làm đúng vậy — nó không biết shard nào chết nên gửi hết
        # rồi lấy 20 cái về trước. Độ trễ khi đó là của lookup CHẬM NHẤT, không
        # phải tổng của 20-30 lookup nối đuôi.
        # Cả 5 object ĐỒNG THỜI, và trong mỗi object thì 20 shard cũng đồng thời.
        accs = [{"rounds": 0, "rpcs": 0, "bytes": 0, "ok": 0} for _ in reranked_top]
        objs = [env.process(_fetch_one_object(env, tag, network_nodes, accs[idx],
                                              k_required))
                for idx, (tag, score) in enumerate(reranked_top)]
        if objs:
            yield env.all_of(objs)
        for a in accs:
            pay_rounds += a["rounds"]
            pay_rpcs += a["rpcs"]
            pay_bytes += a["bytes"]
        shards_collected = accs[-1]["ok"] if accs else 0
    else:
        # TUẦN TỰ: giữ để đối chiếu độ trễ.
        for rank, (tag, score) in enumerate(reranked_top, 1):
            shards_collected = 0
            for s_id in range(30):
                p_key = generate_placement_key(tag, s_id)
                bootstrap_node = random.choice(network_nodes)
                candidate_nodes, _ph, _pr = iterative_find_k_closest_nodes(
                    p_key, bootstrap_node, alpha=DEFAULT_ALPHA, k=PLACEMENT_CANDIDATES
                )
                pay_rounds += _ph
                pay_rpcs += _pr
                pay_bytes += _pr * 8 * 20
                for target_node in candidate_nodes:
                    yield env.timeout(_rtt())
                    shard_data = yield env.process(target_node.get_shard(f"{tag}_shard_{s_id}"))
                    pay_rpcs += 1
                    if shard_data:
                        shards_collected += 1
                        pay_bytes += 4096 // 20
                        break
                if shards_collected >= k_required:
                    break
        if verbose:
            status = "THÀNH CÔNG" if shards_collected >= k_required else "THẤT BẠI"
            print(f"  - Top {rank} ({tag}): Khôi phục {status}! ({shards_collected}/30 Shards)")

    stats = {
        "rounds": total_hops,
        "rpcs": total_rpcs,
        "contacted_nodes": len(contacted),
        "unique_candidates": num_unique_candidates,
        # --- Mục 5: chi phí tách bạch ---
        "disc_rounds": disc_rounds, "disc_rpcs": disc_rpcs, "disc_bytes": disc_bytes,
        "pay_rounds": pay_rounds, "pay_rpcs": pay_rpcs, "pay_bytes": pay_bytes,
        "candidate_tags": len(all_candidates),
        "latency_ms": env.now - t_query_start,
        # --- Mục 21: chạm trần R_max ---
        "lookups_total": lookups_total,
        "lookups_at_cap": lookups_at_cap,
        "r_max": DEFAULT_R_MAX,
    }
    return retrieved_tags, total_hops, num_unique_candidates, stats