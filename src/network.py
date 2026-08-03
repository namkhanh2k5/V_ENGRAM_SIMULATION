import random
from typing import Any, Dict, List
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
# Bỏ HOÀN TOÀN tầng payload (ingest + fetch). Chỉ dùng cho thí nghiệm chỉ cần
# Recall@5, vốn quyết định ở tầng discovery. Ingest tốn 5 lookup cho metadata
# nhưng 30 cho payload shard, nên bỏ payload nhanh gấp ~7 lần.
# KHÔNG dùng khi đo chi phí hay độ bền payload — lúc đó payload là trọng tâm.
SKIP_PAYLOAD = _os.environ.get("SKIP_PAYLOAD", "0") == "1"
# Chế độ định tuyến cho baseline. 'auto' = theo cờ --random-routing (tương thích cũ).
#   semantic      : dùng semantic key
#   random_slots  : oracle, bốc L*K*T node, KHÔNG định tuyến
#   random_unique : oracle, bốc đúng MATCH_UNIQUE_NODES node
#   keyed_lookup  : L*T khoá ngẫu nhiên + lookup Kademlia thật (THỰC THI ĐƯỢC)
ROUTING_MODE = _os.environ.get("ROUTING_MODE", "auto")

# --- MC10: chế độ đặt payload shard ---
#   scan (mặc định, hành vi cũ): đi dọc danh sách ứng viên, đặt vào node đầu
#     tiên chưa giữ shard khác của cùng doc (anti-affinity). Vì thứ tự danh sách
#     lúc GHI và lúc ĐỌC khác nhau (lookup xấp xỉ, bootstrap ngẫu nhiên), client
#     đọc không biết node nào giữ shard, nên phải xin PLACEMENT_CANDIDATES=300
#     ứng viên rồi dò dần. Đó là nguồn của 89% RPC.
#   deterministic: bỏ anti-affinity, luôn đặt vào node GẦN NHẤT với placement key.
#     Đọc chỉ cần xin ít ứng viên vì node gần nhất là tất định theo khoá.
#     Đánh đổi: ~4% object có hai shard cùng node (C(30,2)/N), node đó chết thì
#     mất 2 shard thay vì 1. RS(30,20) chịu được 10 nên vẫn dư biên.
PLACEMENT_MODE = _os.environ.get("PLACEMENT_MODE", "scan")
# Gửi ADC request song song tới các node đã chọn (mặc định BẬT).
# PARALLEL_ADC=0 để đối chiếu với hành vi tuần tự cũ.
PARALLEL_ADC = _os.environ.get("PARALLEL_ADC", "1") != "0"
# Số ứng viên xin mỗi lần lookup placement key. Ở chế độ deterministic có thể
# hạ mạnh vì không phải dò.
PLACEMENT_K = int(_os.environ.get("PLACEMENT_K", str(PLACEMENT_CANDIDATES)))
# Số object fetch payload. Mặc định lấy hết top-k trả về; đặt 1 để chỉ lấy cái đầu.
FETCH_TOP = int(_os.environ.get("FETCH_TOP", "0"))   # 0 = lấy hết

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
        for s_id in ([] if SKIP_PAYLOAD else range(shards_per_file)):
            p_key = generate_placement_key(tag, s_id)

            bootstrap_node = random.choice(network_nodes)
            candidates, _, _ = iterative_find_k_closest_nodes(
                p_key, bootstrap_node, alpha=DEFAULT_ALPHA, k=PLACEMENT_K
            )
            if not candidates:
                candidates = [bootstrap_node]

            if PLACEMENT_MODE == "deterministic":
                # Luôn node GẦN NHẤT, không anti-affinity. Vị trí trở thành hàm
                # của khoá, nên client đọc tính lại được mà không phải dò.
                target = candidates[0]
                if len(target.SSD_Storage) < MAX_SHARDS_PER_NODE:
                    target.store_payload_shard(tag, s_id, {"shard_id": f"{s_id}", "is_aes": True})
                    total_shards += 1
                    nodes_used_for_this_doc.add(target.node_id)
            else:
                # scan: node gần nhất còn chỗ VÀ chưa giữ shard khác của doc này
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


def _fetch_one_shard(env, tag, s_id, network_nodes, acc: Dict[str, Any]):
    """Lấy MỘT shard. Viết thành generator riêng để 30 shard đi SONG SONG.

    Ghi chi phí vào dict acc dùng chung (rounds/rpcs/bytes/ok).
    """
    p_key = generate_placement_key(tag, s_id)
    bootstrap_node = random.choice(network_nodes)
    candidate_nodes, ph, pr = iterative_find_k_closest_nodes(
        p_key, bootstrap_node, alpha=DEFAULT_ALPHA, k=PLACEMENT_K
    )
    acc["rounds"] += ph
    acc["rpcs"] += pr
    acc["bytes"] += pr * 8 * 20
    for depth, target_node in enumerate(candidate_nodes, 1):
        yield env.timeout(_rtt())
        shard_data = yield env.process(target_node.get_shard(f"{tag}_shard_{s_id}"))
        acc["rpcs"] += 1
        if shard_data:
            acc["ok"] += 1
            acc["bytes"] += 4096 // 20
            # MC10: dò tới ứng viên thứ mấy mới thấy. =1 nghĩa là tất định.
            acc.setdefault("depths", []).append(depth)
            return
    acc.setdefault("misses", 0)
    acc["misses"] += 1


def _fetch_one_object(env, tag, network_nodes, acc: Dict[str, Any], k_required=20):
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


def _adc_seq(env, nodes, query_vector, codebook, acc):
    """Bản TUẦN TỰ, giữ để đối chiếu. Đặt PARALLEL_ADC=0 để dùng."""
    for node in nodes:
        yield env.timeout(_rtt())
        acc["rpcs"] += 1
        acc["bytes"] += 512 + LOCAL_TOP_K * 24
        acc["cands"].extend(node.adc_search(query_vector, codebook,
                                            top_k=LOCAL_TOP_K))


def _adc_all(env, nodes, query_vector, codebook, acc):
    """Gọi ADC trên tập node, song song hoặc tuần tự tuỳ PARALLEL_ADC."""
    if not nodes:
        return
    if PARALLEL_ADC:
        yield env.all_of([env.process(_adc_one(env, nd, query_vector, codebook, acc))
                          for nd in nodes])
    else:
        yield env.process(_adc_seq(env, nodes, query_vector, codebook, acc))


def _adc_one(env, node, query_vector, codebook, acc):
    """Gọi ADC trên MỘT node. Viết thành generator để các node chạy SONG SONG.

    VÌ SAO CẦN: bản trước duyệt tuần tự và yield env.timeout(_rtt()) từng node.
    Với 488 node phân biệt và RTT 50 ms, riêng vòng này đã là 24.400 ms — chiếm
    ~91% p50 đo được (26.800 ms). Con số latency vì thế đo CÁCH MÔ PHỎNG XẾP
    HÀNG chứ không đo giao thức: client thật gửi 488 ADC request song song, y
    như nó gửi lookup song song.

    Đường tới hạn thật: vài vòng lookup + MỘT RTT cho ADC, cỡ vài trăm ms.
    """
    yield env.timeout(_rtt())
    acc["rpcs"] += 1
    acc["bytes"] += 512 + LOCAL_TOP_K * 24
    acc["cands"].extend(node.adc_search(query_vector, codebook, top_k=LOCAL_TOP_K))


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
    probe_depths: List[int] = []              # MC10: dò tới ứng viên thứ mấy
    shard_misses = 0                          # MC10: shard không tìm thấy
    t_query_start = env.now

    _mode = ROUTING_MODE if ROUTING_MODE != 'auto' else (
        'random_slots' if random_routing else 'semantic')

    if _mode in ('random_slots', 'random_unique'):
        # BASELINE ORACLE: bốc node trực tiếp, KHÔNG định tuyến. Không có lookup
        # nào, nên chi phí RPC đúng bằng số lần gọi ADC.
        #
        # Bản trước KHÔNG đếm disc_rpcs ở nhánh này, nên đo ra 0 RPC và làm
        # phép so per-RPC vô nghĩa. Đây là chỗ khiến bài phải SUY RA thay vì đo.
        if _mode == 'random_unique':
            # khớp đúng số node PHÂN BIỆT mà semantic chạm: cần biết trước, nên
            # lấy từ env (do lần chạy semantic cùng cấu hình cung cấp)
            n_touch = int(_os.environ.get('MATCH_UNIQUE_NODES', k_query * NUM_PROJECTIONS))
        else:
            n_touch = k_query * NUM_PROJECTIONS * multi_probe
        n_touch = min(n_touch, len(network_nodes))
        _sel = random.sample(network_nodes, n_touch)
        for node in _sel:
            contacted.add(node.node_id)
        _acc = {"rpcs": 0, "bytes": 0, "cands": []}
        yield env.process(_adc_all(env, _sel, query_vector, codebook, _acc))
        total_rpcs += _acc["rpcs"]; disc_rpcs += _acc["rpcs"]
        disc_bytes += _acc["bytes"]; all_candidates.extend(_acc["cands"])

    elif _mode == 'keyed_lookup':
        # BASELINE THỰC THI ĐƯỢC: bốc L*T KHOÁ ngẫu nhiên rồi chạy đúng lookup
        # Kademlia lặp như semantic. Không cần global membership view, và trả
        # ĐÚNG chi phí routing — đây là baseline mà mục 4.12 phải suy ra chi phí,
        # giờ đo trực tiếp.
        for _ in range(NUM_PROJECTIONS * multi_probe):
            r_key = random.getrandbits(160)
            bootstrap_node = random.choice(network_nodes)
            nodes, hops, rpcs = iterative_find_k_closest_nodes(
                r_key, bootstrap_node, alpha=DEFAULT_ALPHA,
                k=k_query, max_rounds=DEFAULT_R_MAX
            )
            total_hops += hops; total_rpcs += rpcs
            disc_rounds += hops; disc_rpcs += rpcs
            disc_bytes += rpcs * 8 * 20
            lookups_total += 1
            if hops >= DEFAULT_R_MAX:
                lookups_at_cap += 1
            _new = [nd for nd in nodes if nd.node_id not in contacted]
            for nd in _new:
                contacted.add(nd.node_id)
            if _new:
                _acc = {"rpcs": 0, "bytes": 0, "cands": []}
                yield env.process(_adc_all(env, _new, query_vector, codebook, _acc))
                total_rpcs += _acc["rpcs"]; disc_rpcs += _acc["rpcs"]
                disc_bytes += _acc["bytes"]; all_candidates.extend(_acc["cands"])
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
                # NGÂN SÁCH THEO TỪNG PREFIX — không cộng dồn qua các bảng.
                # Node đã chạy ADC cho prefix khác thì bỏ qua (dedup toàn cục).
                _new = [nd for nd in nodes if nd.node_id not in contacted]
                for nd in _new:
                    contacted.add(nd.node_id)
                if _new:
                    _acc = {"rpcs": 0, "bytes": 0, "cands": []}
                    yield env.process(_adc_all(env, _new, query_vector,
                                                    codebook, _acc))
                    total_rpcs += _acc["rpcs"]; disc_rpcs += _acc["rpcs"]
                    disc_bytes += _acc["bytes"]
                    all_candidates.extend(_acc["cands"])

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
    if SKIP_PAYLOAD:
        shards_collected = 0          # bỏ hẳn tầng payload
    elif PARALLEL_FETCH:
        # SONG SONG: phóng k_required shard cùng lúc, thiếu thì phóng tiếp phần còn
        # lại. Client thật làm đúng vậy — nó không biết shard nào chết nên gửi hết
        # rồi lấy 20 cái về trước. Độ trễ khi đó là của lookup CHẬM NHẤT, không
        # phải tổng của 20-30 lookup nối đuôi.
        # Cả 5 object ĐỒNG THỜI, và trong mỗi object thì 20 shard cũng đồng thời.
        # FETCH_TOP=1: chỉ lấy payload của kết quả đầu, không lấy cả top-k.
        # Đây là điều một hệ RAG thật thường làm — trả danh sách ID rồi lấy nội
        # dung theo nhu cầu, chứ không lấy sẵn hết.
        _targets = reranked_top[:FETCH_TOP] if FETCH_TOP > 0 else reranked_top
        # Kiểu hỗn hợp: rounds/rpcs/bytes/ok là int, depths là list, misses là int.
        # Không khai báo thì bộ kiểm kiểu suy ra dict[str, int] và báo lỗi ở chỗ
        # a.get("depths", []) vì nó tưởng giá trị trả về là int.
        accs: List[Dict[str, Any]] = [
            {"rounds": 0, "rpcs": 0, "bytes": 0, "ok": 0} for _ in _targets]
        objs = [env.process(_fetch_one_object(env, tag, network_nodes, accs[idx],
                                              k_required))
                for idx, (tag, score) in enumerate(_targets)]
        if objs:
            yield env.all_of(objs)
        for a in accs:
            pay_rounds += a["rounds"]
            pay_rpcs += a["rpcs"]
            pay_bytes += a["bytes"]
            probe_depths.extend(list(a.get("depths") or []))
            shard_misses += int(a.get("misses") or 0)
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
        # --- MC10: chẩn đoán payload ---
        "parallel_adc": PARALLEL_ADC,
        "placement_mode": PLACEMENT_MODE,
        "placement_k": PLACEMENT_K,
        "probe_depth_mean": (sum(probe_depths)/len(probe_depths)) if probe_depths else 0.0,
        "probe_depth_max": max(probe_depths) if probe_depths else 0,
        "probe_depth_p95": (sorted(probe_depths)[int(0.95*len(probe_depths))]
                            if probe_depths else 0),
        "shards_found": len(probe_depths),
        "shard_misses": shard_misses,
        "routing_mode": _mode,
    }
    return retrieved_tags, total_hops, num_unique_candidates, stats