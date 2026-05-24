import numpy as np
import hashlib
import hmac

# ============================================================================
# CẤU HÌNH LSH ĐA VŨ TRỤ (MULTI-INDEX)
# ============================================================================
VECTOR_DIM = 1024   
NUM_PROJECTIONS = 5  # Số lượng ma trận chiếu độc lập
np.random.seed(20235956)   # Đảm bảo tính nhất quán toàn mạng

# Khởi tạo 3 ma trận chiếu thưa {-1, 0, 1}
# Giúp nén thông tin hiệu quả và giảm thiểu sai số ranh giới (Boundary problem)
PROJECTION_MATRICES = []
for _ in range(NUM_PROJECTIONS):
    # Achlioptas Distribution: P(1)=1/6, P(-1)=1/6, P(0)=2/3
    matrix = np.random.choice([0, 1, -1], size=(VECTOR_DIM, 160), p=[2/3, 1/6, 1/6])
    PROJECTION_MATRICES.append(matrix)

def generate_multi_semantic_keys(vector):
    """
    Sinh ra 3 Semantic Keys 160-bit khác nhau cho cùng một Vector.
    Mỗi Key đại diện cho một 'góc nhìn' ngữ nghĩa khác nhau.
    """
    vec = np.asarray(vector).flatten()
    keys = []
    for i in range(NUM_PROJECTIONS):
        # Nhân ma trận thưa và lấy dấu (Sign)
        bits = (np.dot(vec, PROJECTION_MATRICES[i]) > 0).astype(int)
        bit_string = "".join(map(str, bits))
        keys.append(int(bit_string, 2))
    return keys

def generate_semantic_key(vector):
    """Hàm wrapper cho các logic cũ cần 1 key (mặc định lấy key đầu tiên)"""
    return generate_multi_semantic_keys(vector)[0]

USER_SECRET_KEY = b"v_engram_dummy_secret_key"

def generate_placement_key(semantic_key, object_tag, shard_id):
    """Tạo địa chỉ cho mảnh vỡ dựa trên nhiễu HMAC 8-bit"""
    base_key = semantic_key & ((1 << 160) - (1 << 8))
    seed_str = f"{semantic_key}_{object_tag}_shard_{shard_id}".encode("utf-8")
    mac_hash = hmac.new(USER_SECRET_KEY, seed_str, hashlib.sha256).hexdigest()
    noise_8bit = int(mac_hash, 16) % 256
    return base_key | noise_8bit

def iterative_find_k_closest_nodes(key, bootstrap_node, alpha=3, k=20, max_rounds=15):
    """
    Mô phỏng định tuyến Kademlia kiểu iterative, không dùng global view.
    Trả về danh sách k node gần nhất và số vòng nhảy (overlay hops).
    """
    candidates = set([bootstrap_node])
    candidates.update(bootstrap_node.get_neighbors())
    queried = set()
    prev_best = None
    hops = 0

    for _ in range(max_rounds):
        ordered = sorted(candidates, key=lambda node: node.node_id ^ key)
        to_query = [node for node in ordered if node not in queried][:alpha]
        if not to_query:
            break

        hops += 1
        for node in to_query:
            queried.add(node)
            candidates.update(node.get_neighbors())

        best_ids = tuple(node.node_id for node in ordered[:k])
        if best_ids == prev_best:
            break
        prev_best = best_ids

    ordered = sorted(candidates, key=lambda node: node.node_id ^ key)
    return ordered[:k], hops