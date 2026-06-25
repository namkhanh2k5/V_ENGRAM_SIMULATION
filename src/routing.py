import numpy as np
import hashlib
import hmac

# ============================================================================
# CẤU HÌNH LSH ĐA VŨ TRỤ (MULTI-INDEX)
# ============================================================================
VECTOR_DIM = 1024   
NUM_PROJECTIONS = 5  # Số lượng ma trận chiếu độc lập
DEFAULT_LSH_SEED = 20235956

# Khai báo rỗng, KHÔNG sinh ma trận ngay lúc import file
PROJECTION_MATRICES = []

def generate_lsh_projections(seed, vector_dim=VECTOR_DIM, num_projections=None):
    """Sinh ma trận chiếu bằng Local Random Generator, miễn nhiễm với bên ngoài"""
    if num_projections is None:
        num_projections = NUM_PROJECTIONS
    rng = np.random.RandomState(seed)
    projections = []
    for _ in range(num_projections):
        # Achlioptas Distribution: P(1)=1/6, P(-1)=1/6, P(0)=2/3
        matrix = rng.choice([0, 1, -1], size=(vector_dim, 160), p=[2/3, 1/6, 1/6])
        projections.append(matrix)
    return projections

def initialize_lsh_projections(seed=DEFAULT_LSH_SEED):
    """Hàm này sẽ được gọi ở main.py mỗi khi bắt đầu một Seed mới"""
    global PROJECTION_MATRICES
    PROJECTION_MATRICES = generate_lsh_projections(seed)

def generate_multi_semantic_keys(vector):
    """
    Sinh ra 3 Semantic Keys 160-bit khác nhau cho cùng một Vector.
    Mỗi Key đại diện cho một 'góc nhìn' ngữ nghĩa khác nhau.
    """
    # Rào bảo vệ: Đề phòng quên chưa gọi initialize()
    if not PROJECTION_MATRICES:
        initialize_lsh_projections()
        
    vec = np.asarray(vector).flatten()
    keys = []
    for i in range(len(PROJECTION_MATRICES)):
        # Nhân ma trận thưa và lấy dấu (Sign)
        bits = (np.dot(vec, PROJECTION_MATRICES[i]) > 0).astype(int)
        bit_string = "".join(map(str, bits))
        keys.append(int(bit_string, 2))
    return keys

def generate_semantic_key(vector):
    """Hàm wrapper cho các logic cũ cần 1 key (mặc định lấy key đầu tiên)"""
    return generate_multi_semantic_keys(vector)[0]

USER_SECRET_KEY = b"v_engram_dummy_secret_key"

def generate_placement_key(object_tag, shard_id):
    """Placement key độc lập ngữ nghĩa: K_place(s) = HMAC(tag, s) trên toàn bộ
    không gian khoá 160-bit của Kademlia.

    Không phụ thuộc semantic key, nên các shard được rải ĐỀU khắp không gian địa chỉ
    và mọi vị trí đều tái tạo được CHỈ từ object_tag (phục vụ khôi phục stateless).
    Đây là tầng PAYLOAD: mỗi object chỉ đặt MỘT bộ 30 shard (không nhân theo L).
    """
    seed_str = f"{object_tag}_shard_{shard_id}".encode("utf-8")
    mac_hash = hmac.new(USER_SECRET_KEY, seed_str, hashlib.sha256).hexdigest()
    # SHA-256 cho 256 bit -> thu về 160-bit keyspace
    return int(mac_hash, 16) % (1 << 160)

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