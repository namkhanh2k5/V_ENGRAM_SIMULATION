import heapq
import numpy as np
import hashlib

# ============================================================================
# CẤU HÌNH LSH ĐA VŨ TRỤ (MULTI-INDEX)
# ============================================================================
VECTOR_DIM = 1024
NUM_PROJECTIONS = 3  # Số lượng ma trận chiếu độc lập
np.random.seed(2026)   # Đảm bảo tính nhất quán toàn mạng

# Khởi tạo 3 ma trận chiếu thưa {-1, 0, 1}
# Giúp nén thông tin hiệu quả và giảm thiểu sai số ranh giới (Boundary problem)
PROJECTION_MATRICES = []
for i in range(NUM_PROJECTIONS):
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

def generate_placement_key(semantic_key, object_tag, shard_id):
    """Tạo địa chỉ cho mảnh vỡ dựa trên nhiễu XOR 8-bit"""
    base_key = semantic_key & ((1 << 160) - (1 << 8))
    seed_str = f"{semantic_key}_{object_tag}_shard_{shard_id}".encode("utf-8")
    noise_8bit = int(hashlib.sha1(seed_str).hexdigest(), 16) % 256
    return base_key | noise_8bit

def find_closest_node(placement_key, all_nodes):
    """Tìm Node duy nhất có khoảng cách XOR nhỏ nhất"""
    return min(all_nodes, key=lambda node: node.node_id ^ placement_key)

def find_k_closest_nodes(key, all_nodes, k=30):
    """Tìm cụm K Node lân cận trong không gian XOR"""
    return heapq.nsmallest(k, all_nodes, key=lambda node: node.node_id ^ key)