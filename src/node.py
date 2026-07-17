import random
import numpy as np

class VEngramNode:
    def __init__(self, env, node_id):
        self.env = env
        self.node_id = node_id
        self.RAM_Index = {}   
        self.SSD_Storage = {} 
        self.routing_table = set()

    def ping(self, target_node):
        yield self.env.timeout(random.uniform(10, 100))
        self.routing_table.add(target_node)

    def get_neighbors(self):
        return list(self.routing_table)

    def store_metadata(self, object_tag, pq_code):
        """TẦNG 1 (Metadata): neo PQ code (phục vụ ADC rerank) vào RAM.
        Được nhân bản tại các node gần MỖI semantic key trong L key của object."""
        if object_tag not in self.RAM_Index:
            self.RAM_Index[object_tag] = pq_code

    def store_payload_shard(self, object_tag, shard_id, virtual_payload):
        """TẦNG 2 (Payload): lưu MỘT mảnh Reed-Solomon xuống đĩa.
        Mỗi object chỉ đặt một bộ shard (không nhân theo L)."""
        shard_key = f"{object_tag}_shard_{shard_id}"
        self.SSD_Storage[shard_key] = virtual_payload

    def store_shard(self, object_tag, shard_id, pq_code, virtual_payload):
        """[Giữ cho tương thích ngược - không dùng trong pipeline two-tier mới]"""
        self.store_metadata(object_tag, pq_code)
        self.store_payload_shard(object_tag, shard_id, virtual_payload)
        
    def adc_search(self, query_vector, codebook, top_k=5):
        """
        GIAI ĐOẠN 3: TÍNH ADC THỰC TẾ TRÊN RAM.
        Sử dụng Bảng mã Codebook (m, 256, d_sub) và mã PQ uint8.
        """
        candidates = []
        if not self.RAM_Index:
            return candidates
            
        # Đọc m, d_sub TỪ codebook thay vì hardcode -> hỗ trợ mọi biến thể PQ
        # (m=256/d_sub=4 = 256 byte/doc; m=512/d_sub=2 = 512 byte/doc, sai số ~10x nhỏ hơn)
        m, _, d_sub = codebook.shape
        
        # BƯỚC 1: Cắt Vector Query 1024 chiều ra 256 đoạn
        query_subvectors = np.asarray(query_vector, dtype=np.float32).reshape(m, d_sub)
        
        # BƯỚC 2: Tự động tính Bảng tra cứu (Look-Up Table) bằng Vectorization
        # So sánh 256 đoạn của Query với 256 Centroids của Codebook
        diff = query_subvectors[:, np.newaxis, :] - codebook
        LUT = np.sum(diff**2, axis=2) # L2 Distance squared. Shape: (256, 256)
        
        # BƯỚC 3: Duyệt qua RAM Index và tra bảng
        for tag, pq_code in self.RAM_Index.items():
            # pq_code là mảng 256 con số nguyên (0-255).
            # Thay vì nhân vector, ta chỉ cộng 256 con số lấy từ LUT
            dist = np.sum(LUT[np.arange(m), pq_code])
            candidates.append((tag, dist))
            
        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k]

    def get_shard(self, shard_key):
        yield self.env.timeout(random.uniform(2, 8))
        return self.SSD_Storage.get(shard_key, None)