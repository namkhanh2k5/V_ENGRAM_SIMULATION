# Quy Trình Chuẩn Bị Dữ Liệu V-Engram

Folder này chứa các script Python để chuẩn bị dữ liệu cho mô phỏng hệ thống truy xuất phân tán V-Engram. Mỗi script xử lý một giai đoạn của quy trình chuẩn bị dữ liệu.

## Khởi Động Nhanh

### Lần Đầu Tiên (Cài Đặt Từ Đầu)

Chạy các script theo thứ tự sau:

```bash
# 1. Cài đặt dependencies và tạo thư mục ./data/
python 01_setup_local.py

# 2. Tải CodeSearchNet → nhúng vector → xây dựng FAISS baseline
python 02_build_ground_truth.py

# 3. Huấn luyện Product Quantization (khoảng 2-5 phút)
python 05_train_pq.py
```

Sau đó chạy mô phỏng chính:
```bash
cd ..
python main_simulation.py
```

---

## Chi Tiết Từng Script

### 01_setup_local.py
**Mục đích:** Cài đặt dependencies và tạo cấu trúc thư mục dữ liệu.

**Công việc:**
- Cài đặt: `sentence-transformers`, `faiss-cpu`, `datasets`
- Tạo thư mục `./data/` (nếu chưa tồn tại)

**Khi nào chạy:**
- [YES] Luôn chạy trước tiên trên máy mới
- [SKIP] Bỏ qua các lần tiếp theo (thư mục dữ liệu tồn tại)

**Kết quả:** Thư mục `./data/`

**Thời gian dự kiến:** ~2 phút (cài pip)

---

### 02_build_ground_truth.py
**Mục đích:** Tạo embedding, xây dựng chỉ mục FAISS, tính toán baseline gốc.

**Công việc:**
1. Tải dataset CodeSearchNet (20k đoạn mã Python)
2. Mã hóa tất cả với BGE-Large-en-v1.5 (embedding 1024 chiều)
3. L2-normalize tất cả vector
4. Xây dựng FAISS IndexFlatIP (chỉ mục cosine similarity)
5. Tải 15 truy vấn test từ `faiss_absolute_baseline.json` (hoặc dùng mặc định)
6. Tính top-5 kết quả FAISS cho mỗi truy vấn
7. Lưu tất cả vào `./data/`

**Khi nào chạy:**
- [YES] Lần đầu tiên cài đặt
- [NO] KHÔNG chạy lại (mất 10+ phút, ghi đè baseline.json)
- [INSTEAD] Cần tính lại FAISS mà không re-embedding, dùng `03_regenerate_baseline.py`

**Input:** CodeSearchNet (tự động tải), các truy vấn mặc định

**Output:**
- `embeddings_20k.npy` — 20k vector (20000 × 1024, float32, đã L2-normalize)
- `v_engram_dataset_20k.json` — 20k doan code goc
- `faiss_absolute_baseline.json` — 15 truy vấn + top-5 kết quả gốc
- `pipeline_metadata.json` — Thông tin model, kích thước dataset

**Thời gian dự kiến:** ~10-15 phút (tải model + embedding)

---

### 03_regenerate_baseline.py
**Mục đích:** Tính lại FAISS baseline mà không re-embedding (nhanh hơn khi thực nghiệm).

**Công việc:**
1. Tải embedding đã tính từ `embeddings_20k.npy`
2. Tải 15 truy vấn từ `faiss_absolute_baseline.json` (hoặc mặc định)
3. Xây dựng lại chỉ mục FAISS từ embedding
4. Tính lại top-5 kết quả cho mỗi truy vấn
5. Ghi đè `faiss_absolute_baseline.json`

**Khi nào chạy:**
- [YES] Khi điều chỉnh logic truy vấn hoặc tham số FAISS
- [YES] Khi danh sách truy vấn thay đổi nhưng embedding không
- [NO] Nếu embedding chưa có, chạy `02_build_ground_truth.py` trước

**Input:** `embeddings_20k.npy`, `faiss_absolute_baseline.json`

**Output:** `faiss_absolute_baseline.json` cập nhật (cùng cấu trúc)

**Thời gian dự kiến:** ~1-2 phút (chỉ tính lại FAISS)

---

### 04_mine_and_update_queries.py
**Mục đích:** Tự động khai thác truy vấn mới để đủ 15 truy vấn (script hợp nhất).

**Công việc:**
1. Tải 15 truy vấn từ `faiss_absolute_baseline.json` (hoặc 15 mặc định)
2. Nếu số lượng < 15: khai thác thêm bằng:
   - Lấy mẫu ngẫu nhiên từ embedding
   - Mã hóa qua BGE model
   - Tìm kiếm FAISS những tài liệu tương tự
   - Lọc ứng viên có cosine similarity > 0.88
3. Hợp nhất thành đúng 15 truy vấn
4. Tính lại top-5 kết quả cho tất cả truy vấn
5. Ghi đè `faiss_absolute_baseline.json`

**Khi nào chạy:**
- [YES] Muốn mở rộng hoặc làm mới bộ truy vấn
- [YES] Đảm bảo đúng 15 truy vấn (idempotent)
- [NO] Nếu hài lòng với 15 truy vấn hiện tại, bỏ qua

**Input:** `embeddings_20k.npy`, `faiss_absolute_baseline.json`

**Output:** `faiss_absolute_baseline.json` cập nhật với 15 truy vấn hợp nhất

**Thời gian dự kiến:** ~2-3 phút (nếu cần khai thác)

---

### 05_train_pq.py
**Mục đích:** Huấn luyện Product Quantization để nén vector.

**Công việc:**
1. Tải embedding từ `embeddings_20k.npy`
2. Huấn luyện PQ với:
  - **m=256** sub-vector (1024-dim / 256 = 4-dim mỗi cái)
  - **nbits=8** per centroid (256 giá trị mỗi sub)
3. Trích codebook: hình dạng (256, 256, 4)
   - 256 sub-vector × 256 centroid × 4-dim mỗi cái
4. Lượng tử hóa 20k embedding thành PQ code (uint8, 256 byte mỗi cái)
5. Lưu: `pq_codes.npy`, `pq_codebook.npy`

**Khi nào chạy:**
- [YES] Sau `02_build_ground_truth.py` hoàn thành
- [YES] Phải chạy trước `main_simulation.py` (main.py tải pq_codes.npy)
- [ONCE] Chạy 1 lần, không cần chạy lại trừ khi thay đổi tham số PQ

**Input:** `embeddings_20k.npy`

**Output:**
- `pq_codes.npy` — PQ code lượng tử hóa (20000 × 256, uint8)
- `pq_codebook.npy` — PQ centroid (256, 256, 4, float32)

**Thời gian dự kiến:** ~2-5 phút (K-means mỗi sub-vector)

---

## Chuỗi Phụ Thuộc

```
┌─ 01_setup_local.py
│
├─ 02_build_ground_truth.py  ──────┐
│  (tạo embedding)                 │
│                                   ├─ 05_train_pq.py
├─ 03_regenerate_baseline.py ──────┤
│  (tùy chọn, cập nhật baseline)   │
│                                   ├─→ main_simulation.py
├─ 04_mine_and_update_queries.py ──┤
│  (tùy chọn, mở rộng truy vấn)    │
│                                   │
└─────────────────────────────────┘
```

**Đường dẫn quan trọng:**
- `01_setup_local.py` → `02_build_ground_truth.py` → `05_train_pq.py` → `main_simulation.py`

**Đường dẫn tùy chọn:**
- Chèn `03_regenerate_baseline.py` nếu điều chỉnh FAISS
- Chèn `04_mine_and_update_queries.py` nếu mở rộng bộ truy vấn

---

## Định Dạng Baseline Truy Vấn

Tất cả truy vấn được lưu trong `faiss_absolute_baseline.json` (danh sach object):

```json
[
  {
    "query_id": 1,
    "query_text": "def get_logger(name):",
    "top_5_results": [
      {
        "rank": 1,
        "index": 1272,
        "cosine_similarity": 0.8453,
        "code_snippet": "def get_logger(name): ..."
      }
    ]
  }
]
```

**Điểm chính:**
- Đúng **15 truy vấn** (query_id 1-15)
- Mỗi truy vấn có **5 kết quả gốc** (FAISS top-5)
- Dùng để đánh giá recall trong `main_simulation.py`

---

## Cấu Trúc Dữ Liệu (./data/)

Sau khi cài đặt hoàn toàn, thư mục `./data/` chứa:

```
data/
├── embeddings_20k.npy              # 20k vector (20000×1024, float32, đã L2-normalize)
├── faiss_absolute_baseline.json    # 15 truy vấn + top-5 gốc
├── pq_codes.npy                    # Code lượng tử hóa (20000×256, uint8)
├── pq_codebook.npy                 # PQ centroid (256×256×4, float32)
├── pipeline_metadata.json          # Metadata dataset/model
└── v_engram_dataset_20k.json        # 20k doan code goc
```

**Kích thước tổng:** ~100 MB

---

## Xử Lý Sự Cố

### "embeddings_20k.npy not found"
→ Chạy `02_build_ground_truth.py` trước

### "faiss_absolute_baseline.json not found"
→ Chạy `02_build_ground_truth.py` trước

### "pq_codes.npy not found"
→ Chạy `05_train_pq.py` sau khi embedding sẵn sàng

### Script chạy rất chậm
→ Kiểm tra CPU/RAM; PQ training và embedding tính toán nhiều
→ Dùng CPU (không GPU) là bình thường cho cài đặt này

### Module FAISS không tìm thấy
→ Chạy `01_setup_local.py` lại để cài đặt dependencies

---

## Hiệu Năng Dự Kiến

| Script | Thời gian | Ghi chú |
|--------|-----------|--------|
| 01_setup_local.py | ~2 phút | pip install (1 lần/máy) |
| 02_build_ground_truth.py | ~10-15 phút | CodeSearchNet + BGE embedding |
| 03_regenerate_baseline.py | ~1-2 phút | Chỉ tính lại FAISS |
| 04_mine_and_update_queries.py | ~2-3 phút | Khai thác + tìm kiếm FAISS |
| 05_train_pq.py | ~2-5 phút | K-means mỗi sub-PQ |
| **main_simulation.py** | ~7-10 phút | Mô phỏng đầy đủ: 10k nodes, 20k files, 15 queries |

---

## Nâng Cao: Thay Đổi Bộ Truy Vấn

Để dùng truy vấn riêng thay vì mặc định:

1. Sửa `faiss_absolute_baseline.json` thủ công (tối đa 15 truy vấn)
2. Chạy `03_regenerate_baseline.py` để tính top-5 kết quả
3. Hoặc chạy `04_mine_and_update_queries.py` để tự động mở rộng đến 15

**Lưu ý:** Embedding truy vấn được tính on-the-fly trong các script dùng BGE model.

---

## Bước Tiếp Theo

Sau khi script chuẩn bị hoàn thành, trở về thư mục gốc và chạy:

```bash
cd ..
python main_simulation.py
```

Điều này sẽ:
- Tải embedding đã chuẩn bị, PQ code, và baseline
- Khởi tạo 10,000 nút DHT
- Phân phối 20,000 file × 30 shard mỗi cái
- Thực hiện 15 truy vấn qua ripple search
- Tạo báo cáo metrics vào `comparison_report.txt`

---

**Lần cập nhật cuối:** 2026-05-04  
**Phiên bản:** 1.0  
**Trạng thái:** Sẵn sàng sản xuất
