# RE-RUN PAYLOAD-ONCE — Hướng dẫn

Bản sửa này chuyển hệ sang đúng kiến trúc **two-tier** của bài: **metadata nhân L lần,
payload đặt 1 lần**. Trước đây code đặt payload **L lần** (load = 60·L), nay đặt **1 lần**
(load không đổi theo L).

## Đã sửa gì (chép đè vào repo)

| File | Thay đổi |
|------|----------|
| `src/routing.py` | `generate_placement_key(tag, s_id) = HMAC(tag, s)` — full 160-bit, **độc lập semantic key**. Cho phép đổi `NUM_PROJECTIONS` (L) lúc chạy. |
| `src/node.py` | Tách `store_metadata` (PQ→RAM) và `store_payload_shard` (shard→SSD). |
| `src/network.py` | Ingestion 2 tầng: **TẦNG 1** neo PQ code ở `METADATA_ANCHORS` node quanh **mỗi** semantic key (nhân L); **TẦNG 2** đặt **30 shard 1 lần** ở HMAC key. Query khôi phục payload **1 lần từ tag** + **đếm unique-candidate/query**. |
| `main_churn_test.py` | Khôi phục payload-once (bỏ vòng lặp L semantic key). |
| `main_simulation.py`, `main_rag_single_seed.py` | Nhận thêm giá trị `unique-candidate` trả về + in/ghi báo cáo. |
| `run_L_ablation.py` *(mới)* | Quét L=1..5 (Code corpus), ghi log đầy đủ + bảng `consolidated_l`. |
| `run_all.sh` *(mới)* | Chạy tất cả + **tee toàn bộ output ra `logs/run_all_<timestamp>.log`**. |

## Chạy thế nào — MỘT LỆNH, MỘT FILE LOG

1. **Chép đè** các file vào repo `V_ENGRAM_SIMULATION` (giữ cây `src/`).
2. Cài deps: `pip install simpy numpy sentence-transformers faiss-cpu`
3. Kiểm `ls ./data/` có đủ file (xem bảng đường dẫn cuối README).
4. Chạy **một lệnh duy nhất**:
   ```bash
   python run_all_experiments.py        # hoặc:  bash run_all.sh --bg   (chạy nền)
   ```
   → Chạy lần lượt **5 khối**:
   - BLOCK 1: L-ablation (Code, L=1..5) → `consolidated_l`   **[5 seed]**
   - BLOCK 2: SciFact (L=5) → `scifact`                       **[5 seed]**
   - BLOCK 3: Churn (Code, 10/20/30%) → §4.9                  **[5 seed]**
   - BLOCK 4: Scalability (Code, node 10k→25k) → `scalability`  **[1 seed]**
   - BLOCK 5: Placement breadth B_place (Code) → `placement`    **[1 seed]**

   *(§4.4 Baseline và §4.5 Prefix-validation KHÔNG re-run — độc lập với payload, số cũ vẫn dùng.)*

   **TẤT CẢ kết quả ghi vào một file:** `logs/v_engram_all_<timestamp>.log`
   (có log tiến độ + thời gian từng (khối,L,seed) + bảng tổng kết sẵn để dán vào bài).

   ⚠️ **RẤT LÂU**: ~**44 lần ingest** (L-ablation 25 + SciFact 5 + Churn 5 + Scalability 4 + Placement 5), gồm 1 lần ở 25k node.
   Nên chạy nền: `bash run_all.sh --bg` rồi `tail -f logs/v_engram_all_*.log`.
   Thử nhanh trước: tạm sửa `SEEDS=[20235956]`, `L_VALUES_CODE=[3]` trong `run_all_experiments.py`.

## Kỳ vọng số liệu (để đối chiếu)

- **Mean Load (shards/node)** ≈ `NUM_FILES*30/NUM_NODES` = **~60**, và **KHÔNG đổi theo L**
  (trước là 60/120/180/240/300). **Std/Max thấp hẳn** — HMAC rải đều, **hết semantic hotspot ở payload**.
- **SciFact**: Mean Load ~ `5183*30/10000` ≈ **15.5** (trước 77.7).
- **Success@5 / MRR@5 / Avg Hops**: **PHẢI giữ nguyên** so với bản cũ (payload được lấy *sau*
  truy hồi nên không ảnh hưởng discovery). **Nếu các số này tụt → có bug**, báo lại ngay
  (khả năng cao do `METADATA_ANCHORS` quá nhỏ; tăng nó trong `src/network.py` rồi chạy lại).
- **Unique-candidate/query**: số mới (số object thực sự được rerank, % corpus) — phục vụ phản biện Q1.

## Lưu ý churn (§4.9)

Payload giờ chỉ 1 bộ RS(30,20) → chịu mất tối đa 33%, **không còn dự phòng L-fold**. Hãy xem
`main_churn_test.py` ở mức churn 30% có còn khôi phục tốt không. Nếu kết quả churn đổi so với
bài, cần cập nhật §4.9 + §5 cho khớp (đã lường trước).
