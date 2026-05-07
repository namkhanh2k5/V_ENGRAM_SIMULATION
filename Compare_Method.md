# Báo cáo so sánh kỹ thuật: V-Engram và các giải pháp lưu trữ/tìm kiếm phi tập trung

## 1. So sánh với các giao thức cơ sở (Kademlia, IPFS, Filecoin)

| Tiêu chí | Kademlia (nguyên bản) | IPFS / Filecoin | V-Engram (đề xuất) |
| --- | --- | --- | --- |
| Định vị dữ liệu | Dựa trên mã băm ID tuyệt đối (exact match). | Dựa trên CID sinh từ SHA-256. | Dựa trên semantic key sinh từ AI embedding qua LSH. |
| Khả năng tìm kiếm | Chỉ tìm được khi biết chính xác ID. | "Mù ngữ nghĩa", cần CID chính xác để truy xuất. | Hỗ trợ tìm kiếm mờ/ngữ nghĩa tự nhiên. |
| Phân bố shard | Không có cơ chế phân mảnh mặc định. | Sharding theo Merkle DAG, phân tán ngẫu nhiên. | Placement key (152/8-bit): giữ lân cận ngữ nghĩa và cân bằng tải. |
| Cấu trúc node | Node trung lập (chỉ định tuyến). | Storage node (lưu trữ thuần túy). | Node tính toán - lưu trữ: RAM chạy ADC, SSD lưu payload. |
| Hiệu ứng mạng | Tham lam, đâm thẳng 1 đường tới đích. | Đâm thẳng tới CID. | Ripple search: loang rộng động, tìm nhiều ứng viên. |

## 2. So sánh với các nghiên cứu tiền nhiệm (state-of-the-art)

### 2.1 pSearch (2003)
- Điểm giống: sử dụng LSH để ánh xạ dữ liệu vào không gian mạng.
- Hạn chế: sử dụng LSI cũ với vector thưa, cần semantic overlay cồng kềnh đè lên P2P.
- Cải tiến của V-Engram: dùng dense embedding 1024-dim từ LLM, tích hợp trực tiếp vào khóa Kademlia bản địa, giảm độ trễ và tăng tương thích.

### 2.2 Graph Diffusion (2022)
- Điểm giống: sử dụng cơ chế lan truyền để tìm kiếm kết quả tương đồng.
- Hạn chế: phụ thuộc đồ thị, khó mở rộng, không có cơ chế lưu trữ phân mảnh an toàn.
- Cải tiến của V-Engram: giữ O(log N) của Kademlia, bổ sung AES và Reed-Solomon để bảo toàn dữ liệu.

### 2.3 Semantica (2025)
- Điểm giống: sử dụng embedding hiện đại để tìm kiếm trên mạng phân tán.
- Hạn chế: lưu vector gốc dẫn đến tràn RAM node biên, thiếu cơ chế phân bố shard, gây nghẽn cục bộ.
- Cải tiến của V-Engram:
  - Tối ưu RAM: nén vector 16 lần bằng PQ.
  - Cân bằng tải: placement key + nhiễu HMAC để rải 30 mảnh ra 30 node lân cận.

## 3. Phân tích ưu và nhược điểm của V-Engram

### 3.1 Ưu điểm đột phá
- AI-native routing: biến routing thành tìm kiếm ngữ nghĩa.
- Hiệu quả tài nguyên: PQ/ADC cho phép search trên RAM nhỏ, phù hợp edge/IoT.
- Bảo mật zero-knowledge: AES tại client, shard rải khắp nơi, node lưu trữ không biết nội dung.
- Hệ sinh thái Web3: tích hợp smart contract cho truy xuất dữ liệu, tạo thị trường tìm kiếm AI phi tập trung.

### 3.2 Nhược điểm và thách thức
- Cập nhật phức tạp: nội dung đổi -> semantic key đổi -> di dời 30 shard.
- Sai số lượng tử hóa: PQ giảm 4KB -> 256B, sai số cosine ~0.02, cần reranking.
- Lexical gap: dense vector khó xử lý tìm kiếm exact keyword (mã phiên bản, hằng số).

## 4. Kết luận
V-Engram không chỉ là một giải pháp lưu trữ mới. Nó là sự giao thoa giữa lưu trữ bất biến Web3 và truy vấn thông minh AI. So với IPFS/Filecoin, V-Engram "thông minh" hơn. So với các nghiên cứu như Semantica hay pSearch, V-Engram "thực chiến" hơn nhờ tối ưu phần cứng và cân bằng tải.