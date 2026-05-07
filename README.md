### 1. Phương pháp (Method)

#### 1.1 Tổng quan hệ thống (System Overview)
Hệ thống được thiết kế nhằm tích hợp khả năng truy xuất ngữ nghĩa (semantic retrieval) vào trong một hạ tầng định tuyến phi tập trung dựa trên Distributed Hash Table (DHT). Trong các hệ DHT truyền thống, dữ liệu được ánh xạ bằng các hàm băm mật mã, dẫn đến việc mất hoàn toàn cấu trúc ngữ nghĩa: hai đối tượng rất giống nhau vẫn có thể nằm ở hai vùng rất xa nhau trong không gian khóa. Ngược lại, các hệ semantic retrieval hiện đại dựa trên embedding lại thường mang tính tập trung và khó mở rộng.

Hệ thống này giải quyết mâu thuẫn đó bằng cách ánh xạ trực tiếp embedding vào không gian khóa sao cho khoảng cách trong không gian bit phản ánh khoảng cách cosine trong không gian vector. Nhờ đó, quá trình định tuyến trên DHT trở thành một dạng truy xuất ngữ nghĩa phân tán.

Hệ thống hoạt động dựa trên cơ sở **Bảng băm hai tầng (Two-Tier DHT)**, chia vòng đời dữ liệu thành các luồng rõ rệt:
1. **Luồng Ghi (Data Ingestion Pipeline):** Khi một file mới được ném vào mạng lưới, nó bị tách làm 2 nhánh chạy song song. Nhánh 1 (Dữ liệu thực) đi qua AI Model biến thành Vector, đưa vào hàm LSH để sinh ra các `semantic_keys`, sau đó file được cắt thành mảnh (shard), trộn nhiễu để ra `placement_keys` và rải lưu trữ on-disk. Nhánh 2 (Siêu dữ liệu) lấy tên file (`object_tag`) đưa qua hàm SHA-256 để tìm tọa độ tĩnh, rồi gửi "tờ giấy ghi nhớ" chứa `semantic_keys` cho khoảng 10-20 nodes lưu vào RAM để làm bản sao lưu.
2. **Luồng Đọc (Query & Recovery Pipeline):** Diễn ra qua 3 bước thần tốc. Bước 1 là Ripple Search để tìm tên file (`object_tag`) khớp nhất với câu truy vấn. Bước 2 là Tra sổ đỏ, tự băm SHA-256 cái tên file để tìm Node giữ siêu dữ liệu và xin lại `semantic_keys`. Bước 3 là tự tái tạo `placement_keys` để gõ cửa các node giữ mảnh vỡ, lấy về ráp lại thành file hoàn chỉnh.

#### 1.2 Biểu diễn ngữ nghĩa và ánh xạ LSH
Mỗi đối tượng dữ liệu được biểu diễn dưới dạng vector embedding $x \in \mathbb{R}^{1024}$. Để ánh xạ vector này sang không gian khóa rời rạc, hệ thống sử dụng Sign Random Projection (SRP), một dạng Locality-Sensitive Hashing. Mỗi bit được xác định bởi dấu của tích vô hướng:

$$h_i(x) = \text{sgn}(r_i \cdot x)$$

Tập hợp các bit này tạo thành một khóa nhị phân 160 bit. Tính chất quan trọng của SRP là xác suất hai vector cho cùng một bit phụ thuộc trực tiếp vào góc giữa chúng:

$$P(h_i(x) = h_i(y)) = 1 - \frac{\theta(x,y)}{\pi}$$

Do đó, hai vector có cosine similarity cao sẽ có nhiều bit giống nhau, và khoảng cách Hamming giữa hai semantic key sẽ xấp xỉ khoảng cách cosine. Điều này cho phép sử dụng không gian XOR của Kademlia để định tuyến theo ngữ nghĩa.

Để giảm chi phí tính toán, hệ thống sử dụng ma trận chiếu theo phân phối Achlioptas, trong đó các phần tử chỉ nhận giá trị -1, 0, +1. Nhờ tính chất thưa này, phép chiếu trở nên nhẹ hơn đáng kể mà vẫn giữ được cấu trúc khoảng cách.

#### 1.3 Multi-Projection Indexing
Để giảm sai số do hashing ngẫu nhiên, hệ thống sử dụng nhiều phép chiếu độc lập. Với ba ma trận, mỗi embedding sẽ sinh ra ba semantic key khác nhau. Mỗi key tương ứng với một cách phân hoạch không gian khác nhau.

Việc sử dụng nhiều projection giúp giảm xác suất bỏ sót dữ liệu do boundary effect của LSH. Trong quá trình ingest, các shard của một object được phân bố lên các semantic region khác nhau tương ứng với các key này. Trong quá trình truy vấn, query cũng được chiếu qua tất cả các ma trận, từ đó truy vấn được gửi tới nhiều vùng semantic khác nhau. Multi-projection mở rộng không gian tìm kiếm theo chiều rộng, giúp tăng recall mà không cần tăng mạnh độ sâu của từng truy vấn.

#### 1.4 Placement Key và phân bố shard
Semantic key được sử dụng để định vị đối tượng trong không gian ngữ nghĩa, tuy nhiên không thể dùng trực tiếp semantic key này để lưu trữ tất cả các shard của cùng một object. Nếu làm như vậy, toàn bộ shard sẽ bị ánh xạ tới cùng một vị trí hoặc một cụm node rất nhỏ trong DHT, dẫn đến mất cân bằng tải, giảm khả năng chịu lỗi, và làm tăng nguy cơ lộ cấu trúc dữ liệu.

Để giải quyết vấn đề này, hệ thống xây dựng một placement key riêng cho từng shard, cho phép vừa giữ được tính gần nhau về ngữ nghĩa, vừa đảm bảo phân tán dữ liệu trên nhiều node khác nhau. Giá trị `object_tag` đóng vai trò là định danh duy nhất của object và được sinh từ metadata nhận diện của object thay vì chỉ dựa trên tên file. 

Cụ thể, semantic key 160-bit được chia thành hai phần. Phần 152 bit đầu tiên được giữ nguyên và đóng vai trò như một semantic anchor, đảm bảo rằng tất cả các shard của cùng một object vẫn nằm trong cùng một semantic neighborhood. Phần 8 bit cuối cùng được sử dụng để tạo sự đa dạng trong phân bố thông qua phép XOR, đóng vai trò như một phép “nhiễu có kiểm soát” phụ thuộc vào `object_tag` và chỉ số shard:

$$P_i = (B \land \text{Mask}_{152}) \lor (\text{HMAC}(\mathrm{object\_tag}, i) \bmod 256)$$

Với 8 bit, hệ thống tạo ra tối đa 256 vị trí khả dĩ cho mỗi semantic region, đủ lớn để phân tán các shard (ví dụ 30 shard) lên các node khác nhau, nhưng vẫn đủ nhỏ để giữ tất cả shard trong cùng một vùng lân cận.

Việc sử dụng hàm băm với `object_tag` đảm bảo rằng cách phân bố shard là deterministic nhưng khó đoán đối với bên ngoài. Trong quá trình lưu trữ, mỗi placement key được đưa vào cơ chế định tuyến của Kademlia, và node có NodeID gần nhất theo khoảng cách XOR sẽ chịu trách nhiệm lưu shard tương ứng.

#### 1.5 Lưu trữ tại node (Phân tách Trách nhiệm)
Mỗi node lưu trữ dữ liệu theo hai tầng: một tầng in-memory để phục vụ truy vấn nhanh và một tầng on-disk để lưu payload thực. Thiết kế này tách biệt rõ retrieval và storage: RAM tối ưu cho search, disk tối ưu cho lưu trữ.

**Trong RAM**, node giữ chỉ mục truy xuất cực nhẹ, hoàn toàn vắng bóng `semantic_key` để phục vụ riêng cho khâu tìm kiếm xấp xỉ:
```json
{
   "pq_code": "vector đã nén bằng Product Quantization",
   "object_tag": "cryptographic identifier derived from metadata",
   "shard_index": "xác định vị trí shard",
   "pointer_to_shard_on_disk": "trỏ tới dữ liệu thật trên disk"
}
```

**Trên Disk**, node lưu trữ dữ liệu đầy đủ phục vụ cho khâu lưu trữ và tự phục hồi (Self-Healing):
```json
{
   "encrypted_shard_data": "shard ciphertext sau AES",
   "shard_hash": "SHA-256 để kiểm tra integrity",
   "object_tag": "cryptographic identifier",
   "semantic_key": "original 160-bit semantic hash",
   "shard_index": "thứ tự shard",
   "required_shards": "số shard cần để reconstruct (k)",
   "total_shards": "tổng số shard (n)"
}
```
Lưu ý rằng `semantic_key` được cất trong ổ cứng nhằm mục đích cho phép node tự động tái tính toán placement key để di chuyển shard sang node khác nếu mạng lưới có sự cố, mà không cần truy vấn lại tầng Sổ đỏ.

#### 1.6 Truy vấn động với K-search và Ripple Search
Trong V-Engram, không gian truy vấn lân cận không được cố định trước mà được mở rộng một cách động thông qua cơ chế Dynamic K_Search. Hệ thống không truy vấn toàn bộ vùng lân cận ngay từ đầu, mà triển khai một chiến lược Ripple Search, trong đó truy vấn lan truyền dần theo các vòng đồng tâm xung quanh semantic key ban đầu.

Bán kính quét được nội suy theo quy mô mạng, với ngân sách mở rộng tăng xấp xỉ theo $O(\log N)$. Nếu hệ thống đã thu được đủ số lượng candidate hoặc shard vượt ngưỡng tương đồng cosine mong muốn, quá trình truy vấn sẽ dừng lại ngay tại biên thông qua cơ chế Early Stopping. Thiết kế này cho phép truy vấn thích ứng với độ khó ngữ nghĩa của từng yêu cầu, vừa cải thiện recall, vừa tránh hiện tượng bão hòa băng thông mạng.

#### 1.7 Định tuyến đa mục tiêu theo khoảng cách ngữ nghĩa và độ trễ vật lý
Nếu chỉ tối ưu hóa theo khoảng cách ngữ nghĩa trong không gian XOR, hệ thống có thể chọn ra các node rất gần về semantic nhưng lại nằm quá xa về mặt vật lý, dẫn đến độ trễ mạng lớn. Để khắc phục, V-Engram sử dụng định tuyến đa mục tiêu, đánh giá đồng thời khoảng cách XOR đến semantic key và độ trễ mạng thực tế (RTT). Hàm chi phí được định nghĩa như sau:

$$\text{Cost} = \beta \cdot \text{Norm}(D_{XOR}) + (1 - \beta) \cdot \text{Norm}(RTT)$$

Điều này tương đương với một lựa chọn gần tinh thần tối ưu Pareto, ưu tiên các node vừa chứa dữ liệu liên quan cao, vừa có độ ổn định và độ trễ vật lý tốt.

#### 1.8 Product Quantization và ADC
Vector embedding 1024 chiều không thể lưu trực tiếp trong RAM ở quy mô lớn. Hệ thống sử dụng Product Quantization (PQ) để nén vector mà vẫn bảo toàn cấu trúc khoảng cách. Mỗi vector 1024 chiều được chia thành 256 đoạn con, áp dụng K-Means clustering để xây dựng codebook gồm 256 centroid. Kết quả là vector ban đầu (khoảng 4096 bytes) được nén xuống còn chuỗi 256 byte, tương đương một chuỗi 256 chỉ số:

$$C(x) = [c_1, c_2, ..., c_{256}]$$

Trong quá trình truy vấn, hệ thống sử dụng Asymmetric Distance Computation (ADC) để tính khoảng cách gần đúng. Query được giữ nguyên dạng vector đầy đủ, xây dựng bảng tra khoảng cách (LUT) với các centroid, sau đó cộng các giá trị tra bảng tương ứng với từng ID trong PQ code của dữ liệu. Các candidate từ nhiều node được hợp nhất tại phía client, sau đó một bước reranking được thực hiện để chọn ra object phù hợp nhất.

#### 1.9 Reconstruction (Khôi phục dữ liệu)
Quá trình khôi phục tuân thủ nghiêm ngặt nguyên lý phi trạng thái (stateless):
1. Sau khi truy vấn (Ripple Search) và reranking hoàn tất, client có được `object_tag`.
2. Client sử dụng `object_tag` tra cứu Tầng Metadata (Sổ đỏ) để thu lại `semantic_key` ban đầu.
3. Với `semantic_key` và `object_tag`, client tự tính lại placement key cho các shard mà không cần bảng ánh xạ vị trí riêng biệt.
4. Client thực hiện truy vấn DHT để tìm các node gần nhất theo khoảng cách XOR và yêu cầu trả về shard tương ứng.
5. Các shard thu thập được sẽ được kiểm tra toàn vẹn bằng băm SHA-256. Khi có đủ số lượng shard hợp lệ, thuật toán Reed–Solomon được kích hoạt để phục hồi ciphertext gốc, sau đó giải mã bằng AES để thu được plaintext.

#### 1.10 Thực nghiệm và Kết quả
Thực nghiệm được chạy trên 20.000 đoạn mã nguồn, embedding 1024 chiều với BAAI/bge-large-en-v1.5. Mạng mô phỏng gồm 10.000 node. Kết quả thực tế (theo result.txt):

*   **Load distribution:** Mean 179.7 shards/node, Std 292.4, Max load 2,288 shards.
*   **Routing + accuracy:** Trung bình ~22.0 hops mỗi truy vấn, Recall/HitRate@5 = 100.0% so với FAISS.
*   **Resource:** Thời gian mô phỏng ảo 537,264.47 ms; tiết kiệm RAM 16x (4096 bytes -> 256 bytes).

#### 1.10.1 Độ phức tạp và ước tính thời gian

| Quá trình | Tác vụ | Độ phức tạp toán học | Thời gian ước tính (thực tế) | Ghi chú thiết kế |
| --- | --- | --- | --- | --- |
| Ghi (Ingest) | AI Embedding | $O(L^2 \cdot d)$ | 50ms (CPU) / 5ms (GPU) | Nút thắt cổ chai chính, phụ thuộc cấu hình client. |
| Ghi (Ingest) | Reed-Solomon Encode | $O(S \cdot k \cdot (n-k))$ | 3ms | Xử lý CPU cục bộ, nhẹ. |
| Ghi (Ingest) | Phân tán 30 Shards | $O(\max RTT_{1\rightarrow30})$ | 30ms-50ms | I/O song song. |
| Ghi (Ingest) | Lưu sổ đỏ Kademlia | $O(\log N)$ | 15ms | Xử lý song song với ghi shards. |
| Đọc (Query) | LSH + K-Search | $O(P \cdot K + P \cdot K \cdot C)$ | 20ms-40ms | $P$ là số projection, $K$ là số node quét, $C$ là chi phí ADC/node. |
| Đọc (Query) | ADC Search (RAM) | $O(F)$ | <2ms | Duyệt số vector lưu tại node, LUT tạo trực tiếp. |
| Đọc (Query) | Tải 20 Shards | $O(\max RTT_{1\rightarrow20})$ | 20ms-40ms | Không cần chờ 10 shard chậm nhất. |
| Đọc (Query) | Reed-Solomon Decode | $O(S \cdot k^2)$ | 3ms | Thực tế có thể tối ưu với thuật toán nhanh hơn. |

**Giải thích ký hiệu:**

*   $L$: độ dài chuỗi đầu vào (token length)
*   $d$: số chiều embedding
*   $S$: kích thước dữ liệu cần mã hóa (bytes)
*   $k, n$: thông số Reed-Solomon ($k$ shard cần, $n$ shard tổng)
*   $P$: số projection trong LSH
*   $K$: số node được quét trong K-Search
*   $C$: chi phí ADC trên một node
*   $F$: số vector được lưu trong RAM của node
*   $RTT$: round-trip time giữa các node

#### 1.11 Thảo luận và Giới hạn Mô phỏng (Discussion & Simulation Boundaries)
Thiết kế của hệ thống kết hợp hiệu quả giữa semantic retrieval và hạ tầng phi tập trung. Multi-projection mở rộng không gian tìm kiếm theo chiều rộng, trong khi K-search mở rộng lân cận quanh từng key để giảm lỗi biên. Cơ chế placement key đảm bảo cân bằng giữa semantic locality và phân tán tải.

**Về Giới hạn Mô phỏng (Simulation Boundaries):**
Trong môi trường giả lập (code mô phỏng), hệ thống áp dụng một số kỹ thuật tối ưu hóa có chủ đích:
1.  **Mocking Tầng Metadata (Sổ Đỏ):** Để tối ưu thời gian mô phỏng, Tầng Metadata DHT dùng cho Exact Match (Nhánh 2) được giả lập thông qua một In-memory Hash Map toàn cục là `GLOBAL_METADATA_DHT`, cho tốc độ $O(1)$ thay vì $O(\log N)$. Việc dùng SHA-256 tìm Node bằng Kademlia là một công nghệ đã được chứng minh thực tế nên không cần tiêu tốn tài nguyên mô phỏng lại.
2.  **Tập trung vào Core Innovation:** Toàn bộ tài nguyên CPU của mô phỏng được dồn 100% cho việc chứng minh Nhánh 1 (LSH Semantic Routing, Ripple Search, Anti-Affinity, Spill-over Load Balancing) hoạt động mượt mà. 
3.  **Pseudo-reranking:** Do chất lượng nén PQ đạt độ chính xác rất cao, hệ thống mô phỏng chốt trực tiếp danh sách dựa trên điểm ADC tổng hợp mà không kích hoạt pha tải vector gốc để Reranking, nhằm tiết kiệm overhead I/O ảo. Việc Reranking bằng vector gốc sẽ là một tùy chọn trên môi trường Production.

