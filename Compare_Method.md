# Bao cao so sanh ky thuat: V-Engram va cac giai phap luu tru/tim kiem phi tap trung

## 1. So sanh voi cac giao thuc co so (Kademlia, IPFS, Filecoin)

| Tieu chi | Kademlia (nguyen ban) | IPFS / Filecoin | V-Engram (de xuat) |
| --- | --- | --- | --- |
| Dinh vi du lieu | Dua tren ma bam ID tuyet doi (exact match). | Dua tren CID sinh tu SHA-256. | Dua tren semantic key sinh tu AI embedding qua LSH. |
| Kha nang tim kiem | Chi tim duoc khi biet chinh xac ID. | "Mu ngu nghia", can CID chinh xac de truy xuat. | Ho tro tim kiem mo/ngu nghia tu nhien. |
| Phan bo shard | Khong co co che phan manh mac dinh. | Sharding theo Merkle DAG, phan tan ngau nhien. | Placement key (152/8-bit): giu lan can ngu nghia va can bang tai. |
| Cau truc node | Node trung lap (chi dinh tuyen). | Storage node (luu tru thuan tuy). | Compute-Storage node: RAM chay ADC, SSD luu payload. |
| Hieu ung mang | Tham lam, dam thang 1 duong toi dich. | Dam thang toi CID. | Ripple search: loang rong dong, tim nhieu ung vien. |

## 2. So sanh voi cac nghien cuu tien nhiem (state-of-the-art)

### 2.1 pSearch (2003)
- Diem giong: su dung LSH de anh xa du lieu vao khong gian mang.
- Han che: su dung LSI cu voi vector thua, can semantic overlay cong kenh de len P2P.
- Cai tien cua V-Engram: dung dense embedding 1024-dim tu LLM, tich hop truc tiep vao khoa Kademlia ban dia, giam do tre va tang tuong thich.

### 2.2 Graph Diffusion (2022)
- Diem giong: su dung co che lan truyen de tim kiem ket qua tuong dong.
- Han che: phu thuoc do thi, kho mo rong, khong co co che luu tru phan manh an toan.
- Cai tien cua V-Engram: giu O(log N) cua Kademlia, bo sung AES va Reed-Solomon de bao toan du lieu.

### 2.3 Semantica (2025)
- Diem giong: su dung embedding hien dai de tim kiem tren mang phan tan.
- Han che: luu vector goc dan den tran RAM node bien, thieu co che phan bo shard, gay nghen cuc bo.
- Cai tien cua V-Engram:
  - Toi uu RAM: nen vector 16 lan bang PQ.
  - Can bang tai: placement key + nhieu HMAC de rai 30 manh ra 30 node lan can.

## 3. Phan tich uu va nhuoc diem cua V-Engram

### 3.1 Uu diem dot pha
- AI-native routing: bien routing thanh tim kiem ngu nghia.
- Hieu qua tai nguyen: PQ/ADC cho phep search tren RAM nho, phu hop edge/IoT.
- Bao mat zero-knowledge: AES tai client, shard rai khap noi, node luu tru khong biet noi dung.
- He sinh thai Web3: tich hop smart contract cho truy xuat du lieu, tao thi truong tim kiem AI phi tap trung.

### 3.2 Nhuoc diem va thach thuc
- Cap nhat phuc tap: noi dung doi -> semantic key doi -> di doi 30 shard.
- Sai so luong tu hoa: PQ giam 4KB -> 256B, sai so cosine ~0.02, can reranking.
- Lexical gap: dense vector kho xu ly tim kiem exact keyword (ma phien ban, hang so).

## 4. Ket luan
V-Engram khong chi la mot giai phap luu tru moi. No la su giao thoa giua luu tru bat bien Web3 va truy van thong minh AI. So voi IPFS/Filecoin, V-Engram thong minh hon. So voi cac nghien cuu nhu Semantica hay pSearch, V-Engram thuc chien hon nho toi uu phan cung va can bang tai.