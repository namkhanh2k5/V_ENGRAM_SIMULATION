#!/bin/bash
# ============================================================================
# BA MỤC CÒN LẠI: 18 (baseline), 5 (bảng chi phí), 21 (R_max)
#
#   nohup bash run_final_muc5_18_21.sh > final.log 2>&1 &
#
# MỤC 18 — khôi phục bucket-LSH + thêm Random-C
#   bucket-LSH: cách kinh điển đặt LSH lên DHT (băm nhãn bucket b-bit, tra CHÍNH
#   XÁC). Nó MẤT tính khả duyệt: không đi sang bucket kề được. Quét b để thấy
#   đánh đổi recall vs kích thước pool.
#   Random-C: bốc C ứng viên ngẫu nhiên rồi rerank BẰNG CÙNG PQ, với C = số ứng
#   viên duy nhất V-Engram gom được. Tách giá trị của KHÂU CHỌN khỏi khâu rerank.
#
# MỤC 5 — bảng chi phí đầy đủ: Discovery / Payload / Total, gồm rounds, RPC,
#   bytes, unique nodes, candidate tags, unique candidates, và p50/p95/p99 latency.
#
# MỤC 21 (phần còn thiếu) — quét R_max ∈ {5,10,15,20} và đo % lookup chạm trần.
#
# Ước tính: mục 18 ~15 phút | mục 5+21 ~4 lần SimPy × 30-90 phút = 2-6 giờ
#   SimPy RẤT chậm ở 10.000 node. Dùng --nq 20. Nếu quá lâu, Ctrl+C bỏ phần G,
#   các mục khác đã xong vẫn dùng được.
# ============================================================================
set -u
PY=python3

echo "##### MỤC 18 — BUCKET-LSH + RANDOM-C #####"
# VENGRAM_CANDIDATES lấy từ bảng chính: code 22.6% × 20000 = 4510, scifact 35.6% × 5183 = 1845
for cfg in "code 4510" "scifact 1845"; do
    set -- $cfg; ds=$1; vc=$2
    f="muc18_${ds}.txt"
    if [ -s "$f" ]; then echo "  [skip] $f"; continue; fi
    echo "[18] $ds (VENGRAM_CANDIDATES=$vc)"
    CORPUS=$ds POOL_PER_TABLE=100 VENGRAM_CANDIDATES=$vc \
      BUCKET_WIDTHS=8,10,12,14,16 $PY baselines.py > "$f" 2>&1 \
      || echo "  [LỖI] baseline $ds"
done

echo ""
echo "##### MỤC 5 + 21 — CHI PHÍ VÀ R_MAX (SimPy, chậm) #####"
# Quét R_max. R_max=15 là giá trị trong Table 2 của bài.
for rmax in 5 10 15 20; do
    f="muc5_21_rmax${rmax}.txt"
    if [ -s "$f" ]; then echo "  [skip] $f"; continue; fi
    echo "[5+21] R_max=$rmax (có thể mất 30-90 phút)"
    R_MAX=$rmax timeout 7200 $PY main_simulation.py --dataset code --nodes 10000 \
        --seed 20235956 --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 20 \
        > "$f" 2>&1 || echo "  [BỎ QUA] R_max=$rmax quá 2 giờ"
done

echo ""
echo "##### TỔNG HỢP #####"
echo "--- Mục 18: quét bucket-LSH (code) ---"
sed -n '/QUÉT BUCKET-LSH/,/^$/p' muc18_code.txt 2>/dev/null

echo ""
echo "--- Mục 18: bảng baseline đầy đủ (code) ---"
grep -E "Brute-Force|HNSW|Multi-Table|Bucket-LSH|Crypto-DHT|Random-C|Random-5" \
     muc18_code.txt 2>/dev/null | grep "%"

echo ""
echo "--- Mục 5: bảng chi phí (R_max=15) ---"
sed -n '/BẢNG CHI PHÍ/,/^===/p' muc5_21_rmax15.txt 2>/dev/null

echo ""
echo "--- Mục 21: % lookup chạm R_max ---"
for rmax in 5 10 15 20; do
    [ -s "muc5_21_rmax${rmax}.txt" ] || continue
    echo "R_max=$rmax:"
    grep -A3 "MỤC 21" "muc5_21_rmax${rmax}.txt" | grep -E "chạm trần|Recall@5" | head -3
    grep "Recall@5" "muc5_21_rmax${rmax}.txt" | head -1
done