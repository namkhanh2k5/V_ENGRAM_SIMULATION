#!/bin/bash
# ============================================================================
# GỘP MỤC 17 + MỤC 21 (phần chạy được bằng v2) — đều nhanh, cùng công cụ
#
#   nohup bash run_sweeps_muc17_21.sh > muc17_21.log 2>&1 &
#
# MỤC 17 — hotspot dưới workload lệch:
#   Thầy yêu cầu sweep Zipf exponent {0, 0.8, 1.0, 1.2} và báo cáo P95/P99 RPC
#   load, chứ không chỉ kết luận "tolerable" dựa trên dung lượng lưu trữ.
#
# MỤC 21 — sweep tham số:
#   Thầy yêu cầu T ∈ {1,3,5,8,12}, K ∈ {10,20,40,80}, Rmax ∈ {5,10,15,20}.
#   T và K chạy được ở đây. R_max KHÔNG: main_simulation_v2.py dùng lookup lý
#   tưởng hoá (knn toàn cục), không có vòng lặp nào để cắt. Phần R_max và
#   "% query chạm R_max" phải chạy bằng main_simulation.py (SimPy) — để riêng.
#
# Ước tính: ~110 lần chạy mới × 65s ≈ 2 giờ
# ============================================================================
set -u
SEEDS="20235956 1 2 3 4"
NQ=500
PY=python3

# --- MỤC 17: Zipf sweep ---
# Chỉ 3 seed: phân bố tải do corpus + node_id quyết định, không do mẫu query.
echo "##### MỤC 17 — ZIPF SWEEP #####"
for s in 20235956 1 2; do
  for ds in code scifact; do
    for z in 0 0.8 1.0 1.2; do
      zs=""
      [ "$z" != "0" ] && zs="_zipf${z}"
      f="result_${ds}_N10000_L5_K20_MA1_T8_m512${zs}_s${s}_nq${NQ}.json"
      if [ -f "$f" ]; then echo "  [skip] $ds zipf=$z s=$s"; continue; fi
      echo "[17] $ds zipf=$z s=$s"
      $PY main_simulation_v2.py --dataset "$ds" --nodes 10000 --nq $NQ \
          --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
          --use-pq --pq-variant m512 --seed "$s" --zipf "$z" \
          >/dev/null 2>&1 || echo "  [LỖI] $ds zipf=$z s=$s"
    done
  done
done

# --- MỤC 21a: sweep T (K=20 cố định) ---
echo ""
echo "##### MỤC 21a — SWEEP T (K=20) #####"
for s in $SEEDS; do
  for T in 1 3 5 8 12; do
    f="result_code_N10000_L5_K20_MA1_T${T}_m512_s${s}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] T=$T s=$s"; continue; fi
    echo "[21a] T=$T s=$s"
    $PY main_simulation_v2.py --dataset code --nodes 10000 --nq $NQ \
        --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe "$T" \
        --use-pq --pq-variant m512 --seed "$s" \
        >/dev/null 2>&1 || echo "  [LỖI] T=$T s=$s"
  done
done

# --- MỤC 21b: sweep K (T=8 cố định) ---
echo ""
echo "##### MỤC 21b — SWEEP K (T=8) #####"
for s in $SEEDS; do
  for K in 10 20 40 80; do
    f="result_code_N10000_L5_K${K}_MA1_T8_m512_s${s}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] K=$K s=$s"; continue; fi
    echo "[21b] K=$K s=$s"
    $PY main_simulation_v2.py --dataset code --nodes 10000 --nq $NQ \
        --num-tables 5 --k-query "$K" --meta-anchors 1 --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$s" \
        >/dev/null 2>&1 || echo "  [LỖI] K=$K s=$s"
  done
done

echo ""
echo "##### TỔNG HỢP #####"
MIN_NQ=$NQ $PY summarize.py > muc17_21_full.txt 2>&1
$PY analyze_sweeps.py 2>&1 | tee muc17_21.txt
echo ""
echo "-> muc17_21.txt      (bảng Zipf + sweep T + sweep K)"
echo "-> muc17_21_full.txt (bảng tổng hợp đầy đủ)"
echo ""
echo "CHƯA LÀM (cần SimPy, để riêng):"
echo "  - sweep R_max ∈ {5,10,15,20} và % query chạm R_max"
echo "  - bảng chi phí đầy đủ mục 5 (rounds/RPC/bytes, p50/p95/p99)"