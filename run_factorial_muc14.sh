#!/bin/bash
# ============================================================================
# MỤC 14 — FACTORIAL GRID L × r (nhận xét thầy, "Bắt buộc")
#
#   nohup bash run_factorial_muc14.sh > muc14.log 2>&1 &
#
# Thầy chỉ ra: Bảng r* hiện tại chưa có đủ cặp cấu hình CÙNG L·r nhưng KHÁC
# cách phân tách, nên chưa kết luận được "tỉ lệ chỉ phụ thuộc tích L·r".
#
# Lưu ý quan trọng trước khi chạy: công thức
#     P_rand = 1 - (1 - L·r/N)^(K·L·T)
# có L ở CẢ cơ số (qua L·r) LẪN số mũ (qua K·L·T). Nên về lý thuyết, ở cùng
# L·r mà L khác nhau thì P_rand vẫn khác:
#     L·r=40: (L=4,r=10) -> 92.3% | (L=8,r=5) -> 99.4% | (L=10,r=4) -> 99.8%
# Thí nghiệm này nhiều khả năng BÁC BỎ dạng đơn giản của luận điểm, và cho ra
# một phát biểu chính xác hơn. Đó là mục đích.
#
# Grid: L ∈ {4,5,8,10} × r ∈ {1,2,4,5,10} = 20 cấu hình × 2 mode × 5 seed
# Các bộ CÙNG L·r để so trực tiếp:
#     L·r=8  : (4,2) (8,1)
#     L·r=10 : (5,2) (10,1)
#     L·r=16 : (4,4) (8,2)
#     L·r=20 : (4,5) (5,4) (10,2)      <- ba cách phân tách
#     L·r=40 : (4,10) (8,5) (10,4)     <- ba cách phân tách
#     L·r=50 : (5,10) (10,5)
#
# Ước tính: ~150 lần chạy mới (đã trừ cấu hình có sẵn) × 65s ≈ 2,7 giờ
# ============================================================================
set -u
SEEDS="20235956 1 2 3 4"      # 5 seed; cấu hình chốt đã có 10 seed từ trước
NQ=500
PY=python3

run() {
    local L=$1 r=$2 seed=$3 mode=$4
    local sfx=""
    [ "$mode" = "random_slots" ] && sfx="_RANDOM"
    local f="result_code_N10000_L${L}_K20_MA${r}_T8_m512${sfx}_s${seed}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] L=$L r=$r s=$seed $mode"; return; fi
    $PY main_simulation_v2.py --dataset code --nodes 10000 --nq $NQ \
        --num-tables "$L" --k-query 20 --meta-anchors "$r" --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$seed" --routing "$mode" \
        >/dev/null 2>&1 || echo "  [LỖI] L=$L r=$r s=$seed $mode"
}

for s in $SEEDS; do
  for L in 4 5 8 10; do
    for r in 1 2 4 5 10; do
      echo "[grid] L=$L r=$r (L·r=$((L*r))) s=$s"
      run "$L" "$r" "$s" semantic
      run "$L" "$r" "$s" random_slots
    done
  done
done

echo ""
echo "##### TỔNG HỢP #####"
MIN_NQ=$NQ $PY summarize.py > muc14_grid.txt 2>&1
echo "-> muc14_grid.txt"
echo ""
$PY analyze_factorial.py 2>&1 | tee muc14_matched.txt
echo "-> muc14_matched.txt (bảng so các cách phân tách cùng L·r)"