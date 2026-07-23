#!/bin/bash
# ============================================================================
# MỤC 16 — METADATA AVAILABILITY DƯỚI NODE FAILURE (thầy đánh dấu "Bắt buộc")
#
#   nohup bash run_failure_muc16.sh > muc16.log 2>&1 &
#
# Thầy chỉ ra: thí nghiệm cũ chỉ kiểm payload và GIẢ ĐỊNH metadata luôn sẵn có,
# trong khi cấu hình chính dùng r=1. Cần quét r ∈ {1,2,3} × loss ∈ {0,10,20,30}%
# và báo cáo đủ 5 đại lượng.
#
# Câu hỏi khoa học nối thẳng vào mục 14: mục 14 cho thấy r là gánh nặng thuần
# tuý (không tăng recall, phá tỉ lệ). Thứ duy nhất r mua là ĐỘ BỀN. Vậy nó mua
# được bao nhiêu? Nếu r=1 đã trụ được ở 30% node chết thì lập luận khép kín:
# không có lý do gì tăng r.
#
# PHẦN A (nhanh, main_simulation_v2.py): Reachable/Final Recall@5 dưới node loss
# PHẦN B (chậm, main_churn_test.py): Metadata availability + Payload recovery
#
# Ước tính: A = 3r × 4loss × 5seed × 2mode = 120 lần × 65s ≈ 2,2 giờ
#           B = 3r × 3seed × ~5ph ≈ 45 phút
# ============================================================================
set -u
SEEDS="20235956 1 2 3 4"
NQ=500
PY=python3

runA() {
    local r=$1 loss=$2 seed=$3 mode=$4
    local sfx=""
    [ "$mode" = "random_slots" ] && sfx="_RANDOM"
    local lsfx=""
    [ "$loss" != "0" ] && lsfx="_loss${loss}"
    local f="result_code_N10000_L5_K20_MA${r}_T8_m512${sfx}${lsfx}_s${seed}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] r=$r loss=$loss s=$seed $mode"; return; fi
    $PY main_simulation_v2.py --dataset code --nodes 10000 --nq $NQ \
        --num-tables 5 --k-query 20 --meta-anchors "$r" --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$seed" --routing "$mode" \
        --node-loss "$loss" >/dev/null 2>&1 \
        || echo "  [LỖI] r=$r loss=$loss s=$seed $mode"
}

echo "##### PHẦN A — RECALL DƯỚI NODE LOSS (nhanh) #####"
for s in $SEEDS; do
  for r in 1 2 3; do
    for loss in 0 0.1 0.2 0.3; do
      echo "[A] r=$r loss=$loss s=$s"
      runA "$r" "$loss" "$s" semantic
      runA "$r" "$loss" "$s" random_slots
    done
  done
done

echo ""
echo "##### PHẦN B — METADATA/PAYLOAD AVAILABILITY (chậm) #####"
# main_churn_test.py đo 3 tầng tách bạch bằng ĐỊNH TUYẾN THẬT tới anchor còn sống
for r in 1 2 3; do
  for s in 2026 1 2; do
    f="failure_code_r${r}_s${s}.txt"
    if [ -s "$f" ]; then echo "  [skip] $f"; continue; fi
    echo "[B] r=$r s=$s"
    META_ANCHORS=$r DATASET=code NUM_FILES=20000 SAMPLE_SEED=$s \
      CHURN_STEPS="0,0.10,0.20,0.30" $PY main_churn_test.py > "$f" 2>&1 \
      || echo "  [LỖI] churn r=$r s=$s"
  done
done

echo ""
echo "##### TỔNG HỢP #####"
MIN_NQ=$NQ $PY summarize.py > muc16_recall.txt 2>&1
$PY analyze_failure.py 2>&1 | tee muc16_failure.txt
echo ""
echo "-> muc16_recall.txt   (bảng recall đầy đủ)"
echo "-> muc16_failure.txt  (bảng 5 đại lượng theo r × loss)"
echo "-> failure_code_r*_s*.txt (log churn thô)"