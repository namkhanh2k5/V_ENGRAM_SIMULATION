#!/bin/bash
# ============================================================================
# CHẠY QUA ĐÊM — mọi thứ còn thiếu
#
#   tmux new -s overnight
#   source venv/bin/activate
#   bash run_overnight.sh 2>&1 | tee overnight.log
#   # Ctrl+B rồi D để rời
#
# Xếp theo mức quan trọng: việc chặn kết luận chạy TRƯỚC, việc củng cố chạy SAU.
# Mọi phần đều có skip logic, ngắt giữa chừng chạy lại vẫn tiếp tục chỗ dở.
#
# TỔNG ƯỚC TÍNH: ~6 giờ
#   A. Kiểm chứng walk thật, nq=500   ~5,0 giờ   <- việc chính
#   B. Tương quan bảng chiếu, 5 seed  ~0,3 giờ
#   C. Zipf bù lên 5 seed             ~0,4 giờ
#   D. Chạy lại toàn bộ phân tích     ~0,1 giờ
# ============================================================================
set -u
PY=python3
DS=code
N=10000

echo "############################################################"
echo "# A. KIỂM CHỨNG WALK KADEMLIA THẬT (nq=500)"
echo "############################################################"
echo "# Lần chạy trước ở nq=20 cho KTC ±8 điểm — không phân giải nổi hiệu ứng"
echo "# ~3 điểm cần đo, và có dòng walk thật CAO HƠN lý tưởng (bất khả về cơ"
echo "# chế). nq=500 đưa KTC về ±1,6 điểm và khớp đúng số query của sweep."
echo ""

NQ_VAL=500
SEED_VAL=20235956

run_val() {
    local L=$1 r=$2 K=$3 T=$4 mode=$5
    local sfx=""; [ "$mode" = "random" ] && sfx="_RAND"
    local f="validate_L${L}_r${r}_K${K}_T${T}${sfx}.txt"
    if [ -s "$f" ] && grep -q "Recall@5" "$f"; then
        echo "  [skip] $f"; return
    fi
    local extra=""; [ "$mode" = "random" ] && extra="--random-routing"
    echo "  L=$L r=$r K=$K T=$T $mode ..."
    SKIP_PAYLOAD=1 NUM_TABLES=$L timeout 5400 $PY main_simulation.py \
        --dataset $DS --nodes $N --seed $SEED_VAL --k-query "$K" \
        --multi-probe "$T" --meta-anchors "$r" --nq $NQ_VAL $extra \
        > "$f" 2>&1 || echo "  [BỎ QUA] quá 90 phút"
}

# ba cấu hình r* trước — nếu hết giờ vẫn đủ kiểm luận điểm chính
for cfg in "5 1 20 8" "5 3 20 8" "8 5 20 8" "12 1 20 8" "5 1 20 1" "5 1 100 1"; do
    set -- $cfg
    echo ""
    echo "##### L=$1 r=$2 K=$3 T=$4 #####"
    run_val "$1" "$2" "$3" "$4" semantic
    run_val "$1" "$2" "$3" "$4" random
done

echo ""
echo "############################################################"
echo "# B. TƯƠNG QUAN GIỮA CÁC BẢNG CHIẾU — bù lên 5 seed"
echo "############################################################"
echo "# Bảng trong bài hiện chỉ có 2 seed. Với đại lượng dùng để kết luận"
echo "# 'các bảng độc lập' thì 2 seed là mỏng."
echo ""
for s in 20235956 1 2 3 4; do
    f="pertable_s${s}.txt"
    if [ -s "$f" ]; then echo "  [skip] $f"; continue; fi
    echo "  seed $s ..."
    $PY main_simulation_v2.py --dataset $DS --nodes $N --seed "$s" \
        --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
        --use-pq --pq-variant m512 --nq 500 --per-table-stats \
        > "$f" 2>&1 || echo "  [LỖI] seed $s"
done

echo ""
echo "############################################################"
echo "# C. ZIPF — bù lên 5 seed"
echo "############################################################"
echo "# Mục 17 hiện 3 seed. Đại lượng tải là Gini, nhạy với mẫu."
echo ""
for s in 3 4; do
  for ds in code scifact; do
    for z in 0.8 1.0 1.2; do
      f="result_${ds}_N10000_L5_K20_MA1_T8_m512_zipf${z}_s${s}_nq500.json"
      if [ -f "$f" ]; then echo "  [skip] $ds zipf=$z s=$s"; continue; fi
      echo "  $ds zipf=$z s=$s ..."
      $PY main_simulation_v2.py --dataset "$ds" --nodes $N --nq 500 \
          --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
          --use-pq --pq-variant m512 --seed "$s" --zipf "$z" \
          >/dev/null 2>&1 || echo "  [LỖI] $ds zipf=$z s=$s"
    done
  done
done

echo ""
echo "############################################################"
echo "# D. CHẠY LẠI TOÀN BỘ PHÂN TÍCH"
echo "############################################################"
MIN_NQ=500 $PY summarize.py            > paper_tables.txt     2>&1; echo "-> paper_tables.txt"
$PY analyze_factorial.py               > muc14_matched.txt    2>&1; echo "-> muc14_matched.txt"
$PY analyze_failure.py                 > muc16_failure.txt    2>&1; echo "-> muc16_failure.txt"
$PY analyze_sweeps.py                  > muc17_21.txt         2>&1; echo "-> muc17_21.txt"
$PY analyze_weakscaling.py             > weakscaling.txt      2>&1; echo "-> weakscaling.txt"
$PY analyze_churn.py                   > churn_results.txt    2>&1; echo "-> churn_results.txt"
$PY analyze_validate.py                > validate_results.txt 2>&1; echo "-> validate_results.txt"

echo ""
echo "############################################################"
echo "# XONG. Đọc theo thứ tự:"
echo "#   validate_results.txt  <- QUAN TRỌNG NHẤT, kiểm mọi sweep"
echo "#   paper_tables.txt      <- bảng tổng hợp"
echo "#   churn_results.txt     <- churn"
echo "#   weakscaling.txt       <- weak scaling"
echo "#   muc14/16/17_*.txt     <- các mục nhận xét của thầy"
echo "############################################################"
echo ""
echo "--- Tương quan bảng chiếu (5 seed) ---"
grep -h "Trung bình p\|ĐO ĐƯỢC\|ĐỘC LẬP\|Tương quan cặp" pertable_s*.txt 2>/dev/null
