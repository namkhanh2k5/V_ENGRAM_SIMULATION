#!/bin/bash
# ============================================================================
# MỤC 1 — BASELINE CÔNG BẰNG (nhận xét thầy, đánh dấu "Bắt buộc")
#
#   nohup bash run_baselines_muc1.sh > muc1.log 2>&1 &
#
# Thầy chỉ ra: semantic chạm 4,9% node còn random chạm 8,0% — hai bên chỉ cùng
# số contact SLOT danh nghĩa (L*K*T), KHÔNG cùng số node unique. Và random hiện
# tại giả định client chọn được node bất kỳ dù không có global membership view.
#
# Script chạy 4 chế độ trên cùng cấu hình để so công bằng:
#   semantic       — dùng semantic key
#   random_slots   — L*K*T node ngẫu nhiên unique (oracle uniform-node sampling)
#   random_unique  — ĐÚNG số node unique mà semantic chạm cho chính query đó
#   random_keys    — L*T key ngẫu nhiên + Kademlia lookup (không cần global view,
#                    trả đúng chi phí routing, node gom theo cụm)
#
# Ước tính: 4 mode × 10 seed × 2 corpus = 80 lần chạy × ~65s ≈ 1,5 giờ
# ============================================================================
set -u
SEEDS="20235956 1 2 3 4 5 6 7 8 9"
NQ=500
PY=python3

# Bỏ qua nếu file đã có -> ngắt giữa chừng chạy lại vẫn tiếp tục
run() {
    local ds=$1 seed=$2 mode=$3
    local sfx=""
    case "$mode" in
        random_slots)  sfx="_RANDOM"   ;;
        random_unique) sfx="_RANDUNIQ" ;;
        random_keys)   sfx="_RANDKEY"  ;;
    esac
    local f="result_${ds}_N10000_L5_K20_MA1_T8_m512${sfx}_s${seed}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] $f"; return; fi
    $PY main_simulation_v2.py --dataset "$ds" --nodes 10000 --nq $NQ \
        --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$seed" --routing "$mode" \
        >/dev/null 2>&1 || echo "  [LỖI] $ds s=$seed $mode"
}

for s in $SEEDS; do
  for ds in code scifact; do
    echo "[$ds s=$s]"
    for m in semantic random_slots random_unique random_keys; do
        run "$ds" "$s" "$m"
    done
  done
done

echo ""
echo "##### TỔNG HỢP #####"
MIN_NQ=$NQ $PY summarize.py > muc1_baselines.txt 2>&1
echo "-> muc1_baselines.txt"
echo ""
echo "--- Bảng 4 chế độ (code, N=10000) ---"
grep -E "^code +10000 +5 +20 +1 +8" muc1_baselines.txt 2>/dev/null
echo ""
echo "CÁCH ĐỌC:"
echo "  cột node% của rnd-uniq PHẢI bằng semantic (đó là định nghĩa của nó)."
echo "  Nếu semantic vẫn thắng rnd-uniq và rnd-key => lợi thế KHÔNG đến từ việc"
echo "  chạm nhiều node hơn, mà từ chính semantic key. Đó là điều mục 1 hỏi."