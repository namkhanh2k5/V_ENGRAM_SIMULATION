#!/bin/bash
# ============================================================================
# MC10 LỐI A — đo bốn biến thể đặt/lấy payload ở QUY MÔ THẬT
#
#   tmux new -s mc10
#   source venv/bin/activate
#   bash run_mc10_payload.sh 2>&1 | tee mc10.log
#
# CÂU HỎI: payload chiếm 89% RPC. Nguyên nhân là gì, và sửa được bao nhiêu?
#
# Hai giả thuyết:
#   H1 — mất tính tất định: node nhận shard phụ thuộc thứ tự danh sách ứng viên
#        lúc GHI, mà lúc ĐỌC thứ tự khác, nên client phải dò dần.
#   H2 — over-provision: PLACEMENT_CANDIDATES=300 làm mỗi lookup đắt gấp 4 lần
#        lookup discovery (k=20), bất kể có dò hay không.
#
# Đo thử ở N=500 cho thấy độ sâu dò trung bình 1,04 ở CẢ HAI chế độ, tức H1
# KHÔNG bite ở quy mô đó và H2 mới là nguyên nhân. Nhưng ở N=500 thì k=300 là
# 60% mạng, chế độ khác hẳn N=10.000 (3%). Phải đo lại ở quy mô thật.
#
# BỐN BIẾN THỂ:
#   scan k=300 top=0          — hiện tại, mốc so
#   deterministic k=300 top=0 — cô lập ảnh hưởng của tính tất định
#   deterministic k=20 top=0  — cô lập ảnh hưởng của k
#   deterministic k=20 top=1  — thêm: chỉ lấy payload của kết quả đầu
#
# CHỈ SỐ QUYẾT ĐỊNH: probe_depth_mean.
#   ~1.0  -> H1 sai, giữ scan cũng được, chỉ cần hạ k
#   >>1.0 -> H1 đúng, phải đổi sang deterministic
#
# Ước tính: 4 lần chạy x 30-60 phút = 2-4 giờ (có payload nên chậm).
# ============================================================================
set -u
PY=python3
DS=code
N=10000
NQ=100          # đủ để đo chi phí; recall không phải mục tiêu ở đây

run() {
    local mode=$1 k=$2 top=$3
    local f="mc10_${mode}_k${k}_top${top}.txt"
    if [ -s "$f" ] && grep -q "BẢNG CHI PHÍ" "$f"; then
        echo "  [skip] $f"; return
    fi
    echo "  $mode k=$k top=$top (30-60 phút) ..."
    PLACEMENT_MODE=$mode PLACEMENT_K=$k FETCH_TOP=$top R_MAX=20 \
        timeout 10800 $PY main_simulation.py \
        --dataset $DS --nodes $N --seed 20235956 \
        --k-query 20 --multi-probe 8 --meta-anchors 1 --nq $NQ \
        > "$f" 2>&1 || echo "  [BỎ QUA] quá 3 giờ"
}

echo "##### BỐN BIẾN THỂ #####"
run scan          300 0
run deterministic 300 0
run deterministic  20 0
run deterministic  20 1

echo ""
echo "##### TỔNG HỢP #####"
$PY analyze_mc10.py 2>&1 | tee mc10_results.txt
echo "-> mc10_results.txt"
