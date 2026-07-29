#!/bin/bash
# ============================================================================
# THÍ NGHIỆM CHURN — nhân bản hay sửa chữa?
#
#   nohup bash run_churn.sh > churn.log 2>&1 &
#
# CÂU HỎI: Mục r* chứng minh nhân bản rộng giết định tuyến ngữ nghĩa. Thứ duy
# nhất nhân bản mua được là ĐỘ BỀN. Vậy có cách nào mua độ bền mà không phá cơ
# chế không? Giả thuyết: sửa chữa (tái neo định kỳ) ở r=1.
#
# BỐN CẤU HÌNH dưới cùng một mô hình churn:
#   1. r=1,  không sửa        — đối chứng: cho biết churn phá đến đâu
#   2. r=1,  sửa mỗi median/4 — đề xuất của paper: nhân bản thấp, sửa nhanh
#   3. r=1,  sửa mỗi median   — sửa thưa hơn, xem ngưỡng nằm đâu
#   4. r=20, không sửa        — lối IPFS: nhân bản cao, sửa chậm
#
# QUÉT median session {240, 60, 15} phút:
#   240ph = 4 giờ   -> gần mạng thật (IPFS: 87,6% session dưới 8 giờ)
#    60ph = 1 giờ   -> Li et al., IPTPS'04
#    15ph           -> vùng stress của Bamboo (USENIX'04 quét 1,4-47 phút)
#
# N = 10.000 node. LƯU Ý: đừng chạy ở quy mô nhỏ hơn — ở 2.000 node, baseline
# ngẫu nhiên chạm 40% mạng nên tỉ lệ sem/rand bị nén về 1 và mất hết ý nghĩa.
#
# Ước tính: 4 cấu hình × 3 session × 3 seed = 36 lần chạy.
#   r=1  ~5 phút | r=20 ~10 phút  =>  tổng ~4 giờ
# ============================================================================
set -u
SEEDS="20235956 1 2"
DS=code
N=10000
NQ=200
PY=python3

run() {
    local ses=$1 r=$2 rep=$3 seed=$4
    local mode=""
    [ "$rep" != "0" ] && mode="l"
    local f="churn_${DS}_N${N}_r${r}_ses${ses}_weibull_rep${rep}${mode}_s${seed}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] ses=$ses r=$r rep=$rep s=$seed"; return; fi
    # thời lượng = 3 lần median session, đủ để mạng thay lượt vài lần
    local dur=$((ses * 3))
    $PY main_churn_engine.py --dataset $DS --nodes $N --nq $NQ \
        --median-session "$ses" --duration "$dur" --session-dist weibull \
        --meta-anchors "$r" --repair-interval "$rep" --seed "$seed" \
        >/dev/null 2>&1 || echo "  [LỖI] ses=$ses r=$r rep=$rep s=$seed"
}

for ses in 240 60 15; do
    q4=$((ses / 4))
    echo ""
    echo "##### median session = ${ses} phút #####"
    for s in $SEEDS; do
        echo "[s=$s] r=1 không sửa"        ; run "$ses" 1  0     "$s"
        echo "[s=$s] r=1 sửa mỗi ${q4}ph"  ; run "$ses" 1  "$q4" "$s"
        echo "[s=$s] r=1 sửa mỗi ${ses}ph" ; run "$ses" 1  "$ses" "$s"
        echo "[s=$s] r=20 không sửa"       ; run "$ses" 20 0     "$s"
    done
done

echo ""
echo "##### TỔNG HỢP #####"
$PY analyze_churn.py 2>&1 | tee churn_results.txt
echo ""
echo "-> churn_results.txt"
