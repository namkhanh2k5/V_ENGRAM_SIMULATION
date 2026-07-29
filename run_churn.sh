#!/bin/bash
# ============================================================================
# THÍ NGHIỆM CHURN — sửa chữa thưa đến đâu thì hỏng?
#
#   nohup bash run_churn.sh > churn.log 2>&1 &
#
# SỬA THIẾT KẾ so với bản trước: bản trước quét median session {240,60,15} phút,
# nhưng buộc MỌI tham số thời gian (duration, epoch, warmup, repair) vào chính
# median session. Kết quả là mô phỏng KHÔNG THỨ NGUYÊN — ba mức cho kết quả y
# hệt vì chúng là MỘT thí nghiệm đo bằng ba đơn vị thời gian khác nhau.
#
# Trục thật sự có ý nghĩa: SỬA CHỮA THƯA HAY DÀY SO VỚI TỐC ĐỘ CHURN.
# Nên cố định median session và quét chu kỳ sửa chữa theo bội số của nó.
#
# Đây cũng là câu hỏi thực tế: IPFS republish mỗi 22 GIỜ trong mạng mà 87,6%
# session dưới 8 giờ — tức chu kỳ sửa DÀI HƠN session, và họ bù bằng r=20.
# Câu hỏi của paper: ở r=1, chu kỳ sửa phải ngắn đến mức nào?
#
# BẢY CẤU HÌNH:
#   r=1 sửa mỗi 15ph  (median/8)  — dày nhất
#   r=1 sửa mỗi 30ph  (median/4)
#   r=1 sửa mỗi 60ph  (median/2)
#   r=1 sửa mỗi 120ph (= median)
#   r=1 sửa mỗi 240/480/960/1440ph (2x/4x/8x/12x median)
#       1440ph phủ mức IPFS: republish 22 giờ = 11x median 120ph
#   r=1 KHÔNG sửa                 — đối chứng dưới
#   r=20 KHÔNG sửa                — lối IPFS, đối chứng trên
#
# median session = 120 phút (giữa Li et al. 60ph và IPFS ~8 giờ),
# thời lượng 720 phút = 6 lần thay lượt.
#
# Ước tính: 10 cấu hình × 3 seed = 30 lần chạy. Chu kỳ sửa dài cần thời
# lượng dài hơn (>= 3 lần chu kỳ), nên rep=1440 chạy 4320ph mô phỏng.
# Tổng ~7 giờ.
# ============================================================================
set -u
SEEDS="20235956 1 2"
DS=code
N=10000
NQ=200
SES=120           # median session, phút
DUR=720           # thời lượng = 6 x median
PY=python3

run() {
    local r=$1 rep=$2 seed=$3
    local m=""; [ "$rep" != "0" ] && m="l"
    # Thời lượng phải >= 3 lần chu kỳ sửa, nếu không sửa chữa chạy quá ít lần
    # (hoặc không chạy lần nào) và điểm đo vô nghĩa.
    local dur=$DUR
    local need=$((rep * 3))
    [ "$need" -gt "$dur" ] && dur=$need
    local f="churn_${DS}_N${N}_r${r}_ses${SES}_weibull_rep${rep}${m}_s${seed}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] r=$r rep=$rep s=$seed"; return; fi
    $PY main_churn_engine.py --dataset $DS --nodes $N --nq $NQ \
        --median-session $SES --duration "$dur" --session-dist weibull \
        --meta-anchors "$r" --repair-interval "$rep" --seed "$seed" \
        >/dev/null 2>&1 || echo "  [LỖI] r=$r rep=$rep s=$seed"
}

for s in $SEEDS; do
    echo ""
    echo "##### seed $s #####"
    # dải tới 1440ph = 12x median, phủ cả mức IPFS (republish 22 giờ)
    for rep in 15 30 60 120 240 480 960 1440; do
        echo "[s=$s] r=1 sửa mỗi ${rep}ph (median/$((SES/rep)))"
        run 1 "$rep" "$s"
    done
    echo "[s=$s] r=1 KHÔNG sửa" ; run 1  0 "$s"
    echo "[s=$s] r=20 KHÔNG sửa"; run 20 0 "$s"
done

echo ""
echo "##### TỔNG HỢP #####"
$PY analyze_churn.py 2>&1 | tee churn_results.txt
echo "-> churn_results.txt"