#!/bin/bash
# ============================================================================
# NÂNG TERMINATION VÀ MARGIN ABLATION LÊN MƯỜI SEED
#
#   tmux new -s ten
#   source venv/bin/activate
#   PARALLEL=4 bash run_ten_seeds.sh 2>&1 | tee ten.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation.py | grep -v grep | wc -l
#   echo "B: $(grep -l 'Recall@5' kbf_B_*.txt 2>/dev/null | wc -l)/40"
#   echo "C: $(grep -l 'Recall@5' kbf_C_*.txt 2>/dev/null | wc -l)/20"
#   tail -5 ten.log
#
# XEM KẾT QUẢ
#   python3 analyze_ten_seeds.py
#
# ---------------------------------------------------------------------------
# VÌ SAO: headline báo 76,1 (mười seed) còn bảng termination báo 76,3 (năm seed)
# ở CÙNG cấu hình. Bài có ghi rõ số seed từng chỗ nhưng hai con số khác nhau cho
# cùng một thứ vẫn dễ bị hỏi — đúng kiểu thầy đã bắt với 80,0 và 80,9.
#
# KÈM MỘT PHÉP KIỂM CHÉO: nhóm A chạy với ROUTING_MODE=semantic còn nhóm B chạy
# mặc định "auto" và bật MEASURE_OVERLAP. Nếu auto giải ra semantic và
# MEASURE_OVERLAP chỉ ĐO thêm chứ không đổi hành vi walk, thì sau khi cùng mười
# seed, stable/all PHẢI bằng headline semantic. Lệch nhau là dấu hiệu có gì đó
# khác giữa hai đường chạy mà ta chưa nhận ra.
#
# Chỉ chạy 5 seed CÒN THIẾU (5..9); 5 seed cũ đã có nên script skip.
#   B: 4 cấu hình x 5 seed mới = 20 lần
#   C: 2 chế độ  x 5 seed mới = 10 lần
# Ước tính với PARALLEL=4: ~3,5 giờ (nhóm B chậm hơn vì MEASURE_OVERLAP phải
# sắp toàn mạng mỗi lookup).
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
NEW_SEEDS="5 6 7 8 9"
export ROUTING_TABLE=kbucket

$PY -c "import numpy, simpy" 2>/dev/null || {
    echo "THIẾU numpy/simpy — chạy: source venv/bin/activate"; exit 1; }
grep -q "ROUTING_TABLE" src/network.py || {
    echo "src/network.py chưa có k-bucket — git pull rồi chạy lại"; exit 1; }
ls kbf_A_semantic_s9.txt >/dev/null 2>&1 || {
    echo "CẢNH BÁO: chưa thấy nhóm A mười seed. Chạy run_kbucket_full.sh trước."; }
echo "✓ venv OK, ROUTING_TABLE=$ROUTING_TABLE"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

echo ""
echo "########## B. TERMINATION, 5 SEED CÒN THIẾU ##########"
run_b() {
    local sr=$1 fs=$2 seed=$3
    local f="kbf_B_${sr}-${fs}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] B $sr/$fs s=$seed"; return; }
    env SKIP_PAYLOAD=1 STOP_RULE=$sr FRONTIER_SCOPE=$fs MEASURE_OVERLAP=1 \
        timeout 14400 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$seed" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 500 \
        --out "kbf_termabl_${sr}-${fs}_s${seed}.json" \
        > "$f" 2>&1 || echo "  [LỖI] B $sr/$fs s=$seed"
}
for s in $NEW_SEEDS; do
    for sr in stable exhaust; do
        for fs in all topk; do run_b "$sr" "$fs" "$s" & wait_slot; done
    done
done
wait

echo ""
echo "########## C. MARGIN ABLATION, 5 SEED CÒN THIẾU ##########"
run_c() {
    local po=$1 seed=$2
    local f="kbf_C_${po}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] C $po s=$seed"; return; }
    env SKIP_PAYLOAD=1 PROBE_ORDER=$po timeout 14400 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] C $po s=$seed"
}
for s in $NEW_SEEDS; do
    for po in margin random; do run_c "$po" "$s" & wait_slot; done
done
wait

echo ""
echo "########## TỔNG HỢP ##########"
$PY analyze_ten_seeds.py
