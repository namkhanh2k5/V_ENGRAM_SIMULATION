#!/bin/bash
# ============================================================================
# HAI SWEEP CÒN LẠI — điểm 4 và 5 trong TODO cuối của thầy
#
#   tmux new -s sweeps
#   source venv/bin/activate
#   PARALLEL=4 bash run_two_sweeps.sh 2>&1 | tee sweeps.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation_v2 | grep -v grep | wc -l
#   ls indep_*_s*.json 2>/dev/null | wc -l        # A, cần 20
#   ls repl_r*_s*.json 2>/dev/null | wc -l        # B, cần 40
#   ls fail_L*_s*.json 2>/dev/null | wc -l        # C, cần 40
#   tail -5 sweeps.log
#
# XEM KẾT QUẢ
#   python3 analyze_two_sweeps.py
#
# ---------------------------------------------------------------------------
# A. PROJECTION-INDEPENDENCE (điểm 4)
#    Chẩn đoán "các bảng có độc lập không" hiện chỉ dựa trên HAI lần rút ma
#    trận, nên chênh giữa OR đo được và OR nếu độc lập chưa biết là hiệu ứng
#    thật hay nhiễu của lần rút. Chạy 10 seed mỗi corpus và gộp lại.
#
#    Đã thêm xuất p_each, p_bar, or_observed, or_if_independent vào JSON —
#    trước đây chỉ in ra màn hình nên không gộp được.
#
# B. HEALTHY REPLICATION (điểm 5, phần 1)
#    Sweep r trong {1,2,3,4} ở overlay lành, hiện một seed. Chạy 10 seed để
#    báo cáo mean, SD và số seed.
#
# C. STATIC FAILURE (điểm 5, phần 2)
#    Sweep node loss {0, 10, 20, 30}% ở r=1, hiện một seed. Chạy 10 seed.
#
# Ba sweep này dùng main_simulation_v2.py (mô hình closest-peer lý tưởng hoá),
# KHÔNG đi walk, nên không bị ảnh hưởng bởi bảng định tuyến hay bootstrap.
# Chúng độc lập với các thay đổi ở phần iterative.
#
# Ước tính với PARALLEL=4: 20 + 40 + 40 = 100 lần chạy, ~2,5 giờ. Nhanh hơn
# phần iterative nhiều vì không mô phỏng từng message.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2 3 4 5 6 7 8 9"

$PY -c "import numpy" 2>/dev/null || {
    echo "THIẾU numpy — chạy: source venv/bin/activate"; exit 1; }
grep -q '_PT_STATS' main_simulation_v2.py || {
    echo "main_simulation_v2.py chưa xuất per-table stats — git pull"; exit 1; }
echo "✓ venv OK, v2 xuất per-table stats"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

# ------------------------------------------------------------------ A
echo ""
echo "########## A. PROJECTION-INDEPENDENCE, 10 SEED x 2 CORPUS ##########"
for ds in code scifact; do
    for s in $SEEDS; do
        f="indep_${ds}_s${s}.json"
        [ -f "$f" ] && { echo "  [skip] A $ds s=$s"; continue; }
        $PY main_simulation_v2.py --dataset "$ds" --nodes $N --nq 500 \
            --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
            --use-pq --pq-variant m512 --seed "$s" --per-table-stats \
            --out "$f" >/dev/null 2>&1 || echo "  [LỖI] A $ds s=$s"
        wait_slot
    done
done
wait

# ------------------------------------------------------------------ B
echo ""
echo "########## B. HEALTHY REPLICATION, r=1..4 x 10 SEED ##########"
for r in 1 2 3 4; do
    for s in $SEEDS; do
        f="repl_r${r}_s${s}.json"
        [ -f "$f" ] && { echo "  [skip] B r=$r s=$s"; continue; }
        $PY main_simulation_v2.py --dataset code --nodes $N --nq 500 \
            --num-tables 5 --k-query 20 --meta-anchors "$r" --multi-probe 8 \
            --use-pq --pq-variant m512 --seed "$s" \
            --out "$f" >/dev/null 2>&1 || echo "  [LỖI] B r=$r s=$s"
        wait_slot
    done
done
wait

# ------------------------------------------------------------------ C
echo ""
echo "########## C. STATIC FAILURE, loss 0/10/20/30% x 10 SEED ##########"
for L in 0 10 20 30; do
    for s in $SEEDS; do
        f="fail_L${L}_s${s}.json"
        [ -f "$f" ] && { echo "  [skip] C loss=$L s=$s"; continue; }
        $PY main_simulation_v2.py --dataset code --nodes $N --nq 500 \
            --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
            --use-pq --pq-variant m512 --seed "$s" --node-loss "0.${L}" \
            --out "$f" >/dev/null 2>&1 || echo "  [LỖI] C loss=$L s=$s"
        wait_slot
    done
done
wait

echo ""
echo "########## TỔNG HỢP ##########"
$PY analyze_two_sweeps.py
