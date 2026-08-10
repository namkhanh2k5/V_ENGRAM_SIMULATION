#!/bin/bash
# ============================================================================
# TOÀN BỘ THÍ NGHIỆM VÒNG 3 — theo danh sách của thầy
#
#   tmux new -s v3
#   source venv/bin/activate
#   PARALLEL=4 bash run_round3.sh 2>&1 | tee round3.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation.py | grep -v grep | wc -l
#   bash run_round3.sh 2>&1 | grep -c skip          # số việc đã xong
#   tail -5 round3.log
#
# XEM KẾT QUẢ
#   bash run_round3.sh 2>&1 | tail -60
#
# ---------------------------------------------------------------------------
# SIMULATOR ĐÃ SỬA TRƯỚC KHI CHẠY (điều kiện tiên quyết của thầy):
#   - lookup TIÊU thời gian mô phỏng: mỗi vòng định tuyến một RTT
#   - 40 probe chạy SONG SONG, đường tới hạn = probe chậm nhất
#   Kiểm nhanh ở N=1500: 151,6 vòng -> 363ms; 600 vòng -> 928ms. Latency giờ
#   tỉ lệ với số vòng, trước đây thì không.
#
# BỐN NHÓM:
#   A. Rerun toàn bộ iterative results với simulator đã sửa
#   B. Termination ablation 2x2 + chẩn đoán cơ chế
#   C. Margin-ranked probing ablation
#   D. Bảng chi phí (có payload) — chỉ số liệu đường tới hạn mới có nghĩa
#
# Multi-probe LSH đã chạy xong ở lượt trước, không nằm trong script này.
#
# Ước tính với PARALLEL=4: A ~6h, B ~2,5h, C ~1,5h, D ~1,5h. TỔNG ~11,5 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000

$PY -c "import numpy, simpy" 2>/dev/null || {
    echo "THIẾU numpy/simpy — chạy: source venv/bin/activate"; exit 1; }
for v in STOP_RULE FRONTIER_SCOPE PROBE_ORDER MEASURE_OVERLAP; do
    grep -q "^$v" src/routing.py || { echo "src/routing.py thiếu $v"; exit 1; }
done
grep -q "_one_probe" src/network.py || {
    echo "src/network.py chưa sửa: thiếu 40 probe song song"; exit 1; }
grep -q "_probe_diag_summary" main_simulation.py || {
    echo "main_simulation.py chưa xuất chẩn đoán probe"; exit 1; }
echo "✓ venv OK, simulator đã sửa, đủ chẩn đoán"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

SEEDS_A="20235956 1 2 3 4 5 6 7 8 9"

run_mode() {
    local mode=$1 seed=$2 extra=$3
    local f="r3_A_${mode}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] A $mode s=$seed"; return; }
    env SKIP_PAYLOAD=1 ROUTING_MODE="$mode" $extra timeout 14400 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] A $mode s=$seed"
}
echo "-- semantic --"
for s in $SEEDS_A; do run_mode semantic "$s" "" & wait_slot; done; wait

MATCH=$($PY - <<'EOF'
import glob, re, statistics as st
v = []
for f in glob.glob('r3_A_semantic_s*.txt'):
    m = re.search(r'Unique nodes contacted\s+([\d,]+)', open(f, errors='ignore').read())
    if m: v.append(float(m.group(1).replace(',', '')))
print(int(round(st.mean(v))) if v else 504)
EOF
)
echo "   semantic chạm TB $MATCH node"
for m in keyed_lookup random_slots random_unique; do
    echo "-- $m --"
    ex=""; [ "$m" = "random_unique" ] && ex="MATCH_UNIQUE_NODES=$MATCH"
    for s in $SEEDS_A; do run_mode "$m" "$s" "$ex" & wait_slot; done; wait
done

# ------------------------------------------------------------------ B
echo ""
echo "########## B. TERMINATION ABLATION 2x2 + CHẨN ĐOÁN ##########"
run_term() {
    local sr=$1 fs=$2 seed=$3
    local f="r3_B_${sr}-${fs}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] B $sr/$fs s=$seed"; return; }
    env SKIP_PAYLOAD=1 STOP_RULE=$sr FRONTIER_SCOPE=$fs MEASURE_OVERLAP=1 \
        timeout 14400 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$seed" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 500 \
        > "$f" 2>&1 || echo "  [LỖI] B $sr/$fs s=$seed"
}
for s in 20235956 1 2; do
    for sr in stable exhaust; do
        for fs in all topk; do run_term "$sr" "$fs" "$s" & wait_slot; done
    done
done
wait

# ------------------------------------------------------------------ C
echo ""
echo "########## C. MARGIN-RANKED PROBING ABLATION ##########"
run_probe() {
    local po=$1 seed=$2
    local f="r3_C_${po}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] C $po s=$seed"; return; }
    env SKIP_PAYLOAD=1 PROBE_ORDER=$po timeout 14400 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] C $po s=$seed"
}
for s in 20235956 1 2 3 4; do
    for po in margin random; do run_probe "$po" "$s" & wait_slot; done
done
wait

# ------------------------------------------------------------------ D
echo ""
echo "########## D. BẢNG CHI PHÍ (có payload) ##########"
for s in 20235956 1 2; do
    f="r3_D_cost_s${s}.txt"
    [ -s "$f" ] && grep -q "BẢNG CHI PHÍ" "$f" && { echo "  [skip] D s=$s"; continue; }
    echo "  cost s=$s ..."
    PLACEMENT_MODE=deterministic PLACEMENT_K=20 FETCH_TOP=1 PARALLEL_ADC=1 \
        R_MAX=20 timeout 14400 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$s" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 100 \
        > "$f" 2>&1 || echo "  [LỖI] D s=$s"
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY analyze_round3.py