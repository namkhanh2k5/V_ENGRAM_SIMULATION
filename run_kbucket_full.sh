#!/bin/bash
# ============================================================================
# RERUN TOÀN BỘ VỚI K-BUCKET
#
#   tmux new -s kbfull
#   source venv/bin/activate
#   PARALLEL=4 bash run_kbucket_full.sh 2>&1 | tee kbfull.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation.py | grep -v grep | wc -l
#   bash check_kb.sh
#   tail -5 kbfull.log
#
# XEM KẾT QUẢ
#   python3 analyze_kbucket_full.py
#
# ---------------------------------------------------------------------------
# VÌ SAO: mọi số iterative trong bài đo bằng small-world ring, nhưng bài giờ mô
# tả k-bucket Kademlia. Đo ở N=10.000 qua ba seed chung cho thấy k-bucket hơn
# 2,5 điểm Recall@5 (ghép cặp t=6,7) và ít hơn 13% message, nên rerun không chỉ
# để nhất quán mà còn cho số tốt hơn.
#
# BỐN NHÓM, tất cả đặt ROUTING_TABLE=kbucket:
#   A. headline 4 chế độ x 10 seed              ~5 giờ
#   B. termination ablation 4 cấu hình x 5 seed ~3 giờ
#   C. margin ablation 2 x 5 seed               ~1,5 giờ
#   D. bảng chi phí 3 seed, có payload          ~1,5 giờ
#   TỔNG 73 lần chạy, khoảng 8-10 giờ với PARALLEL=4.
#
# Các sweep dùng mô hình closest-peer lý tưởng hoá (main_simulation_v2.py) KHÔNG
# bị ảnh hưởng: chúng không đi walk nên không dùng bảng định tuyến.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
export ROUTING_TABLE=kbucket

$PY -c "import numpy, simpy" 2>/dev/null || {
    echo "THIẾU numpy/simpy — chạy: source venv/bin/activate"; exit 1; }
grep -q "ROUTING_TABLE" src/network.py || {
    echo "src/network.py chưa có k-bucket — git pull rồi chạy lại"; exit 1; }
grep -q "kb_closest" src/node.py || {
    echo "src/node.py chưa có kb_closest — git pull rồi chạy lại"; exit 1; }
echo "✓ venv OK, k-bucket có sẵn, ROUTING_TABLE=$ROUTING_TABLE"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

# ------------------------------------------------------------------ A
echo ""
echo "########## A. HEADLINE, 4 CHẾ ĐỘ x 10 SEED ##########"
SEEDS_A="20235956 1 2 3 4 5 6 7 8 9"

run_a() {
    local mode=$1 seed=$2 extra=$3
    local f="kbf_A_${mode}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] A $mode s=$seed"; return; }
    env SKIP_PAYLOAD=1 ROUTING_MODE="$mode" $extra timeout 14400 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] A $mode s=$seed"
}

echo "-- semantic --"
for s in $SEEDS_A; do run_a semantic "$s" "" & wait_slot; done; wait

MATCH=$($PY - <<'EOF'
import glob, re, statistics as st
v = []
for f in glob.glob('kbf_A_semantic_s*.txt'):
    m = re.search(r'Unique nodes contacted\s+([\d,]+)', open(f, errors='ignore').read())
    if m:
        v.append(float(m.group(1).replace(',', '')))
print(int(round(st.mean(v))) if v else 504)
EOF
)
echo "   semantic chạm TB $MATCH peer"

for m in keyed_lookup random_slots random_unique; do
    echo "-- $m --"
    ex=""; [ "$m" = "random_unique" ] && ex="MATCH_UNIQUE_NODES=$MATCH"
    for s in $SEEDS_A; do run_a "$m" "$s" "$ex" & wait_slot; done; wait
done

# ------------------------------------------------------------------ B
echo ""
echo "########## B. TERMINATION ABLATION, 4 CẤU HÌNH x 5 SEED ##########"
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
for s in 20235956 1 2 3 4; do
    for sr in stable exhaust; do
        for fs in all topk; do run_b "$sr" "$fs" "$s" & wait_slot; done
    done
done
wait

# ------------------------------------------------------------------ C
echo ""
echo "########## C. MARGIN ABLATION, 2 x 5 SEED ##########"
run_c() {
    local po=$1 seed=$2
    local f="kbf_C_${po}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] C $po s=$seed"; return; }
    env SKIP_PAYLOAD=1 PROBE_ORDER=$po timeout 14400 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] C $po s=$seed"
}
for s in 20235956 1 2 3 4; do
    for po in margin random; do run_c "$po" "$s" & wait_slot; done
done
wait

# ------------------------------------------------------------------ D
echo ""
echo "########## D. BẢNG CHI PHÍ, 3 SEED CÓ PAYLOAD ##########"
for s in 20235956 1 2; do
    f="kbf_D_cost_s${s}.txt"
    [ -s "$f" ] && grep -q "BẢNG CHI PHÍ" "$f" && { echo "  [skip] D s=$s"; continue; }
    echo "  cost s=$s ..."
    PLACEMENT_MODE=deterministic PLACEMENT_K=20 FETCH_TOP=1 PARALLEL_ADC=1 \
        R_MAX=20 timeout 14400 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$s" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 100 \
        > "$f" 2>&1 || echo "  [LỖI] D s=$s"
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY analyze_kbucket_full.py
