#!/bin/bash
# ============================================================================
# CHẠY LẠI SAU BỐN BẢN VÁ (mục 1, 2, 3, 6 trong danh sách của thầy)
#
#   tmux new -s join
#   source venv/bin/activate
#   PARALLEL=4 bash run_join_bootstrap.sh 2>&1 | tee join.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation.py | grep -v grep | wc -l
#   bash check_join.sh
#
# XEM KẾT QUẢ
#   python3 analyze_join.py
#
# ---------------------------------------------------------------------------
# VÌ SAO: thầy hỏi ở mục 3 "Có dùng global membership để xây routing table hay
# không". Kiểm code thì CÓ, và nặng hơn tưởng.
#
# Bootstrap k-bucket cũ cấp cho mỗi peer sẵn K_BUCKET peer XOR-gần nhất TOÀN
# CỤC. Đó là oracle, và nó làm lookup hội tụ hoàn hảo: XOR rank trung bình ra
# đúng 9,5, tức mean của [0..19] — walk tìm ĐÚNG global top-20 không sai một
# peer nào. Đó chính là mâu thuẫn thầy chỉ ra ở mục 1: con số 9,5 không phải
# phép đo chất lượng định tuyến mà là dấu hiệu bootstrap đã cho sẵn đáp án.
#
# Đã cài BOOTSTRAP=join: peer gia nhập TUẦN TỰ, mỗi peer tra chính node_id của
# mình bằng lookup THẬT trên trạng thái định tuyến hiện có, học hai chiều với
# các peer nó gặp, rồi một vòng refresh sau khi mạng đủ. Không peer nào thấy
# danh sách toàn cục.
#
# Đo thử ở N=1.500, nq=10:
#     oracle: 119 contact/peer, XOR rank 9,50, Recall 80,0%
#     join  :  42 contact/peer, XOR rank 12,27, Recall 80,0%
# XOR rank khác rõ, xác nhận 9,5 là artifact. Recall ở quy mô nhỏ chưa khác,
# phải đo ở quy mô thật.
#
# BỐN NHÓM, tất cả BOOTSTRAP=join:
#   A. headline 4 chế độ x 10 seed
#   B. termination ablation 4 cấu hình x 10 seed
#   C. margin ablation 2 x 10 seed
#   D. bảng chi phí 3 seed có payload
#
# Ước tính với PARALLEL=4: 103 lần chạy, ~12 giờ. Bootstrap join chậm hơn
# oracle vì phải chạy N lookup thật lúc dựng mạng.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
export ROUTING_TABLE=kbucket
export BOOTSTRAP=join
export SHARED_ORIGIN=1      # mọi probe của một query dùng chung origin (mục 2.2)
export NORMALIZE_ROWS=1     # L2-normalize cột ma trận chiếu (mục 2.1)
export PQ_VARIANT=m512      # KHỚP BÀI. Trước đây main_simulation.py hardcode
                            # bản không suffix (m=256) trong khi các sweep chạy
                            # m=512 — hai nửa bài dùng hai quantizer khác nhau.

$PY -c "import numpy, simpy" 2>/dev/null || {
    echo "THIẾU numpy/simpy — chạy: source venv/bin/activate"; exit 1; }
grep -q 'BOOTSTRAP' src/network.py || {
    echo "src/network.py chưa có cờ BOOTSTRAP — git pull rồi chạy lại"; exit 1; }
grep -q 'joined = \[network_nodes\[0\]\]' src/network.py || {
    echo "src/network.py chưa có nhánh join tuần tự — git pull"; exit 1; }
grep -q 'SHARED_ORIGIN' src/network.py || {
    echo "src/network.py chưa có SHARED_ORIGIN — git pull"; exit 1; }
grep -q 'NORMALIZE_ROWS' src/routing.py || {
    echo "src/routing.py chưa có NORMALIZE_ROWS — git pull"; exit 1; }
grep -q 'rpcs_routing' src/network.py || {
    echo "src/network.py chưa tách routing/eval RPC — git pull"; exit 1; }
grep -q 'PQ_VARIANT' main_simulation.py || {
    echo "main_simulation.py chưa có cờ PQ_VARIANT — git pull"; exit 1; }
[ -f "data/code_pq_codebook_m512.npy" ] || {
    echo "thiếu data/code_pq_codebook_m512.npy"; exit 1; }
echo "✓ venv OK, ROUTING_TABLE=$ROUTING_TABLE BOOTSTRAP=$BOOTSTRAP"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }
SEEDS="20235956 1 2 3 4 5 6 7 8 9"

# ------------------------------------------------------------------ A
echo ""
echo "########## A. HEADLINE ##########"
run_a() {
    local mode=$1 seed=$2 extra=$3
    local f="q5_A_${mode}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] A $mode s=$seed"; return; }
    env SKIP_PAYLOAD=1 ROUTING_MODE="$mode" $extra timeout 21600 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] A $mode s=$seed"
}
echo "-- semantic --"
for s in $SEEDS; do run_a semantic "$s" "" & wait_slot; done; wait

MATCH=$($PY - <<'EOF'
import glob, re, statistics as st
v = []
for f in glob.glob('q5_A_semantic_s*.txt'):
    m = re.search(r'Unique nodes contacted\s+([\d,]+)', open(f, errors='ignore').read())
    if m:
        v.append(float(m.group(1).replace(',', '')))
print(int(round(st.mean(v))) if v else 500)
EOF
)
echo "   semantic chạm TB $MATCH peer"
for m in keyed_lookup random_slots random_unique; do
    echo "-- $m --"
    ex=""; [ "$m" = "random_unique" ] && ex="MATCH_UNIQUE_NODES=$MATCH"
    for s in $SEEDS; do run_a "$m" "$s" "$ex" & wait_slot; done; wait
done

# ------------------------------------------------------------------ B
echo ""
echo "########## B. TERMINATION ABLATION ##########"
run_b() {
    local sr=$1 fs=$2 seed=$3
    local f="q5_B_${sr}-${fs}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] B $sr/$fs s=$seed"; return; }
    env SKIP_PAYLOAD=1 STOP_RULE=$sr FRONTIER_SCOPE=$fs MEASURE_OVERLAP=1 \
        timeout 21600 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$seed" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 500 \
        --out "q5_termabl_${sr}-${fs}_s${seed}.json" \
        > "$f" 2>&1 || echo "  [LỖI] B $sr/$fs s=$seed"
}
for s in $SEEDS; do
    for sr in stable exhaust; do
        for fs in all topk; do run_b "$sr" "$fs" "$s" & wait_slot; done
    done
done
wait

# ------------------------------------------------------------------ C
echo ""
echo "########## C. MARGIN ABLATION ##########"
run_c() {
    local po=$1 seed=$2
    local f="q5_C_${po}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] C $po s=$seed"; return; }
    env SKIP_PAYLOAD=1 PROBE_ORDER=$po timeout 21600 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] C $po s=$seed"
}
for s in $SEEDS; do
    for po in margin random; do run_c "$po" "$s" & wait_slot; done
done
wait

# ------------------------------------------------------------------ D
echo ""
echo "########## D. BẢNG CHI PHÍ ##########"
for s in 20235956 1 2; do
    f="q5_D_cost_s${s}.txt"
    [ -s "$f" ] && grep -q "BẢNG CHI PHÍ" "$f" && { echo "  [skip] D s=$s"; continue; }
    echo "  cost s=$s ..."
    PLACEMENT_MODE=deterministic PLACEMENT_K=20 FETCH_TOP=1 PARALLEL_ADC=1 \
        R_MAX=20 timeout 21600 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$s" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 100 \
        > "$f" 2>&1 || echo "  [LỖI] D s=$s"
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY analyze_join.py