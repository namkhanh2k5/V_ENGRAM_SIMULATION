#!/bin/bash
# ============================================================================
# CHẠY QUA ĐÊM — ba việc dài nhất trong 15 mục
#
#   nproc                          # xem có bao nhiêu core
#   tmux new -s v2run          # tên khác 'overnight' đã dùng lần trước
#   source venv/bin/activate
#   PARALLEL=4 bash run_overnight_vong2.sh 2>&1 | tee overnight2.log
#   # Ctrl+B rồi D
#
# Xếp theo mức chặn kết luận: việc nào mà thiếu nó thì không viết được phần
# tương ứng của bài thì chạy trước.
#
#   A. MC1  — headline bằng discrete-event, 10 seed      ~4,2 giờ (PARALLEL=4)
#   B. MC13 — crossover cho hai baseline mode còn thiếu  ~3,1 giờ
#   C. MC8  — baseline centralized đa seed                ~1,0 giờ
#
# TỔNG ~8,5 giờ với PARALLEL=4. Chạy một đêm.
#
# LƯU Ý: A dùng cấu hình payload ĐÃ TỐI ƯU (deterministic, k=20, top-1) và ADC
# song song, vì đó là cấu hình bài sẽ báo cáo. Nếu chạy bằng cấu hình cũ thì số
# latency và RPC không khớp Bảng 12.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
DS=code
N=10000

export PLACEMENT_MODE=deterministic
export PLACEMENT_K=20
export FETCH_TOP=1
export PARALLEL_ADC=1

# Kiểm công cụ phân tích TRƯỚC khi chạy tám tiếng. Lần trước script chạy xong
# cả ba phần rồi mới hỏng ở bước tổng hợp vì thiếu hai file này — dữ liệu không
# mất, nhưng không ai biết cho tới lúc xem kết quả.
_missing=""
for f in analyze_mc1.py analyze_mc13.py main_simulation.py main_simulation_v2.py baselines.py; do
    [ -f "$f" ] || _missing="$_missing $f"
done
if [ -n "$_missing" ]; then
    echo "THIẾU FILE:$_missing"
    echo "Tải về rồi chạy lại. Dừng ở đây thay vì chạy 8 tiếng rồi mới hỏng."
    exit 1
fi
echo "✓ đủ công cụ phân tích"

echo "PARALLEL=$PARALLEL | nproc=$(nproc 2>/dev/null || echo '?')"
echo "Cấu hình payload: $PLACEMENT_MODE k=$PLACEMENT_K top=$FETCH_TOP, ADC song song"
echo ""

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

# ---------------------------------------------------------------- A. MC1
echo "############ A. MC1 — HEADLINE DISCRETE-EVENT (10 seed) ############"
SEEDS_A="20235956 1 2 3 4 5 6 7 8 9"

run_a() {
    local mode=$1 seed=$2 extra=$3
    local f="mc1_${mode}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] $f"; return; }
    env SKIP_PAYLOAD=0 ROUTING_MODE="$mode" $extra timeout 7200 $PY main_simulation.py \
        --dataset $DS --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] $mode s=$seed"
}

echo "-- semantic --"
for s in $SEEDS_A; do run_a semantic "$s" "" & wait_slot; done; wait

MATCH=$($PY - <<'EOF'
import glob, re, statistics as st
v=[]
for f in glob.glob('mc1_semantic_s*.txt'):
    m=re.search(r'Unique nodes contacted\s+([\d,]+)', open(f,errors='ignore').read())
    if m: v.append(float(m.group(1).replace(',','')))
print(int(round(st.mean(v))) if v else 504)
EOF
)
echo "   semantic chạm TB $MATCH node -> dùng cho random_unique"

for m in keyed_lookup random_slots random_unique; do
    echo "-- $m --"
    ex=""; [ "$m" = "random_unique" ] && ex="MATCH_UNIQUE_NODES=$MATCH"
    for s in $SEEDS_A; do run_a "$m" "$s" "$ex" & wait_slot; done; wait
done
$PY analyze_mc1.py > mc1_results.txt 2>&1; echo "-> mc1_results.txt"

# ---------------------------------------------------------------- B. MC13
echo ""
echo "############ B. MC13 — CROSSOVER CHO HAI MODE CÒN THIẾU ############"
run_b() {
    local L=$1 r=$2 seed=$3 mode=$4
    local sfx; case "$mode" in
        random_unique) sfx="_RANDUNIQ" ;; random_keys) sfx="_RANDKEY" ;; *) sfx="" ;;
    esac
    local f="result_${DS}_N${N}_L${L}_K20_MA${r}_T8_m512${sfx}_s${seed}_nq500.json"
    [ -f "$f" ] && return
    $PY main_simulation_v2.py --dataset $DS --nodes $N --nq 500 \
        --num-tables "$L" --k-query 20 --meta-anchors "$r" --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$seed" --routing "$mode" \
        >/dev/null 2>&1 || echo "  [LỖI] L=$L r=$r s=$seed $mode"
}
for mode in random_unique random_keys; do
    echo "-- $mode --"
    for s in 20235956 1 2 3 4; do
        for L in 4 5 8 10; do
            for r in 1 2 4 5 10; do run_b "$L" "$r" "$s" "$mode" & wait_slot; done
        done
        echo "   xong seed $s"
    done
    wait
done
$PY analyze_mc13.py > mc13_results.txt 2>&1; echo "-> mc13_results.txt"

# ---------------------------------------------------------------- C. MC8
echo ""
echo "############ C. MC8 — BASELINE CENTRALIZED ĐA SEED ############"
for s in 20235956 1 2 3 4; do
    for ds in code scifact; do
        vc=$([ "$ds" = "code" ] && echo 4510 || echo 1845)
        f="mc8_${ds}_s${s}.txt"
        [ -s "$f" ] && { echo "  [skip] $f"; continue; }
        CORPUS=$ds POOL_PER_TABLE=902 VENGRAM_CANDIDATES=$vc \
            RNG_SEED=$s LSH_SEED=$s \
            BUCKET_WIDTHS=8,10,12,14,16 $PY baselines.py > "$f" 2>&1 &
        wait_slot
    done
done
wait
echo "-> mc8_*.txt"

echo ""
echo "############ XONG. Đọc theo thứ tự: ############"
echo "  mc1_results.txt   <- headline dưới định tuyến thật, QUAN TRỌNG NHẤT"
echo "  mc13_results.txt  <- crossover theo từng baseline"
echo "  mc8_*.txt         <- baseline đa seed"