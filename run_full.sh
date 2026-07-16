#!/bin/bash
# ============================================================================
# V-Engram — chạy full cho paper. 5 seed × 500 query.
# Chạy:  bash run_full.sh 2>&1 | tee run_full.log
# Xuất:  result_*.json + summary.csv
# ============================================================================
set -u
SEEDS="20235956 1 2 3 4"
NQ=500
mkdir -p results && cd results 2>/dev/null || true
cd "$(dirname "$0")"

run() {  # bỏ qua nếu file kết quả đã có -> chạy lại được sau khi ngắt
    local tag="$1"; shift
    if ls result_*"$tag"*.json >/dev/null 2>&1; then
        echo "  [bỏ qua, đã có] $tag"; return
    fi
    python main_simulation_v2.py "$@" >/dev/null 2>&1 || echo "  [LỖI] $tag"
}

echo "########## NHÓM 1: BẢNG CHÍNH (r=1, K=20, T=3, PQ on/off) ##########"
for s in $SEEDS; do for ds in code scifact; do
    echo "[N1] $ds seed=$s"
    run "${ds}_K20_MA1_pq_T3_s${s}"   --dataset $ds --seed $s --k-query 20 --meta-anchors 1 --multi-probe 3 --nq $NQ --use-pq
    run "${ds}_K20_MA1_nopq_T3_s${s}" --dataset $ds --seed $s --k-query 20 --meta-anchors 1 --multi-probe 3 --nq $NQ --no-pq
done; done

echo "########## NHÓM 2: NGƯỠNG r* (contribution chính) ##########"
for s in $SEEDS; do for r in 1 5 10 30 150; do
    echo "[N2] r=$r seed=$s"
    run "code_K20_MA${r}_pq_T3_s${s}"        --dataset code --seed $s --k-query 20 --meta-anchors $r --multi-probe 3 --nq $NQ --use-pq
    run "code_K20_MA${r}_pq_T3_RANDOM_s${s}" --dataset code --seed $s --k-query 20 --meta-anchors $r --multi-probe 3 --nq $NQ --use-pq --random-routing
done; done

echo "########## NHÓM 3: PROBE vs WIDEN (ngân sách khớp) ##########"
for s in $SEEDS; do
    echo "[N3] seed=$s"
    run "code_K20_MA1_pq_T3_s${s}" --dataset code --seed $s --k-query 20 --meta-anchors 1 --multi-probe 3 --nq $NQ --use-pq
    run "code_K40_MA1_pq_T1_s${s}" --dataset code --seed $s --k-query 40 --meta-anchors 1 --multi-probe 1 --nq $NQ --use-pq
    run "code_K20_MA1_pq_T5_s${s}" --dataset code --seed $s --k-query 20 --meta-anchors 1 --multi-probe 5 --nq $NQ --use-pq
    run "code_K65_MA1_pq_T1_s${s}" --dataset code --seed $s --k-query 65 --meta-anchors 1 --multi-probe 1 --nq $NQ --use-pq
done

echo "########## NHÓM 4: ĐƯỜNG CONG NGÂN SÁCH ##########"
for s in $SEEDS; do for K in 5 10 20 50 100 300; do
    echo "[N4] K=$K seed=$s"
    run "code_K${K}_MA1_pq_T3_s${s}"        --dataset code --seed $s --k-query $K --meta-anchors 1 --multi-probe 3 --nq $NQ --use-pq
    run "code_K${K}_MA1_pq_T3_RANDOM_s${s}" --dataset code --seed $s --k-query $K --meta-anchors 1 --multi-probe 3 --nq $NQ --use-pq --random-routing
done; done

echo "########## TỔNG HỢP ##########"
python summarize.py