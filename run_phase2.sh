#!/bin/bash
# PHA 2 — chạy 5 seed × 500 query cho cấu hình ĐÃ CHỐT từ pha 1.
# Dùng:  bash run_phase2.sh <L> <T> <K> <r>
#   vd:  bash run_phase2.sh 8 5 20 1
set -u
L=${1:?thiếu L}; T=${2:?thiếu T}; K=${3:?thiếu K}; R=${4:?thiếu r}
echo "PHA 2: L=$L T=$T K=$K r=$R | 5 seed × 500 query | code + scifact"
for s in 20235956 1 2 3 4; do for ds in code scifact; do
    echo "[$ds seed=$s]"
    for extra in "--use-pq" "--no-pq" "--use-pq --random-routing"; do
        python main_simulation_v2.py --dataset $ds --nodes 10000 --seed $s --nq 500 \
            --num-tables $L --multi-probe $T --k-query $K --meta-anchors $R $extra \
            >/dev/null 2>&1 || echo "  [LỖI] $ds $s $extra"
    done
done; done
echo ""; python summarize.py