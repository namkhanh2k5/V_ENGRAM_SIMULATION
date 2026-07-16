#!/bin/bash
# ĐO THỜI GIAN TRƯỚC — chạy cái này TRƯỚC khi đẩy lên server.
# Mục đích: biết 1 cấu hình mất bao lâu, rồi nhân lên để ước tính cả bộ.
set -e
echo "=== SMOKE TEST: 3 cấu hình, --nq 100 ==="
for cfg in "20 3" "40 1" "20 3 --random-routing"; do
    set -- $cfg
    K=$1; T=$2; shift 2
    START=$(date +%s)
    python main_simulation_v2.py --dataset code --nodes 10000 --seed 20235956 \
        --k-query $K --meta-anchors 1 --multi-probe $T --nq 100 --use-pq "$@" \
        2>&1 | grep -E "Tầng 3|Node chạm|Candidate"
    echo "    ^ K=$K T=$T $* mất $(( $(date +%s) - START ))s"
    echo ""
done
echo "=== ƯỚC TÍNH ==="
echo "Nhân thời gian trên với 5 (nq 100->500) rồi nhân số cấu hình trong run_full.sh (90)."
echo "Nếu 1 cấu hình @nq100 = 60s  -> full ~ 60*5*90/3600 = 7.5 giờ"