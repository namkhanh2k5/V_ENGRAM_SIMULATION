#!/bin/bash
# ============================================================================
# BỔ SUNG CẤU HÌNH CÒN THIẾU CHO HAI BẢNG
#
#   PARALLEL=4 bash run_sweeps_extra.sh 2>&1 | tee extra.log
#
# KIỂM: ls repl_r*_s*.json | wc -l   (cần 60)
#       ls fail_r*L30_s*.json | wc -l (cần 20)
# XEM : python3 analyze_two_sweeps.py && python3 analyze_extra.py
#
# ---------------------------------------------------------------------------
# run_two_sweeps.sh chạy replication r=1..4 và failure r=1 ở 0/10/20/30%.
# Nhưng Table replication trong bài còn có r=5 và r=10, và Table failure còn có
# r=2, r=3 ở mức mất 30%. Không chạy nốt thì bảng nửa mười seed nửa một seed,
# mà thầy yêu cầu mean, SD và seed count cho CẢ bảng.
#
# 40 lần chạy, ~1 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2 3 4 5 6 7 8 9"

$PY -c "import numpy" 2>/dev/null || { echo "chưa vào venv"; exit 1; }
wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

echo "########## replication r=5, r=10 ##########"
for r in 5 10; do
    for s in $SEEDS; do
        f="repl_r${r}_s${s}.json"
        [ -f "$f" ] && { echo "  [skip] r=$r s=$s"; continue; }
        $PY main_simulation_v2.py --dataset code --nodes $N --nq 500 \
            --num-tables 5 --k-query 20 --meta-anchors "$r" --multi-probe 8 \
            --use-pq --pq-variant m512 --seed "$s" --out "$f" >/dev/null 2>&1 \
            || echo "  [LỖI] r=$r s=$s"
        wait_slot
    done
done
wait

echo ""
echo "########## failure r=2, r=3 ở mức mất 30% ##########"
for r in 2 3; do
    for s in $SEEDS; do
        f="fail_r${r}L30_s${s}.json"
        [ -f "$f" ] && { echo "  [skip] r=$r s=$s"; continue; }
        $PY main_simulation_v2.py --dataset code --nodes $N --nq 500 \
            --num-tables 5 --k-query 20 --meta-anchors "$r" --multi-probe 8 \
            --use-pq --pq-variant m512 --seed "$s" --node-loss 0.30 \
            --out "$f" >/dev/null 2>&1 || echo "  [LỖI] r=$r s=$s"
        wait_slot
    done
done
wait

echo ""
$PY - <<'PYEOF'
import glob, json, statistics as st
print('=' * 60)
print('BỔ SUNG — REPLICATION r=5, r=10')
print('=' * 60)
for r in (5, 10):
    v = [json.load(open(f)) for f in glob.glob(f'repl_r{r}_s*.json')]
    v = [x for x in v if x.get('recall5') is not None]
    if v:
        m = st.mean(x['recall5'] for x in v)
        sd = st.stdev([x['recall5'] for x in v]) if len(v) > 1 else 0
        print(f'  r={r:<3} n={len(v):<3} Recall@5 {m:.1f}±{sd:.1f}')
print()
print('=' * 60)
print('BỔ SUNG — FAILURE r=2, r=3 ở mất 30%')
print('=' * 60)
for r in (2, 3):
    v = [json.load(open(f)) for f in glob.glob(f'fail_r{r}L30_s*.json')]
    v = [x for x in v if x.get('recall5') is not None]
    if v:
        m = st.mean(x['recall5'] for x in v)
        sd = st.stdev([x['recall5'] for x in v]) if len(v) > 1 else 0
        print(f'  r={r} loss=30%  n={len(v):<3} Recall@5 {m:.1f}±{sd:.1f}')
PYEOF
