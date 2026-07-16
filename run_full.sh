#!/bin/bash
# ============================================================================
# V-Engram — QUÉT TÌM CẤU HÌNH, 2 PHA
#
# PHA 1 (quét rộng, 1 seed, nq=200): tìm cấu hình đạt Recall@5 > 80% MÀ semantic
#        vẫn thắng random đậm. Mọi cấu hình đều chạy KÈM random baseline.
# PHA 2 (chỉ chạy sau khi chọn): 5 seed × 500 query cho cấu hình đã chốt.
#
# NGUYÊN TẮC: recall cao mà random cũng cao thì VÔ NGHĨA.
#   r=150 cho semantic 70.0% nhưng random 87.6% -> bỏ.
#   r=1,T=3 cho semantic 62.2%, random 13.6%   -> giữ.
# Chọn theo TỈ LỆ semantic/random, không theo recall tuyệt đối.
#
# Chạy:  nohup bash run_full.sh > run_full.log 2>&1 &
# ============================================================================
set -u
NQ=${NQ:-200}
SEED=${SEED:-20235956}
DS=${DS:-code}

run() {
    local desc="$1"; shift
    python main_simulation_v2.py --dataset $DS --nodes 10000 --seed $SEED --nq $NQ "$@" \
        >/dev/null 2>&1 || echo "  [LỖI] $desc"
}

pair() {  # chạy cả semantic + random ở CÙNG cấu hình
    local desc="$1"; shift
    echo "  $desc"
    run "$desc"          "$@"
    run "$desc RANDOM"   "$@" --random-routing
}

echo "##### PHA 1A: quét L × T (r=1, K=20) — trục rẻ nhất #####"
# Tăng L nhân metadata theo L (L=8,r=1 -> 8 bản/doc, vẫn rẻ hơn r=30,L=5 -> 150 bản)
# Tăng T KHÔNG nhân metadata chút nào -> ưu tiên T trước L
for L in 5 8 12; do for T in 1 3 5 8; do
    pair "L=$L T=$T" --num-tables $L --multi-probe $T --meta-anchors 1 --k-query 20 --use-pq
done; done

echo "##### PHA 1B: K cao hơn cho cấu hình hứa hẹn #####"
for L in 5 8; do for T in 3 5; do for K in 50 100; do
    pair "L=$L T=$T K=$K" --num-tables $L --multi-probe $T --meta-anchors 1 --k-query $K --use-pq
done; done; done

echo "##### PHA 1C: r nhỏ (kiểm tra r=5 có đáng không) #####"
for r in 1 5 10; do for T in 3 5; do
    pair "r=$r T=$T" --num-tables 8 --multi-probe $T --meta-anchors $r --k-query 20 --use-pq
done; done

echo "##### PHA 1D: trần no-PQ cho top cấu hình #####"
for L in 5 8 12; do for T in 3 5 8; do
    run "L=$L T=$T noPQ" --num-tables $L --multi-probe $T --meta-anchors 1 --k-query 20 --no-pq
done; done

echo "##### PHA 1E: probe_bits (c) — trục chưa test #####"
for c in 8 12 16 24; do
    pair "c=$c" --num-tables 8 --multi-probe 5 --meta-anchors 1 --k-query 20 --probe-bits $c --use-pq
done

echo "##### PHA 1F: xác nhận trên scifact #####"
for L in 5 8; do for T in 3 5; do
    echo "  scifact L=$L T=$T"
    python main_simulation_v2.py --dataset scifact --nodes 10000 --seed $SEED --nq $NQ \
        --num-tables $L --multi-probe $T --meta-anchors 1 --k-query 20 --use-pq >/dev/null 2>&1
    python main_simulation_v2.py --dataset scifact --nodes 10000 --seed $SEED --nq $NQ \
        --num-tables $L --multi-probe $T --meta-anchors 1 --k-query 20 --use-pq --random-routing >/dev/null 2>&1
done; done

echo ""
echo "##### TỔNG HỢP #####"
python summarize.py
echo ""
echo "ĐỌC BẢNG: chọn cấu hình có Recall@5 cao MÀ tỉ lệ sem/rand còn lớn."
echo "Rồi chạy PHA 2:  bash run_phase2.sh L T K r"