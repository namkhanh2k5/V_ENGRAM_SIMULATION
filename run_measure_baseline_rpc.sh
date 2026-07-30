#!/bin/bash
# ============================================================================
# MỤC 9 — ĐO chi phí RPC của baseline thay vì SUY RA
#
#   tmux new -s rpc
#   source venv/bin/activate
#   bash run_measure_baseline_rpc.sh 2>&1 | tee measure_rpc.log
#
# VẤN ĐỀ: Threats to Validity tự thừa nhận "The RPC figures for the baselines
# are derived from their structure rather than measured". Bảng "Recall per unit
# of cost" dựa trên phép suy: keyed lookup chạy cùng L*T lookup nên "thừa hưởng
# cùng 641 RPC routing", và hai oracle "không lookup nên RPC = số ADC".
#
# Phép suy đó có một giả định CHƯA KIỂM: khoá ngẫu nhiên tốn như khoá ngữ nghĩa
# để định tuyến tới. Nếu prefix dày mà semantic key nhắm vào lại dễ hoặc khó tới
# hơn một cách hệ thống, giả định sai.
#
# Script này chạy CẢ BỐN chế độ trong bộ mô phỏng discrete-event và đo trực tiếp.
#
# BỐN CHẾ ĐỘ:
#   semantic      — dùng semantic key
#   keyed_lookup  — L*T khoá NGẪU NHIÊN + lookup Kademlia thật (thực thi được)
#   random_slots  — oracle: bốc L*K*T node, không lookup
#   random_unique — oracle: bốc đúng số node phân biệt semantic chạm
#
# SKIP_PAYLOAD=1 vì phép so này chỉ cần chi phí DISCOVERY. Payload giống nhau ở
# mọi chế độ nên không ảnh hưởng so sánh, mà bỏ nó nhanh hơn ~7 lần.
#
# Ước tính: 4 lần chạy × 25-45 phút = 2-3 giờ. keyed_lookup chậm nhất vì nó
# chạm nhiều node phân biệt hơn semantic.
# ============================================================================
set -u
PY=python3
DS=code
N=10000
NQ=500
SEED=20235956

# Số node phân biệt mà semantic chạm, để random_unique khớp đúng.
# Lấy từ bảng chi phí trong bài; script sẽ cập nhật lại sau lần chạy semantic.
MATCH=504

run() {
    local mode=$1 extra_env=$2
    local f="rpc_${mode}.txt"
    if [ -s "$f" ] && grep -q "RPC/query" "$f"; then
        echo "  [skip] $f"; return
    fi
    echo "  chạy $mode (25-45 phút) ..."
    env SKIP_PAYLOAD=1 ROUTING_MODE="$mode" $extra_env \
        timeout 7200 $PY main_simulation.py \
        --dataset $DS --nodes $N --seed $SEED \
        --k-query 20 --multi-probe 8 --meta-anchors 1 --nq $NQ \
        > "$f" 2>&1 || echo "  [BỎ QUA] $mode quá 2 giờ"
}

echo "##### 1. SEMANTIC (mốc so) #####"
run semantic ""

# đọc số node phân biệt thật từ lần chạy semantic, để random_unique khớp đúng
if [ -s rpc_semantic.txt ]; then
    M=$(grep -oE "Unique nodes contacted +[0-9]+" rpc_semantic.txt | grep -oE "[0-9]+$" | head -1)
    [ -n "${M:-}" ] && MATCH=$M
fi
echo "  -> semantic chạm $MATCH node phân biệt, dùng cho random_unique"

echo ""
echo "##### 2. KEYED LOOKUP (thực thi được — quan trọng nhất) #####"
run keyed_lookup ""

echo ""
echo "##### 3. RANDOM SLOTS (oracle, L*K*T node) #####"
run random_slots ""

echo ""
echo "##### 4. RANDOM UNIQUE (oracle, khớp $MATCH node) #####"
run random_unique "MATCH_UNIQUE_NODES=$MATCH"

echo ""
echo "##### TỔNG HỢP #####"
$PY analyze_baseline_rpc.py 2>&1 | tee rpc_results.txt
echo "-> rpc_results.txt"
