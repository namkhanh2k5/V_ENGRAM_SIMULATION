#!/bin/bash
# ============================================================================
# WEAK SCALING — giữ tỉ lệ doc/node cố định
#
#   nohup bash run_weakscaling.sh > weakscaling.log 2>&1 &
#
# Khác N-sweep đã có: N-sweep giữ CORPUS cố định 20.000 doc và chỉ tăng node,
# nên số doc mỗi node GIẢM. Đó là phép kiểm công thức r*, KHÔNG phải scalability.
#
# Weak scaling giữ tỉ lệ 2 doc/node và tăng CẢ HAI:
#     10.000 node /  20.000 doc  (= cấu hình headline, đã có kết quả)
#     25.000 node /  50.000 doc
#     50.000 node / 100.000 doc
#
# Câu hỏi: recall và tỉ lệ sem/rand có giữ được khi hệ lớn lên ở tỉ lệ cố định?
#
# ĐIỀU KIỆN: đã có data/code50k_* và data/code100k_* (dựng trên Colab).
#
# Ước tính: 3 mức × 5 seed × 2 chế độ = 30 lần chạy.
#   10K/20K   ~1 phút  (đa số đã có -> skip)
#   25K/50K   ~4 phút
#   50K/100K  ~15 phút
#   Tổng ~3,5 giờ.
# ============================================================================
set -u
SEEDS="20235956 1 2 3 4"
NQ=500
PY=python3

# Kiểm dữ liệu trước khi chạy, tránh chạy 3 giờ rồi mới lỗi
for d in code50k code100k; do
    for f in corpus_embeddings query_embeddings pq_codes_m512 pq_codebook_m512; do
        [ -f "data/${d}_${f}.npy" ] || { echo "THIẾU data/${d}_${f}.npy"; exit 1; }
    done
    [ -f "data/${d}_ground_truth.json" ] || { echo "THIẾU data/${d}_ground_truth.json"; exit 1; }
done
echo "✓ Dữ liệu đủ"
echo ""

run() {
    local ds=$1 nodes=$2 seed=$3 mode=$4
    local sfx=""
    [ "$mode" = "random_slots" ] && sfx="_RANDOM"
    local f="result_${ds}_N${nodes}_L5_K20_MA1_T8_m512${sfx}_s${seed}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] $ds N=$nodes s=$seed $mode"; return; fi
    $PY main_simulation_v2.py --dataset "$ds" --nodes "$nodes" --nq $NQ \
        --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$seed" --routing "$mode" \
        >/dev/null 2>&1 || echo "  [LỖI] $ds N=$nodes s=$seed $mode"
}

for s in $SEEDS; do
  # cặp (dataset, số node) giữ tỉ lệ 2 doc/node
  for cfg in "code 10000" "code50k 25000" "code100k 50000"; do
    set -- $cfg; ds=$1; nodes=$2
    echo "[weak] $ds N=$nodes s=$s"
    run "$ds" "$nodes" "$s" semantic
    run "$ds" "$nodes" "$s" random_slots
  done
done

echo ""
echo "##### TỔNG HỢP #####"
MIN_NQ=$NQ $PY summarize.py > weakscaling_full.txt 2>&1
$PY analyze_weakscaling.py 2>&1 | tee weakscaling.txt
echo ""
echo "-> weakscaling.txt      (bảng weak scaling)"
echo "-> weakscaling_full.txt (bảng tổng hợp đầy đủ)"