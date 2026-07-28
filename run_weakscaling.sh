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
# Ước tính: đường A 3 mức + đường B 2 mức = 5 cấu hình × 5 seed × 2 chế độ
#           = 50 lần chạy (một số đã có -> skip).
#   10K/20K    ~1 phút
#   25K/50K    ~4 phút   (K=20)  |  ~6 phút  (K=50)
#   50K/100K  ~15 phút   (K=20)  | ~25 phút  (K=100)
#   Tổng ~7 giờ. Chạy qua đêm.
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
    local ds=$1 nodes=$2 seed=$3 mode=$4 K=${5:-20}
    local sfx=""
    [ "$mode" = "random_slots" ] && sfx="_RANDOM"
    local f="result_${ds}_N${nodes}_L5_K${K}_MA1_T8_m512${sfx}_s${seed}_nq${NQ}.json"
    if [ -f "$f" ]; then echo "  [skip] $ds N=$nodes K=$K s=$seed $mode"; return; fi
    $PY main_simulation_v2.py --dataset "$ds" --nodes "$nodes" --nq $NQ \
        --num-tables 5 --k-query "$K" --meta-anchors 1 --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$seed" --routing "$mode" \
        >/dev/null 2>&1 || echo "  [LỖI] $ds N=$nodes K=$K s=$seed $mode"
}

# HAI ĐƯỜNG, vì một mình không kể được câu chuyện:
#
#   ĐƯỜNG A — K CỐ ĐỊNH ở 20:
#     Chi phí truy vấn không đổi khi hệ lớn lên. Nhưng K/N co lại, nên mỗi query
#     phủ ít corpus dần và recall GIẢM. Đo thử: 80% (20K) -> 56% (100K).
#     Đây là điều xảy ra nếu triển khai giữ nguyên tham số.
#
#   ĐƯỜNG B — K TỈ LỆ với N (giữ K/N = 0,20%):
#     K = 20, 50, 100 tương ứng N = 10k, 25k, 50k. Recall GIỮ ĐƯỢC (đo thử:
#     76% ở 100K), giá là chi phí truy vấn tăng TUYẾN TÍNH theo N.
#
# Hai đường cạnh nhau cho thấy đánh đổi thật: ở quy mô lớn, hoặc mất recall,
# hoặc trả chi phí tuyến tính. Không có bữa trưa miễn phí.

echo "##### ĐƯỜNG A — K CỐ ĐỊNH (chi phí không đổi, recall giảm) #####"
for s in $SEEDS; do
  for cfg in "code 10000 20" "code50k 25000 20" "code100k 50000 20"; do
    set -- $cfg; ds=$1; nodes=$2; K=$3
    echo "[A] $ds N=$nodes K=$K s=$s"
    run "$ds" "$nodes" "$s" semantic "$K"
    run "$ds" "$nodes" "$s" random_slots "$K"
  done
done

echo ""
echo "##### ĐƯỜNG B — K TỈ LỆ N (giữ K/N=0,20%, recall giữ, chi phí tăng) #####"
for s in $SEEDS; do
  # K=20 ở N=10000 đã chạy ở đường A, bỏ qua để khỏi trùng
  for cfg in "code50k 25000 50" "code100k 50000 100"; do
    set -- $cfg; ds=$1; nodes=$2; K=$3
    echo "[B] $ds N=$nodes K=$K s=$s"
    run "$ds" "$nodes" "$s" semantic "$K"
    run "$ds" "$nodes" "$s" random_slots "$K"
  done
done

echo ""
echo "##### TỔNG HỢP #####"
MIN_NQ=$NQ $PY summarize.py > weakscaling_full.txt 2>&1
$PY analyze_weakscaling.py 2>&1 | tee weakscaling.txt
echo ""
echo "-> weakscaling.txt      (bảng weak scaling)"
echo "-> weakscaling_full.txt (bảng tổng hợp đầy đủ)"