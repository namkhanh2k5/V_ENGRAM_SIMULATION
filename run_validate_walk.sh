#!/bin/bash
# ============================================================================
# KIỂM CHỨNG: các sweep có đứng vững dưới WALK KADEMLIA THẬT không?
#
#   nohup bash run_validate_walk.sh > validate.log 2>&1 &
#
# VẤN ĐỀ: mọi sweep trong bài dùng main_simulation_v2.py, vốn LÝ TƯỞNG HOÁ
# lookup — nó tính XOR tới TOÀN BỘ node rồi lấy K node gần nhất tuyệt đối.
# Kademlia thật đi lặp và chỉ XẤP XỈ tập đó.
#
# Đã đo chênh lệch ở MỘT cấu hình: 80,0% (lý tưởng) vs 77% (walk thật) = 3 điểm.
# Nhưng không có gì bảo đảm chênh lệch đó GIỮ NGUYÊN qua các cấu hình. Nếu
# cấu hình ngân sách nhỏ mất nhiều hơn cấu hình ngân sách lớn, các khác biệt
# mà sweep báo cáo sẽ bị NÉN LẠI, và kết luận có thể đổi.
#
# Rủi ro cụ thể cho đóng góp chính: baseline "random_slots" KHÔNG định tuyến
# (nó bốc node trực tiếp), nên nó KHÔNG mất gì dưới walk thật. Chỉ semantic mất.
# Vậy tỉ lệ sem/rand chỉ có thể GIẢM. Ngưỡng r* có thể dịch.
#
# SÁU CẤU HÌNH, chọn theo các luận điểm mà bài dựa vào:
#   L=5  r=1  K=20  T=8   — cấu hình chốt         (lý tưởng: 80,0 / 34,5 = 2,32x)
#   L=5  r=3  K=20  T=8   — giữa dải r*           (lý tưởng: 80,4 / 70,6 = 1,14x)
#   L=8  r=5  K=20  T=8   — QUA điểm giao r*      (lý tưởng: 92,3 / 95,6 = 0,97x)
#   L=5  r=1  K=20  T=1   — không thăm dò         (lý tưởng: 39,8)
#   L=5  r=1  K=100 T=1   — mở rộng thay thăm dò  (lý tưởng: 65,2)
#   L=12 r=1  K=20  T=8   — L cao                 (lý tưởng: 95,3 / 89,6 = 1,06x)
#
# Mỗi cấu hình chạy CẢ semantic LẪN random để tính tỉ lệ dưới walk thật.
#
# TỐI ƯU: đặt SKIP_PAYLOAD=1, bỏ hẳn tầng payload.
#   Ingest tốn 5 lookup cho metadata nhưng 30 cho payload shard, nên bỏ payload
#   nhanh hơn ~7 lần. Phép kiểm này chỉ cần Recall@5, vốn quyết định hoàn toàn ở
#   tầng discovery — payload chỉ lấy nội dung sau khi đã tìm ra object.
#   Đo thực tế: code corpus, N=10.000, 5 query -> 99 giây (trước đó vài chục phút).
#
# Ước tính: ~3 phút mỗi lần chạy với nq=20.
#   12 lần chạy (6 cấu hình × 2 chế độ) = KHOẢNG 40 PHÚT.
# Nếu vẫn muốn dừng sớm, 3 cấu hình đầu đã đủ kiểm luận điểm r*.
# ============================================================================
set -u
PY=python3
NQ=20
SEED=20235956

run() {
    local L=$1 r=$2 K=$3 T=$4 mode=$5
    local sfx=""; [ "$mode" = "random" ] && sfx="_RAND"
    local f="validate_L${L}_r${r}_K${K}_T${T}${sfx}.txt"
    if [ -s "$f" ]; then echo "  [skip] $f"; return; fi
    local extra=""; [ "$mode" = "random" ] && extra="--random-routing"
    echo "  chạy L=$L r=$r K=$K T=$T $mode (có thể mất tới 2 giờ)"
    SKIP_PAYLOAD=1 NUM_TABLES=$L timeout 3600 $PY main_simulation.py --dataset code --nodes 10000 \
        --seed $SEED --k-query "$K" --multi-probe "$T" --meta-anchors "$r" \
        --nq $NQ $extra > "$f" 2>&1 \
        || echo "  [BỎ QUA] L=$L r=$r K=$K T=$T $mode quá 2 giờ"
}

# xếp theo mức quan trọng: ba cấu hình r* trước, nếu hết giờ vẫn có kết luận
for cfg in "5 1 20 8" "5 3 20 8" "8 5 20 8" "5 1 20 1" "5 1 100 1" "12 1 20 8"; do
    set -- $cfg
    echo ""
    echo "##### L=$1 r=$2 K=$3 T=$4 #####"
    run "$1" "$2" "$3" "$4" semantic
    run "$1" "$2" "$3" "$4" random
done

echo ""
echo "##### TỔNG HỢP #####"
$PY analyze_validate.py 2>&1 | tee validate_results.txt
echo "-> validate_results.txt"