#!/bin/bash
# ============================================================================
# BUCKET-LSH MƯỜI SEED — TODO 3 điểm 2-3 và TODO 6
#
#   tmux new -s lsh10
#   source venv/bin/activate
#   bash run_lsh_10seeds.sh 2>&1 | tee lsh10.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep baselines.py | grep -v grep | wc -l
#   grep -l "Recall@5" lsh10_*_T*_s*.txt 2>/dev/null | wc -l     # cần 60
#   tail -5 lsh10.log
#
# XEM KẾT QUẢ
#   python3 analyze_lsh10.py
#
# ---------------------------------------------------------------------------
# BA VIỆC:
#
#   TODO 3 điểm 2: Bucket-LSH hiện chạy NĂM seed còn V-Engram MƯỜI. Chạy lên
#       mười để hai bên cùng số seed, khi đó bảng so được trực tiếp và không
#       cần caveat về số mẫu lệch nhau.
#
#   TODO 3 điểm 3: ghi lại BỀ RỘNG BUCKET được chọn và SỐ ỨNG VIÊN trung bình
#       cho từng cấu hình. Hiện bài chỉ nêu một con số ở T=8 trên code; thiếu
#       phần còn lại thì người khác không lặp lại được.
#
#   TODO 6: nếu hai việc trên xong thì rút gọn caveat về baseline.
#
# CÁCH CHỌN BỀ RỘNG: script quét b in {8,10,12,14,16} rồi analyze_lsh10.py chọn
# b có pool ứng viên GẦN NHẤT với V-Engram (4.510 trên code, 1.845 trên scifact).
# Đó là cách khớp ngân sách ứng viên, và giờ được ghi lại tường minh thay vì
# chọn ngầm.
#
# Ước tính: 2 corpus x 3 mức T x 10 seed = 60 lần chạy, ~2 giờ.
# ============================================================================
set -u
PY=python3
SEEDS="20235956 1 2 3 4 5 6 7 8 9"

$PY -c "import numpy, faiss" 2>/dev/null || {
    echo "THIẾU numpy/faiss — chạy: source venv/bin/activate"; exit 1; }
grep -q "BUCKET_PROBES" baselines.py || {
    echo "baselines.py chưa có multi-probe — tải bản mới rồi chạy lại"; exit 1; }
echo "✓ venv OK, baselines.py có multi-probe"

for ds in code scifact; do
    vc=$([ "$ds" = "code" ] && echo 4510 || echo 1845)
    for T in 1 4 8; do
        for s in $SEEDS; do
            f="lsh10_${ds}_T${T}_s${s}.txt"
            [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] $ds T=$T s=$s"; continue; }
            echo "  $ds T=$T seed=$s ..."
            CORPUS=$ds BUCKET_PROBES=$T POOL_PER_TABLE=902 \
                VENGRAM_CANDIDATES=$vc RNG_SEED=$s LSH_SEED=$s \
                BUCKET_WIDTHS=8,10,12,14,16 \
                timeout 3600 $PY baselines.py > "$f" 2>&1 \
                || echo "    [LỖI] $ds T=$T s=$s"
        done
    done
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY analyze_lsh10.py
