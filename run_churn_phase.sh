#!/bin/bash
# ============================================================================
# MC3 + MC12 — CHẠY LẠI TÁM CẤU HÌNH r=1 VỚI ĐO THEO PHA
#
#   tmux new -s phase
#   source venv/bin/activate
#   PARALLEL=4 bash run_churn_phase.sh 2>&1 | tee phase.log
#
# VÌ SAO PHẢI CHẠY LẠI: tám cấu hình r=1 trong bảng churn chạy bằng engine cũ,
# trước khi thêm đo theo pha. JSON của chúng không có meta_physical /
# meta_routable / phase, nên bảng MC3 hiện chỉ có hai dòng r=4.
#
# Mà tám cấu hình đó chính là ĐỀ XUẤT CỦA BÀI. Tuyên bố trung tâm — "mọi chu kỳ
# sửa đã thử đều giữ 100% availability" — dựa trên phép đo ở mốc epoch, mà mốc
# đó luôn trùng lúc vừa repair xong. Hai dòng r=4 đã có cho thấy chênh lệch là
# thật: phys MIN 96,2% so với mean 98,9%.
#
# Nếu min của r=1 cũng thấp hơn 100% thì câu trong bài phải sửa. Nếu vẫn 100%
# thì tuyên bố đứng vững và giờ có bằng chứng đúng phương pháp.
#
# XOÁ FILE CŨ TRƯỚC: tên file không mã hoá phiên bản engine, nên file cũ sẽ làm
# script skip. Đây là lớp lỗi đã gặp nhiều lần trong dự án.
#
# Ước tính với PARALLEL=4: 8 cấu hình x 3 seed = 24 lần chạy, ~2 giờ.
#   Chu kỳ dài cần thời lượng dài (>= 3 lần chu kỳ) nên rep=1440 chạy 4320 phút.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2"

grep -q "phase" main_churn_engine.py || {
    echo "main_churn_engine.py chưa có đo theo pha — tải bản mới rồi chạy lại"; exit 1; }
grep -q "keep-alive vẫn tốn một message" main_churn_engine.py || {
    echo "main_churn_engine.py chưa vá bug lazy+TTL — tải bản mới rồi chạy lại"
    echo "  (bản cũ bỏ qua doc khoẻ nên TTL hết hạn, availability không đơn điệu)"
    exit 1; }
echo "✓ engine có đo theo pha VÀ đã vá bug lazy+TTL"

# Bug lazy+TTL làm mọi số cũ sai, kể cả file đã có phase. Xoá hết r=1.
mkdir -p backup_churn_buggy
_n=$(ls churn_code_N${N}_r1_ses120_*_nq200.json 2>/dev/null | wc -l)
[ "$_n" -gt 0 ] && {
    echo "Sao lưu $_n file r=1 (chạy bằng engine có bug) rồi xoá"
    mv churn_code_N${N}_r1_ses120_*_nq200.json backup_churn_buggy/ 2>/dev/null
}

# Sao lưu rồi xoá file cũ của r=1: chúng thiếu trường phase nên không dùng được,
# mà tên file lại trùng nên sẽ khiến script skip.
mkdir -p backup_churn_old
_old=$(ls churn_code_N${N}_r1_ses120_*_nq200.json 2>/dev/null | wc -l)
if [ "$_old" -gt 0 ]; then
    echo "Sao lưu $_old file r=1 cũ vào backup_churn_old/ rồi xoá"
    cp churn_code_N${N}_r1_ses120_*_nq200.json backup_churn_old/ 2>/dev/null
    # chỉ xoá file THIẾU trường phase, giữ file đã chạy bằng engine mới
    $PY - <<'EOF'
import glob, json, os
n = 0
for f in glob.glob('churn_code_N10000_r1_ses120_*_nq200.json'):
    try:
        d = json.load(open(f))
        if d['history'] and 'phase' in d['history'][-1]:
            continue          # đã có phase, giữ
    except Exception:
        pass
    os.remove(f); n += 1
print(f"  xoá {n} file thiếu trường phase")
EOF
fi

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local rep=$1 seed=$2
    local f="churn_code_N${N}_r1_ses120_weibull_rep${rep}l_s${seed}_nq200.json"
    [ -f "$f" ] && { echo "  [skip] rep=$rep s=$seed"; return; }
    local dur=720; local need=$((rep*3)); [ "$need" -gt "$dur" ] && dur=$need
    $PY main_churn_engine.py --dataset code --nodes $N --nq 200 \
        --median-session 120 --duration "$dur" --session-dist weibull \
        --meta-anchors 1 --repair-interval "$rep" --seed "$seed" \
        > "phaselog_rep${rep}_s${seed}.txt" 2>&1 \
        || echo "  [LỖI] rep=$rep s=$seed"
}

echo ""
echo "########## TÁM CHU KỲ SỬA, BA SEED ##########"
for s in $SEEDS; do
    for rep in 15 30 60 120 240 480 960 1440; do
        run "$rep" "$s" & wait_slot
    done
    echo "  xong seed $s"
done
wait

echo ""
echo "########## TỔNG HỢP ##########"
$PY analyze_churn.py 2>&1 | tee churn_results_phase.txt
echo "-> churn_results_phase.txt"
echo ""
echo "Đọc khối 'MC3 + MC12'. Cột phys MIN là con số một deployment phải chịu;"
echo "nếu nó thấp hơn 100% thì câu 'mọi chu kỳ giữ 100%' trong bài phải sửa."