#!/bin/bash
# ============================================================================
# TÁCH HAI TRỤC CỦA RIPPLE SEARCH — theo yêu cầu đo thêm của thầy
#
#   tmux new -s axes
#   source venv/bin/activate
#   PARALLEL=4 bash run_ripple_axes.sh 2>&1 | tee axes.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation.py | grep -v grep
#   ls result_full_code_*-*_s*_nq500.json 2>/dev/null | wc -l    # cần 12
#   tail -5 axes.log
#
# XEM KẾT QUẢ
#   bash run_ripple_axes.sh 2>&1 | tail -35
#
# ---------------------------------------------------------------------------
# VÌ SAO: phép đo trước trộn HAI thay đổi làm một, nên chênh 5,1 điểm recall
# chưa quy được cho trục nào.
#
#   STOP_RULE      — khi nào dừng
#     stable   : frontier K-peer không đổi qua hai vòng (frontier-stability)
#     exhaust  : mọi node trong top-K đã được hỏi (exhaustive top-K frontier)
#
#   FRONTIER_SCOPE — hỏi node nào
#     all   : node gần nhất chưa hỏi trong TOÀN BỘ tập ứng viên
#             -> khi top-K hỏi hết, tiếp tục ra ngoài frontier
#     topk  : chỉ node trong top-K hiện tại
#
# Điều này quyết định cách MÔ TẢ thuật toán trong bài:
#
#   Nếu lợi thế đến từ STOP_RULE  -> "Ripple Search chấp nhận frontier xấp xỉ
#       thay vì tiêu thêm RPC để hội tụ về peer XOR-gần hơn trên toàn cục."
#
#   Nếu lợi thế đến từ SCOPE      -> "Ripple Search chủ động hỏi ra NGOÀI
#       frontier hiện tại, vì mục tiêu là ĐỘ PHỦ chứ không phải hội tụ."
#
# Cách thứ hai mạnh hơn vì nó là chủ ý thiết kế, không phải chấp nhận xấp xỉ.
#
# KÈM PHÉP ĐO TRỰC TIẾP: MEASURE_OVERLAP=1 đo tập trả về trùng bao nhiêu với
# tập K peer XOR-gần nhất TOÀN CỤC. Nếu overlap THẤP mà recall CAO thì đó là
# bằng chứng trực tiếp rằng hội tụ XOR không phải mục tiêu — chính là luận điểm
# thầy đề xuất, chứng minh bằng số thay vì bằng lập luận.
#
# Ước tính với PARALLEL=4: 4 cấu hình x 3 seed = 12 lần chạy, ~2,5 giờ.
#   (đo overlap phải sắp xếp toàn mạng mỗi lookup nên chậm hơn bình thường)
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2"

grep -q "FRONTIER_SCOPE" src/routing.py || {
    echo "src/routing.py chưa tách hai trục — tải bản mới rồi chạy lại"; exit 1; }
grep -q "STOP_RULE}-" main_simulation.py || {
    echo "main_simulation.py chưa mã hoá hai cờ vào tên file — tải bản mới"; exit 1; }
# Kiểm venv TRƯỚC. Chạy ngoài venv thì numpy/simpy không có và cả 12 lần chạy
# hỏng ngay lập tức mà lỗi bị nuốt vào file log riêng — mất công phát hiện.
$PY -c "import numpy, simpy" 2>/dev/null || {
    echo "THIẾU numpy/simpy — chưa vào venv?"
    echo "  chạy:  source venv/bin/activate"
    exit 1; }
echo "✓ venv OK"
echo "✓ code đủ điều kiện"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local sr=$1 fs=$2 seed=$3
    local f
    # Tên file thật có dạng ..._padc_stable-all_nopay_semantic_scan300_s1_nq500.json
    # nên giữa "$sr-$fs" và "_s$seed" còn nhiều đoạn khác — phải có * ở giữa.
    f=$(ls result_full_code_N${N}_*_${sr}-${fs}_*_s${seed}_nq500.json 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "  [skip] $sr/$fs s=$seed"; return; }
    SKIP_PAYLOAD=1 STOP_RULE=$sr FRONTIER_SCOPE=$fs MEASURE_OVERLAP=1 \
        timeout 14400 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$seed" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 500 \
        > "axes_${sr}-${fs}_s${seed}.txt" 2>&1 \
        || echo "  [LỖI] $sr/$fs s=$seed"
}

echo ""
echo "########## QUÉT 2x2 ##########"
for s in $SEEDS; do
    for sr in stable exhaust; do
        for fs in all topk; do
            run "$sr" "$fs" "$s" & wait_slot
        done
    done
    echo "  xong seed $s"
done
wait

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, json, re, statistics as st
from collections import defaultdict

g = defaultdict(list)
for f in glob.glob('result_full_code_N10000_*_s*_nq500.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    sr, fs = d.get('stop_rule'), d.get('frontier_scope')
    if sr and fs:
        g[(sr, fs)].append(d)

if not g:
    print('  chưa có dữ liệu')
else:
    print(f"{'dừng':9s} {'phạm vi':8s} {'n':>2s} {'Recall@5':>9s} {'RPC':>7s} "
          f"{'vòng':>6s} {'node':>6s} {'overlap':>8s}")
    print('-' * 60)
    res = {}
    for sr in ('stable', 'exhaust'):
        for fs in ('all', 'topk'):
            v = g.get((sr, fs), [])
            if not v:
                print(f'{sr:9s} {fs:8s} {"(chưa chạy)":>12s}'); continue
            a = lambda k: st.mean(x[k] for x in v if x.get(k) is not None)
            r = a('recall_at_5')
            res[(sr, fs)] = r
            ov = a('overlap_mean') if any(x.get('overlap_mean') for x in v) else None
            print(f"{sr:9s} {fs:8s} {len(v):>2} {r:>8.1f}% {a('disc_rpcs'):>7.0f} "
                  f"{a('disc_rounds'):>6.0f} {a('nodes_touched'):>6.0f} "
                  f"{(100*ov if ov else 0):>7.0f}%")

    if len(res) == 4:
        print()
        print('=' * 60)
        print('TÁCH ĐÓNG GÓP CỦA TỪNG TRỤC')
        print('=' * 60)
        base = res[('stable', 'all')]
        d_stop = res[('exhaust', 'all')] - base       # chỉ đổi điều kiện dừng
        d_scope = res[('stable', 'topk')] - base      # chỉ đổi phạm vi hỏi
        d_both = res[('exhaust', 'topk')] - base
        print(f"  mốc (stable, all)          {base:>6.1f}%")
        print(f"  chỉ đổi ĐIỀU KIỆN DỪNG     {d_stop:>+6.1f} điểm")
        print(f"  chỉ đổi PHẠM VI HỎI        {d_scope:>+6.1f} điểm")
        print(f"  đổi cả hai                 {d_both:>+6.1f} điểm")
        print()
        if abs(d_scope) > abs(d_stop) + 1:
            print('  => PHẠM VI HỎI là trục chính.')
            print('     Mô tả trong bài: "Ripple Search chủ động hỏi ra ngoài frontier')
            print('     hiện tại, vì mục tiêu là ĐỘ PHỦ chứ không phải hội tụ."')
        elif abs(d_stop) > abs(d_scope) + 1:
            print('  => ĐIỀU KIỆN DỪNG là trục chính.')
            print('     Mô tả trong bài: "Ripple Search chấp nhận frontier xấp xỉ thay')
            print('     vì tiêu thêm RPC để hội tụ về peer XOR-gần hơn trên toàn cục."')
        else:
            print('  => Hai trục đóng góp tương đương, hoặc tương tác với nhau.')
            print(f'     Kiểm cộng tính: {d_stop:+.1f} + {d_scope:+.1f} = '
                  f'{d_stop+d_scope:+.1f} so với {d_both:+.1f} khi đổi cả hai.')
            if abs(d_stop + d_scope - d_both) > 1.5:
                print('     Lệch nhiều -> hai trục TƯƠNG TÁC, phải mô tả cả hai.')

    # overlap vs recall
    print()
    print('=' * 60)
    print('OVERLAP VỚI TẬP XOR-GẦN NHẤT TOÀN CỤC vs RECALL')
    print('=' * 60)
    pts = []
    for (sr, fs), v in g.items():
        ov = [x['overlap_mean'] for x in v if x.get('overlap_mean')]
        rc = [x['recall_at_5'] for x in v if x.get('recall_at_5')]
        if ov and rc:
            pts.append((st.mean(ov), st.mean(rc), f'{sr}/{fs}'))
    for o, r, lbl in sorted(pts):
        print(f'  {lbl:16s} overlap {100*o:>5.1f}%   recall {r:>5.1f}%')
    if len(pts) >= 2:
        hi_ov = max(pts)[0]; lo_ov = min(pts)[0]
        r_hi = [r for o, r, _ in pts if o == hi_ov][0]
        r_lo = [r for o, r, _ in pts if o == lo_ov][0]
        print()
        if r_lo > r_hi:
            print('  => Cấu hình có overlap THẤP HƠN lại cho recall CAO HƠN.')
            print('     Đây là bằng chứng TRỰC TIẾP rằng hội tụ về tập XOR-gần nhất')
            print('     KHÔNG phải mục tiêu đúng cho khám phá ứng viên ngữ nghĩa.')
            print('     Đưa số này vào bài để đỡ phải lập luận suông.')
        else:
            print('  => Overlap cao đi kèm recall cao. Luận điểm "hội tụ XOR không')
            print('     phải mục tiêu" KHÔNG được số liệu ủng hộ ở đây; nên mô tả')
            print('     thuật toán là xấp xỉ vì ngân sách, không vì mục tiêu khác.')
EOF