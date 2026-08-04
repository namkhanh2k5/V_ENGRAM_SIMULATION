#!/bin/bash
# ============================================================================
# HAI THÍ NGHIỆM TIẾP THEO
#
#   tmux new -s budget
#   source venv/bin/activate
#   PARALLEL=4 bash run_min_budget.sh 2>&1 | tee budget.log
#
# ---------------------------------------------------------------------------
# A. NGÂN SÁCH TỐI THIỂU
#
# Đường đẳng lưu lượng vừa cho thấy: ở 20 msg/doc/22h — đúng ngân sách IPFS —
# thì r=1 giữ tỉ lệ 2,13x còn IPFS chỉ 1,01x. Nhưng 20 là ngân sách CỦA IPFS,
# không phải ngân sách tối thiểu của V-Engram.
#
# Từ dữ liệu r=1 đã có:
#     240ph -> avail 99,7%  27,5 msg   đạt
#     330ph -> avail 99,4%  20,0 msg   đạt
#     480ph -> avail 98,4%  13,8 msg   KHÔNG đạt
#
# Ngưỡng nằm giữa 330 và 480ph, chưa ai đo. Nếu 400ph vẫn giữ >= 99% thì ngân
# sách tối thiểu là 16,5 msg, tức RẺ HƠN IPFS 18% — và bài nói được "cùng độ
# bền, ít lưu lượng hơn, ít lưu trữ hơn, và giữ được cơ chế", tức thắng cả bốn.
#
# Quét 360, 400, 440ph để kẹp ngưỡng.
#
# ---------------------------------------------------------------------------
# B. NGÂN SÁCH TỐI THIỂU PHỤ THUỘC TỐC ĐỘ CHURN THẾ NÀO
#
# Mọi thí nghiệm churn đến giờ dùng median session = 120 phút, một con số chọn
# ở giữa Li et al. (60 phút) và IPFS (~8 giờ). Nhưng ngân sách sửa chữa phải
# phụ thuộc tốc độ churn, và bài chưa đo quan hệ đó.
#
# Đo tại median 60, 120, 240 phút, mỗi mức tìm chu kỳ giữ được >= 99%.
# Nếu ngân sách tối thiểu tỉ lệ NGHỊCH với median session thì có một quy tắc
# thiết kế phát biểu được: "chu kỳ sửa nên bằng khoảng 3 lần median session".
#
# Ước tính với PARALLEL=4:
#   A. 3 chu kỳ x 3 seed              ~0,8 giờ
#   B. 2 median x 3 chu kỳ x 3 seed   ~1,7 giờ
#   TỔNG ~2,5 giờ
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2"

grep -q "keep-alive vẫn tốn một message" main_churn_engine.py || {
    echo "main_churn_engine.py chưa vá bug lazy+TTL — tải bản mới"; exit 1; }
echo "✓ engine đã vá"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local ses=$1 rep=$2 seed=$3
    local f="churn_code_N${N}_r1_ses${ses}_weibull_rep${rep}l_s${seed}_nq200.json"
    [ -f "$f" ] && { echo "  [skip] ses=$ses rep=$rep s=$seed"; return; }
    local dur=$((ses*6)); local need=$((rep*3))
    [ "$need" -gt "$dur" ] && dur=$need
    $PY main_churn_engine.py --dataset code --nodes $N --nq 200 \
        --median-session "$ses" --duration "$dur" --session-dist weibull \
        --meta-anchors 1 --repair-interval "$rep" --seed "$seed" \
        > "budlog_ses${ses}_rep${rep}_s${seed}.txt" 2>&1 \
        || echo "  [LỖI] ses=$ses rep=$rep s=$seed"
}

echo ""
echo "########## A. KẸP NGƯỠNG Ở median = 120 phút ##########"
for s in $SEEDS; do
    for rep in 360 400 440; do run 120 "$rep" "$s" & wait_slot; done
done
wait

echo ""
echo "########## B. NGƯỠNG THEO TỐC ĐỘ CHURN ##########"
# median 60: churn nhanh gấp đôi -> thử chu kỳ ngắn hơn
for s in $SEEDS; do
    for rep in 120 165 240; do run 60 "$rep" "$s" & wait_slot; done
done
wait
# median 240: churn chậm một nửa -> thử chu kỳ dài hơn
for s in $SEEDS; do
    for rep in 480 660 960; do run 240 "$rep" "$s" & wait_slot; done
done
wait

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, json, statistics as st
from collections import defaultdict

g = defaultdict(list)
for f in glob.glob('churn_code_N10000_r1_ses*_weibull_rep*l_s*_nq200.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    c, h = d['config'], d['history']
    if not h or 'phase' not in h[-1]:
        continue
    m = [x for x in h if x.get('epoch', 0) > 0]
    if not m:
        continue
    dur = c['duration']
    g[(c['median_session'], c['repair_interval'])].append({
        'avail': min(x['meta_physical'] for x in m),
        'msg': h[-1]['repair_msgs'] / 20000 * 1320 / dur if dur else 0,
        'ratio': h[-1]['ratio'], 'sem': h[-1]['final_r5'],
        'fp': h[-1]['footprint']})

for ses in sorted({k[0] for k in g}):
    print()
    print('=' * 76)
    print(f'MEDIAN SESSION = {ses:.0f} PHÚT')
    print('=' * 76)
    print(f"{'chu kỳ':>8s} {'ngưỡng/med':>11s} {'avail min':>10s} {'msg/22h':>8s} "
          f"{'R@5':>6s} {'TỈ LỆ':>7s}")
    print('-' * 60)
    thresh = None
    for rep in sorted(k[1] for k in g if k[0] == ses):
        v = g[(ses, rep)]
        a = lambda k: st.mean(x[k] for x in v)
        ok = a('avail') >= 99.0
        if ok:
            thresh = (rep, a('msg'), a('ratio'))
        mark = '' if ok else '  < 99%'
        print(f"{rep:>6.0f}ph {rep/ses:>10.1f}x {a('avail'):>9.1f}% {a('msg'):>8.1f} "
              f"{a('sem'):>5.1f}% {a('ratio'):>6.2f}x{mark}")
    if thresh:
        rep, msg, ratio = thresh
        print()
        print(f"  Chu kỳ THƯA NHẤT giữ >= 99%: {rep:.0f}ph = {rep/ses:.1f}x median")
        print(f"  Ngân sách tối thiểu: {msg:.1f} msg/doc/22h, tỉ lệ {ratio:.2f}x")
        if msg < 20:
            print(f"    => RẺ HƠN IPFS {100*(1-msg/20):.0f}% ở cùng độ bền.")
        else:
            print(f"    => vẫn tốn hơn IPFS {100*(msg/20-1):.0f}%.")

print()
print('=' * 76)
print('QUY TẮC THIẾT KẾ RÚT RA')
print('=' * 76)
ths = {}
for ses in sorted({k[0] for k in g}):
    best = None
    for rep in sorted(k[1] for k in g if k[0] == ses):
        v = g[(ses, rep)]
        if st.mean(x['avail'] for x in v) >= 99.0:
            best = (rep, st.mean(x['msg'] for x in v))
    if best:
        ths[ses] = best
if len(ths) >= 2:
    print(f"  {'median':>8s} {'chu kỳ':>8s} {'tỉ số':>7s} {'msg tối thiểu':>14s}")
    for ses, (rep, msg) in sorted(ths.items()):
        print(f'  {ses:>6.0f}ph {rep:>6.0f}ph {rep/ses:>6.1f}x {msg:>13.1f}')
    ratios = [rep / ses for ses, (rep, _) in ths.items()]
    msgs = [m for _, (_, m) in ths.items()]
    if max(ratios) - min(ratios) < 0.6:
        print()
        print(f'  => Tỉ số chu kỳ/median gần như HẰNG SỐ ({st.mean(ratios):.1f}x).')
        print(f'     Quy tắc: đặt chu kỳ sửa bằng ~{st.mean(ratios):.0f} lần median session.')
        print(f'     Và ngân sách tối thiểu KHÔNG phụ thuộc tốc độ churn '
              f'({min(msgs):.1f}-{max(msgs):.1f} msg).')
    else:
        print()
        print(f'  => Tỉ số đổi từ {min(ratios):.1f}x tới {max(ratios):.1f}x — không hằng số.')
        print(f'     Ngân sách tối thiểu {min(msgs):.1f}-{max(msgs):.1f} msg, phụ thuộc churn.')
else:
    print('  chưa đủ mức median để rút quy tắc')
EOF
