#!/bin/bash
# ============================================================================
# THÍ NGHIỆM MỚI — ĐƯỜNG ĐẲNG LƯU LƯỢNG (iso-traffic)
#
#   tmux new -s iso
#   source venv/bin/activate
#   PARALLEL=4 bash run_iso_traffic.sh 2>&1 | tee iso.log
#
# VÌ SAO: kết quả churn mới cho thấy lưu lượng bảo trì = (số bản) x (tần suất
# làm mới). Bài mới chỉ quét MỘT trục — cố định r=1 rồi đổi chu kỳ. Nhưng cùng
# một mức lưu lượng đạt được bằng NHIỀU cặp (r, chu kỳ) khác nhau, và bài chưa
# hỏi cặp nào tốt nhất.
#
# Câu hỏi: ở CÙNG ngân sách lưu lượng, ít bản + sửa dày có tốt hơn nhiều bản +
# sửa thưa không?
#
#   msg/doc/22h = L*r * 1320/chu_ky,  với L=5
#
#   r=1  ->  5 bản, chu kỳ  330ph  -> 20 msg
#   r=2  -> 10 bản, chu kỳ  660ph  -> 20 msg
#   r=3  -> 15 bản, chu kỳ  990ph  -> 20 msg
#   r=4  -> 20 bản, chu kỳ 1320ph  -> 20 msg   (đã có: tỉ lệ 1,01x)
#
# VÌ SAO ĐÁNG LÀM: hiện bài phải thừa nhận V-Engram tốn nhiều lưu lượng hơn
# IPFS 38% (27,5 so với 20,0). Nếu r=1 ở chu kỳ 330ph giữ được availability
# >= 99% thì V-Engram KHỚP ĐÚNG ngân sách của IPFS mà vẫn giữ tỉ lệ ~2,1x, và
# kết luận đổi từ "tốn hơn nhưng đáng" thành "cùng chi phí, hơn hẳn cơ chế".
#
# Và đường cong r=1..4 ở cùng 20 msg cho thấy tỉ lệ suy giảm theo r như thế
# nào KHI CHI PHÍ ĐƯỢC GIỮ NGUYÊN — đó là hình vẽ mà mục r* còn thiếu.
#
# Ước tính với PARALLEL=4: 4 cấu hình x 3 seed = 12 lần chạy, ~1,5 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2"
TARGET_MSG=20        # msg/doc/22h, khớp IPFS

grep -q "keep-alive vẫn tốn một message" main_churn_engine.py || {
    echo "main_churn_engine.py chưa vá bug lazy+TTL — tải bản mới rồi chạy lại"; exit 1; }
echo "✓ engine đã vá"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local r=$1 rep=$2 seed=$3
    local f="churn_code_N${N}_r${r}_ses120_weibull_rep${rep}l_s${seed}_nq200.json"
    [ -f "$f" ] && { echo "  [skip] r=$r rep=$rep s=$seed"; return; }
    local dur=720; local need=$((rep*3)); [ "$need" -gt "$dur" ] && dur=$need
    $PY main_churn_engine.py --dataset code --nodes $N --nq 200 \
        --median-session 120 --duration "$dur" --session-dist weibull \
        --meta-anchors "$r" --repair-interval "$rep" --seed "$seed" \
        > "isolog_r${r}_rep${rep}_s${seed}.txt" 2>&1 \
        || echo "  [LỖI] r=$r rep=$rep s=$seed"
}

echo ""
echo "########## ĐƯỜNG ĐẲNG LƯU LƯỢNG ${TARGET_MSG} msg/doc/22h ##########"
echo "  L=5, nên số bản = 5r, và chu kỳ = 5r * 1320 / $TARGET_MSG"
echo ""
for s in $SEEDS; do
    run 1  330 "$s" & wait_slot     #  5 bản
    run 2  660 "$s" & wait_slot     # 10 bản
    run 3  990 "$s" & wait_slot     # 15 bản
    # r=4, 1320ph đã chạy ở lượt trước
done
wait

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, json, statistics as st

TARGET = 20.0
rows = []
for f in glob.glob('churn_code_N10000_r*_ses120_weibull_rep*l_s*_nq200.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    c, h = d['config'], d['history']
    if not h or 'phase' not in h[-1]:
        continue
    r, rep, dur = c['meta_anchors'], c['repair_interval'], c['duration']
    msg = h[-1]['repair_msgs'] / 20000 * 1320 / dur if dur else 0
    if abs(msg - TARGET) > 2.5:          # chỉ lấy cấu hình trên đường đẳng
        continue
    m = [x for x in h if x.get('epoch', 0) > 0]
    rows.append({'r': r, 'rep': rep, 'copies': 5 * r, 'msg': msg,
                 'avail_min': min(x['meta_physical'] for x in m),
                 'route_min': min(x['meta_routable'] for x in m),
                 'sem': h[-1]['final_r5'], 'ratio': h[-1]['ratio'],
                 'fp': h[-1]['footprint'], 'r5min': min(x['final_r5'] for x in m)})

if not rows:
    print('  chưa có cấu hình nào trên đường đẳng lưu lượng')
else:
    agg = {}
    for x in rows:
        agg.setdefault((x['r'], x['rep']), []).append(x)
    print(f"{'r':>2s} {'bản':>4s} {'chu kỳ':>8s} {'msg/22h':>8s} {'avail min':>10s} "
          f"{'R@5':>6s} {'R@5 min':>8s} {'TỈ LỆ':>7s} {'dấu vết':>8s}")
    print('-' * 76)
    best = None
    for (r, rep), v in sorted(agg.items()):
        g = lambda k: st.mean(x[k] for x in v)
        print(f"{r:>2} {5*r:>4} {rep:>6}ph {g('msg'):>8.1f} {g('avail_min'):>9.1f}% "
              f"{g('sem'):>5.1f}% {g('r5min'):>7.1f}% {g('ratio'):>6.2f}x {g('fp'):>8.2f}")
        if g('avail_min') >= 99.0 and (best is None or g('ratio') > best[1]):
            best = ((r, rep), g('ratio'), g('avail_min'), g('fp'))
    print()
    if best:
        (r, rep), ratio, av, fp = best
        print(f"  TỐT NHẤT trên đường đẳng {TARGET:.0f} msg với availability >= 99%:")
        print(f"    r={r}, chu kỳ {rep}ph -> tỉ lệ {ratio:.2f}x, avail min {av:.1f}%, "
              f"dấu vết {fp:.2f}")
        print()
        print(f"  So với IPFS ở CÙNG {TARGET:.0f} msg: tỉ lệ 1,01x, dấu vết 21,99")
        if ratio > 1.5:
            print(f"    => Ở CÙNG ngân sách lưu lượng, V-Engram giữ {ratio:.2f}x còn IPFS chỉ 1,01x.")
            print(f"       Kết luận đổi từ 'tốn hơn nhưng đáng' thành 'cùng chi phí, hơn hẳn'.")
        else:
            print(f"    => Ở cùng ngân sách, lợi thế chỉ còn {ratio:.2f}x. Kết luận hiện tại giữ nguyên.")
EOF

echo ""
echo "-> đọc bảng trên; nếu r=1 ở 330ph giữ avail >= 99% thì luận điểm mạnh hẳn"
