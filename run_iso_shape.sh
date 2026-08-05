#!/bin/bash
# ============================================================================
# ĐƯỜNG ĐẲNG LƯU LƯỢNG Ở ĐUÔI NHẸ — hướng dẫn thiết kế có bền không?
#
#   tmux new -s isoexp
#   source venv/bin/activate
#   PARALLEL=4 bash run_iso_shape.sh 2>&1 | tee isoexp.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_churn_engine | grep -v grep | wc -l
#   ls isoshape_k*_r*_s*.json 2>/dev/null | wc -l        # cần 72
#   tail -5 isoexp.log
#
# XEM KẾT QUẢ
#   bash run_iso_shape.sh 2>&1 | tail -30
#
# ---------------------------------------------------------------------------
# VÌ SAO: Bảng đường đẳng lưu lượng kết luận rằng ở cùng ngân sách, tiêu vào
# TẦN SUẤT (r=1, sửa dày) tốt hơn tiêu vào SỐ BẢN (r=4, sửa thưa) — tỉ lệ 2,13x
# so với 1,01x. Nhưng phép đo đó chỉ chạy ở κ=0,5.
#
# Lập luận ấy dựa vào việc ÍT BẢN vẫn sống sót giữa hai lần sửa. Mà quét hình
# dạng vừa cho thấy đó chính là thứ đuôi nặng cung cấp: ở κ=0,5 xác suất một
# node sống thêm 4 median là 59%, ở κ=1,0 chỉ 6%.
#
# Nên ở đuôi nhẹ, ít bản mất anchor nhanh hơn nhiều, và r cao có thể lại thắng.
# Nếu vậy thì hướng dẫn thiết kế của bài PHẢI kèm điều kiện về hình dạng churn.
#
# CÁCH LÀM: lặp lại đường đẳng lưu lượng (r=1..4, chu kỳ tỉ lệ nghịch với r để
# giữ lưu lượng không đổi) ở BA hình dạng. Mỗi hình dạng dùng ngân sách riêng,
# vì ngân sách tối thiểu khác nhau hẳn — dùng chung một mức thì κ=1,0 hỏng sạch
# và không so được gì.
#
#   κ=0,3: ngân sách 11 msg   -> chu kỳ = L*r*1320/11
#   κ=0,5: ngân sách 20 msg   -> giữ nguyên phép so IPFS đã có
#   κ=1,0: ngân sách 110 msg  -> mức r=1 cần để đạt 99%
#
# Câu hỏi mỗi hình dạng: trong bốn cách chia ngân sách, cách nào cho tỉ lệ
# sem/rand cao nhất mà vẫn giữ availability >= 99%?
#
# Ước tính với PARALLEL=4: 3 shape x 4 r x 3 seed = 36 lần chạy, ~2 giờ.
#   (κ=1,0 chạy chậm hơn vì chu kỳ ngắn -> nhiều lần repair)
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2"
MED=120
L=5

grep -q "keep-alive vẫn tốn một message" main_churn_engine.py || {
    echo "main_churn_engine.py chưa vá bug lazy+TTL — tải bản mới"; exit 1; }
echo "✓ engine đã vá"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local shape=$1 r=$2 budget=$3 seed=$4
    # chu kỳ giữ lưu lượng = budget:  msg = L*r*1320/tau  =>  tau = L*r*1320/budget
    local tau=$(( L*r*1320/budget ))
    local tag; tag=$(echo "$shape" | tr -d '.')
    local f="isoshape_k${tag}_r${r}_s${seed}.json"
    [ -f "$f" ] && { echo "  [skip] k=$shape r=$r s=$seed"; return; }
    local dur=$((MED*6)); local need=$((tau*3))
    [ "$need" -gt "$dur" ] && dur=$need
    echo "  k=$shape r=$r tau=${tau}ph (ngân sách $budget msg)"
    $PY main_churn_engine.py --dataset code --nodes $N --nq 200 \
        --median-session $MED --duration "$dur" --session-dist weibull \
        --weibull-shape "$shape" --meta-anchors "$r" --repair-interval "$tau" \
        --seed "$seed" --out "$f" \
        > "isoshapelog_k${tag}_r${r}_s${seed}.txt" 2>&1 \
        || echo "  [LỖI] k=$shape r=$r s=$seed"
}

echo ""
for cfg in "0.3 11" "0.5 20" "1.0 110"; do
    set -- $cfg; shape=$1; budget=$2
    echo "########## κ=$shape, ngân sách $budget msg/doc/22h ##########"
    for s in $SEEDS; do
        for r in 1 2 3 4; do run "$shape" "$r" "$budget" "$s" & wait_slot; done
    done
    wait
    echo ""
done

echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, json, re, statistics as st
from collections import defaultdict

g = defaultdict(list)
for f in glob.glob('isoshape_k*_r*_s*.json'):
    m = re.match(r'isoshape_k(\d+)_r(\d+)_s(\d+)\.json', f)
    if not m:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    h = d['history']
    if not h or 'phase' not in h[-1]:
        continue
    k = float(m.group(1)) / (10 ** (len(m.group(1)) - 1))
    r = int(m.group(2))
    pts = [x for x in h if x.get('epoch', 0) > 0]
    if pts:
        g[(k, r)].append({'avail': min(x['meta_physical'] for x in pts),
                          'sem': h[-1]['final_r5'], 'ratio': h[-1]['ratio'],
                          'fp': h[-1]['footprint']})

if not g:
    print('  chưa có dữ liệu')
else:
    for k in sorted({a for a, _ in g}):
        lbl = {0.3: 'đuôi rất nặng', 0.5: 'đo thật (bài dùng)',
               1.0: 'phân bố mũ (giả định văn liệu)'}.get(k, '')
        print()
        print('=' * 70)
        print(f'κ = {k}  —  {lbl}')
        print('=' * 70)
        print(f"{'r':>2s} {'bản':>4s} {'avail min':>10s} {'R@5':>7s} {'TỈ LỆ':>7s} "
              f"{'dấu vết':>8s}")
        print('-' * 46)
        best = None
        for r in sorted(b for a, b in g if a == k):
            v = g[(k, r)]
            a = lambda x: st.mean(y[x] for y in v)
            ok = a('avail') >= 99.0
            mark = '' if ok else '  < 99%'
            print(f"{r:>2} {5*r:>4} {a('avail'):>9.1f}% {a('sem'):>6.1f}% "
                  f"{a('ratio'):>6.2f}x {a('fp'):>8.2f}{mark}")
            if ok and (best is None or a('ratio') > best[1]):
                best = (r, a('ratio'), a('avail'))
        if best:
            print(f"  -> r={best[0]} tốt nhất: tỉ lệ {best[1]:.2f}x, "
                  f"avail {best[2]:.1f}%")
        else:
            print('  -> không cấu hình nào giữ được 99% ở ngân sách này')

    bests = {}
    for k in sorted({a for a, _ in g}):
        b = None
        for r in sorted(x for a, x in g if a == k):
            v = g[(k, r)]
            if st.mean(y['avail'] for y in v) >= 99.0:
                rt = st.mean(y['ratio'] for y in v)
                if b is None or rt > b[1]:
                    b = (r, rt)
        if b:
            bests[k] = b
    print()
    print('=' * 70)
    print('HƯỚNG DẪN THIẾT KẾ CÓ BỀN QUA HÌNH DẠNG KHÔNG?')
    print('=' * 70)
    for k, (r, rt) in sorted(bests.items()):
        print(f'  κ={k}: r={r} tốt nhất (tỉ lệ {rt:.2f}x)')
    rs = {r for r, _ in bests.values()}
    if len(rs) == 1:
        print()
        print(f'  => BỀN: r={rs.pop()} tốt nhất ở MỌI hình dạng.')
        print('     Hướng dẫn "tiêu ngân sách vào tần suất" phát biểu vô điều kiện được.')
    elif rs:
        print()
        print(f'  => KHÔNG BỀN: r tối ưu đổi từ {min(rs)} tới {max(rs)} theo hình dạng.')
        print('     Hướng dẫn PHẢI kèm điều kiện về phân bố phiên. Đó là giới hạn')
        print('     thật của kết luận hiện tại, và phải nêu trong bài.')
EOF
