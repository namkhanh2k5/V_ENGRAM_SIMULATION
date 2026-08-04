#!/bin/bash
# ============================================================================
# QUÉT HÌNH DẠNG PHÂN BỐ PHIÊN — trục THẬT SỰ độc lập
#
#   tmux new -s shape
#   source venv/bin/activate
#   PARALLEL=4 bash run_shape_sweep.sh 2>&1 | tee shape.log
#
# VÌ SAO CẦN: quét median session KHÔNG kiểm được gì. Mô hình churn chỉ có một
# thang thời gian — median session — và mọi khoảng khác trong mô phỏng đều đặt
# theo nó (epoch, warmup, duration, chu kỳ sửa, TTL). Nên availability chỉ phụ
# thuộc TỈ SỐ chu_kỳ/median, và ba lần chạy ở median 60/120/240 trả về số
# TRÙNG KHÍT tới chữ số cuối. Chúng là một thí nghiệm đo bằng ba đơn vị.
#
# Trục thật sự độc lập là HÌNH DẠNG phân bố phiên. Weibull shape k điều khiển
# độ nặng của đuôi:
#     k = 0,3  đuôi rất nặng — đa số node cực ổn định, thiểu số ra vào liên tục
#     k = 0,5  giá trị bài đang dùng, khớp Stutzbach & Rejaie
#     k = 0,7  đuôi nhẹ hơn
#     k = 1,0  phân bố MŨ — giả định của Li et al. và Bamboo, mà đo thật cho
#              thấy là SAI, nhưng mọi bài mô phỏng kinh điển đều dùng
#
# Câu hỏi: ngưỡng chu_kỳ/median = 4 có phụ thuộc hình dạng không?
#
#   Nếu KHÔNG: quy tắc bền, phát biểu được cho mọi mạng bất kể phân bố phiên.
#              Đó là kết quả mạnh, vì nó nói ngưỡng chỉ phụ thuộc median.
#   Nếu CÓ   : quy tắc chỉ đúng cho k=0,5, và bài phải nêu rõ điều kiện.
#              Cũng đáng biết — nhất là vì k=1,0 là giả định mà nửa văn liệu dùng.
#
# Đây là câu hỏi mà quét median KHÔNG trả lời được, vì đổi median chỉ đổi đơn vị
# còn đổi shape thì đổi thật sự động lực học.
#
# Ước tính với PARALLEL=4: 4 shape x 4 tỉ số x 3 seed = 48 lần chạy, ~2,5 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2"
MED=120          # cố định, vì đổi nó không đổi gì ngoài đơn vị

grep -q "keep-alive vẫn tốn một message" main_churn_engine.py || {
    echo "main_churn_engine.py chưa vá bug lazy+TTL — tải bản mới"; exit 1; }
grep -q "weibull-shape" main_churn_engine.py || {
    echo "main_churn_engine.py không có cờ --weibull-shape"; exit 1; }
echo "✓ engine đủ điều kiện"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local shape=$1 rep=$2 seed=$3
    local tag; tag=$(echo "$shape" | tr -d '.')
    local f="shape_k${tag}_rep${rep}_s${seed}.json"
    [ -f "$f" ] && { echo "  [skip] k=$shape rep=$rep s=$seed"; return; }
    local dur=$((MED*6)); local need=$((rep*3))
    [ "$need" -gt "$dur" ] && dur=$need
    $PY main_churn_engine.py --dataset code --nodes $N --nq 200 \
        --median-session $MED --duration "$dur" --session-dist weibull \
        --weibull-shape "$shape" \
        --meta-anchors 1 --repair-interval "$rep" --seed "$seed" \
        --out "$f" > "shapelog_k${tag}_rep${rep}_s${seed}.txt" 2>&1 \
        || echo "  [LỖI] k=$shape rep=$rep s=$seed"
}

echo ""
echo "########## QUÉT SHAPE x TỈ SỐ ##########"
echo "  median cố định $MED phút; tỉ số 2x, 3x, 4x, 5x"
for s in $SEEDS; do
    for k in 0.3 0.5 0.7 1.0; do
        for mult in 2 3 4 5; do
            run "$k" $((MED*mult)) "$s" & wait_slot
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

MED = 120
g = defaultdict(list)
for f in glob.glob('shape_k*_rep*_s*.json'):
    m = re.match(r'shape_k(\d+)_rep(\d+)_s(\d+)\.json', f)
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
    rep = int(m.group(2))
    pts = [x for x in h if x.get('epoch', 0) > 0]
    if pts:
        g[(k, rep / MED)].append({
            'avail': min(x['meta_physical'] for x in pts),
            'sem': h[-1]['final_r5'], 'ratio': h[-1]['ratio'],
            'fp': h[-1]['footprint']})

if not g:
    print('  chưa có dữ liệu')
else:
    shapes = sorted({k for k, _ in g})
    mults = sorted({m for _, m in g})
    print('AVAILABILITY MIN theo (hình dạng, tỉ số chu kỳ/median)')
    print()
    print(f"{'shape k':>9s} " + ' '.join(f'{m:>7.0f}x' for m in mults) + '   ngưỡng')
    print('-' * (12 + 8 * len(mults) + 10))
    thresholds = {}
    for k in shapes:
        cells, thr = [], None
        for m in mults:
            v = g.get((k, m), [])
            if not v:
                cells.append('   --  '); continue
            a = st.mean(x['avail'] for x in v)
            cells.append(f'{a:>6.1f}%')
            if a >= 99.0:
                thr = m
        thresholds[k] = thr
        note = {0.5: '  <- bài dùng', 1.0: '  <- giả định mũ'}.get(k, '')
        print(f'{k:>9.1f} ' + ' '.join(cells) +
              f"   {thr if thr else '<2':>4}x{note}")

    print()
    vals = [t for t in thresholds.values() if t]
    if len(vals) >= 2:
        if max(vals) == min(vals):
            print(f'  => NGƯỠNG KHÔNG PHỤ THUỘC HÌNH DẠNG: {max(vals):.0f}x ở mọi k.')
            print('     Quy tắc "chu kỳ = 4x median" bền qua cả phân bố mũ, tức nó')
            print('     dùng được cho các mô hình churn trong văn liệu chứ không chỉ')
            print('     cho phân bố Weibull mà đo thật ủng hộ.')
        else:
            print(f'  => NGƯỠNG PHỤ THUỘC HÌNH DẠNG: {min(vals):.0f}x tới {max(vals):.0f}x.')
            print('     Quy tắc phải nêu kèm điều kiện về phân bố phiên. Đáng chú ý vì')
            print('     k=1,0 (mũ) là giả định của Li et al. và Bamboo.')
    print()
    print('  Đuôi nặng (k nhỏ) nghĩa là đa số node rất ổn định còn thiểu số ra vào')
    print('  liên tục. Đuôi nhẹ (k=1) nghĩa là mọi node rời đi với cùng tốc độ.')
EOF
