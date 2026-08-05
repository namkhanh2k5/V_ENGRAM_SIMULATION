#!/bin/bash
# ============================================================================
# OVER-DRAW AND SELECT — cải tiến bài HỨA BA LẦN mà chưa bao giờ đo
#
#   tmux new -s overdraw
#   source venv/bin/activate
#   PARALLEL=4 bash run_overdraw.sh 2>&1 | tee overdraw.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation_v2 | grep -v grep | wc -l
#   ls result_*_POOL*_s*_nq300.json 2>/dev/null | wc -l      # cần 40
#   ls result_code_N10000_L5_K20_MA1_T8_m512_s*_nq300.json | wc -l   # mốc, cần 10
#   tail -5 overdraw.log
#
# XEM KẾT QUẢ
#   bash run_overdraw.sh 2>&1 | tail -30
#
# ---------------------------------------------------------------------------
# Ý TƯỞNG: p_l đo được dao động 18,1% tới 46,4% trong MỘT lượt rút L=5, tức
# năm ma trận ngẫu nhiên là một mẫu tuỳ tiện từ phân bố rộng. Rút nhiều hơn rồi
# giữ những cái tốt nhất sẽ nâng p trung bình mà KHÔNG đổi giao thức, KHÔNG làm
# khoá phụ thuộc dữ liệu, và KHÔNG tốn gì ở thời điểm truy vấn.
#
# Bài đã đề xuất điều này ba lần trong Discussion và Future Work nhưng chưa
# bao giờ đo. Đo thử ở scifact N=3000 cho tầng discovery 93,3% -> 94,7% (pool
# 12) -> 96,9% (pool 20), và mức cải thiện TĂNG theo pool, đúng như phải có.
#
# TÁCH TẬP: ma trận được chọn trên 150 query CUỐI, đánh giá trên 300 query ĐẦU.
# Không tách thì là chọn trên chính dữ liệu test và con số sẽ lạc quan giả.
#
# TIÊU CHÍ CHỌN: số bit khớp trung bình trong c bit đầu giữa sketch của query và
# sketch của láng giềng thật. Không dùng "trùng prefix chính xác" vì ở c=16 tỉ
# lệ đó chỉ 0,1-3%, quá thưa để phân biệt với vài trăm mẫu.
#
# Ước tính với PARALLEL=4: 5 mức pool x 2 corpus x 5 seed = 50 lần chạy, ~2 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
NQ=300              # chừa 200 query cuối cho hiệu chuẩn
CALIB=150
SEEDS="20235956 1 2 3 4"

grep -q "table-pool" main_simulation_v2.py || {
    echo "main_simulation_v2.py chưa có --table-pool — tải bản mới"; exit 1; }
grep -q "bit khớp" main_simulation_v2.py || {
    echo "main_simulation_v2.py dùng tiêu chí cũ — tải bản mới"; exit 1; }
echo "✓ engine đủ điều kiện"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local ds=$1 pool=$2 seed=$3
    local sfx=""; [ "$pool" -gt 0 ] && sfx="_POOL${pool}"
    local f="result_${ds}_N${N}_L5_K20_MA1_T8_m512${sfx}_s${seed}_nq${NQ}.json"
    [ -f "$f" ] && { echo "  [skip] $ds pool=$pool s=$seed"; return; }
    $PY main_simulation_v2.py --dataset "$ds" --nodes $N --nq $NQ \
        --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
        --use-pq --pq-variant m512 --seed "$seed" \
        --table-pool "$pool" --calib-queries $CALIB \
        >/dev/null 2>&1 || echo "  [LỖI] $ds pool=$pool s=$seed"
}

echo ""
for ds in code scifact; do
    echo "########## $ds ##########"
    for s in $SEEDS; do
        for pool in 0 8 12 20 30; do run "$ds" "$pool" "$s" & wait_slot; done
    done
    wait
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, json, re, statistics as st
from collections import defaultdict

g = defaultdict(list)
for f in glob.glob('result_*_N10000_L5_K20_MA1_T8_m512*_s*_nq300.json'):
    m = re.match(r'result_(\w+?)_N10000.*?(?:_POOL(\d+))?_s(\d+)_nq300\.json', f)
    if not m:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    g[(d['dataset'], d.get('table_pool', 0))].append(d)

for ds in sorted({a for a, _ in g}):
    print()
    print('=' * 72)
    print(f'{ds}')
    print('=' * 72)
    print(f"{'pool':>5s} {'n':>2s} {'discovery':>14s} {'final R@5':>14s} "
          f"{'Δ so pool=0':>12s}")
    print('-' * 52)
    base = None
    for pool in sorted(b for a, b in g if a == ds):
        v = g[(ds, pool)]
        r1 = st.mean(x['reachable_recall5'] for x in v)
        r3 = st.mean(x['recall5'] for x in v)
        s1 = st.stdev([x['reachable_recall5'] for x in v]) if len(v) > 1 else 0
        s3 = st.stdev([x['recall5'] for x in v]) if len(v) > 1 else 0
        if base is None:
            base = r3
        lbl = 'không' if pool == 0 else str(pool)
        print(f'{lbl:>5s} {len(v):>2} {r1:>8.1f}±{s1:<4.1f} {r3:>8.1f}±{s3:<4.1f} '
              f'{r3-base:>+11.1f}')
    pools = sorted(b for a, b in g if a == ds and b > 0)
    if pools and base is not None:
        best = max(pools, key=lambda p: st.mean(x['recall5'] for x in g[(ds, p)]))
        rb = st.mean(x['recall5'] for x in g[(ds, best)])
        sd = st.stdev([x['recall5'] for x in g[(ds, best)]]) if len(g[(ds, best)]) > 1 else 0
        print()
        print(f'  Tốt nhất: pool={best} -> {rb:.1f}% so với {base:.1f}% '
              f'({rb-base:+.1f} điểm)')
        if rb - base > 2 * sd:
            print(f'  => Cải thiện VƯỢT hai lần độ lệch chuẩn. Đáng đưa vào bài.')
        elif rb - base > 0:
            print(f'  => Cải thiện {rb-base:+.1f} nhưng chưa vượt 2σ ({sd:.1f}). '
                  f'Cần thêm seed.')
        else:
            print(f'  => Không cải thiện. Ý tưởng không hiệu quả ở quy mô này.')

print()
print('=' * 72)
print('CÁCH ĐỌC')
print('=' * 72)
print('  Cột discovery là recall trước rerank — nơi ma trận chiếu tác động trực')
print('  tiếp. Cột final đã qua PQ rerank nên một phần cải thiện bị nuốt.')
print('  Nếu discovery tăng mà final không tăng thì nút thắt là rerank, không')
print('  phải ma trận, và đó cũng là kết quả đáng nói.')
print()
print('  Chi phí: chọn ma trận chạy MỘT LẦN lúc khởi tạo mạng, không tốn gì ở')
print('  thời điểm truy vấn. Nên bất kỳ cải thiện nào cũng là miễn phí.')
EOF
