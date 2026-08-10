#!/bin/bash
# ============================================================================
# CHẨN ĐOÁN: VÌ SAO ĐIỀU KIỆN DỪNG ĐÚNG LẠI CHO RECALL THẤP HƠN?
#
#   tmux new -s diag
#   source venv/bin/activate
#   PARALLEL=4 bash run_stoprule_diag.sh 2>&1 | tee diag.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation.py | grep -v grep
#   grep -l "Recall@5" diag_*.txt 2>/dev/null | wc -l      # cần 18
#   tail -5 diag.log
#
# XEM KẾT QUẢ
#   bash run_stoprule_diag.sh 2>&1 | tail -30
#
# ---------------------------------------------------------------------------
# QUAN SÁT: ở N=10.000, 500 query, 3 seed:
#     unchanged  recall 72,9%  vòng 211,6  RPC 1.149  node 513,7
#     exhaust    recall 67,8%  vòng 405,5  RPC 1.661  node 494,3
#
# Điều kiện ĐÚNG cho recall THẤP HƠN 5,2 điểm. Có hai giải thích, và chúng đòi
# hai hành động hoàn toàn khác nhau.
#
# GIẢ THUYẾT 1 — R_max đang cắt.
#   exhaust chạy 10,1 vòng mỗi lookup, sát trần R_max=15. Nhiều lookup có thể
#   chạm trần và bị cắt giữa chừng, trả về top-k chưa hoàn chỉnh. unchanged chỉ
#   5,3 vòng nên còn xa trần. Nếu vậy thì phép so KHÔNG công bằng, và nới trần
#   sẽ khôi phục recall.
#   KIỂM: chạy exhaust ở R_max = 15, 30, 60. Nếu recall tăng theo trần thì đúng.
#
# GIẢ THUYẾT 2 — hội tụ chặt làm giảm ĐỘ PHỦ.
#   unchanged query node gần nhất chưa hỏi trong TOÀN BỘ candidates, nên khi
#   top-k đã hỏi hết nó đi tiếp ra node #21, #22... exhaust chỉ query trong
#   top-k rồi dừng. Bản cũ vì thế thăm dò rộng hơn, và số liệu ủng hộ: nó chạm
#   513,7 node phân biệt so với 494,3, dù ít RPC hơn.
#
#   Điều này nối thẳng với lập luận của bài: truy vấn tương tự cần PHỦ VÙNG chứ
#   không cần node gần nhất. T=8 probe lệch nhau một bit, định tuyến càng chính
#   xác thì tám tập node trả về càng trùng nhau, tổng độ phủ càng hẹp.
#   KIỂM: nếu nới trần KHÔNG khôi phục recall, giả thuyết 2 đúng.
#
# NẾU GIẢ THUYẾT 2 ĐÚNG thì kết luận không phải "thuật toán có bug" mà là
# "thuật toán không phải Kademlia chuẩn, và đó là CHỦ Ý" — lúc đó phải mô tả
# lại pseudocode cho đúng thay vì sửa code theo Kademlia.
#
# Ước tính với PARALLEL=4: 18 lần chạy x ~35 phút = ~3 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2"

grep -q "STOP_RULE" src/routing.py || { echo "thiếu STOP_RULE"; exit 1; }
echo "✓ routing.py có STOP_RULE"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

run() {
    local rule=$1 rmax=$2 seed=$3
    local f="diag_${rule}_R${rmax}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] $f"; return; }
    SKIP_PAYLOAD=1 STOP_RULE=$rule R_MAX=$rmax timeout 14400 $PY main_simulation.py \
        --dataset code --nodes $N --seed "$seed" --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 500 > "$f" 2>&1 || echo "  [LỖI] $rule R=$rmax s=$seed"
}

echo ""
echo "########## QUÉT R_max CHO CẢ HAI ĐIỀU KIỆN ##########"
for s in $SEEDS; do
    for rmax in 15 30 60; do
        run exhaust   "$rmax" "$s" & wait_slot
        run unchanged "$rmax" "$s" & wait_slot
    done
done
wait

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, re, statistics as st
from collections import defaultdict

g = defaultdict(list)
for f in glob.glob('diag_*_R*_s*.txt'):
    m = re.match(r'diag_(\w+)_R(\d+)_s(\d+)\.txt', f)
    if not m:
        continue
    t = open(f, encoding='utf-8', errors='ignore').read()
    gg = lambda p: (float(x.group(1).replace(',', ''))
                    if (x := re.search(p, t)) else None)
    r = {'recall': gg(r'Recall@5\s*:\s*([\d.]+)%'),
         'rounds': gg(r'Rounds/query\s*:\s*([\d,.]+)'),
         'rpc': gg(r'RPC/query\s*:\s*([\d,.]+)'),
         'nodes': gg(r'Node chạm/query:\s*([\d,.]+)'),
         'cap': gg(r'Lookup ch[aạ]m tr[aầ]n\s*:\s*\d+/\d+\s*\(([\d.]+)%')}
    if r['recall']:
        g[(m.group(1), int(m.group(2)))].append(r)

if not g:
    print('  chưa có dữ liệu')
else:
    print(f"{'điều kiện':11s} {'R_max':>6s} {'n':>2s} {'Recall@5':>12s} "
          f"{'vòng':>7s} {'RPC':>7s} {'node':>6s} {'chạm trần':>10s}")
    print('-' * 70)
    for rule in ('unchanged', 'exhaust'):
        for rm in sorted(b for a, b in g if a == rule):
            v = g[(rule, rm)]
            a = lambda k: st.mean(x[k] for x in v if x[k] is not None)
            sd = st.stdev([x['recall'] for x in v]) if len(v) > 1 else 0
            cap = a('cap') if any(x['cap'] is not None for x in v) else float('nan')
            print(f"{rule:11s} {rm:>6} {len(v):>2} {a('recall'):>7.1f}±{sd:<4.1f} "
                  f"{a('rounds'):>7.0f} {a('rpc'):>7.0f} {a('nodes'):>6.0f} "
                  f"{cap:>9.1f}%")
        print()

    # phán giả thuyết
    ex = {rm: st.mean(x['recall'] for x in g[('exhaust', rm)])
          for rm in sorted(b for a, b in g if a == 'exhaust')}
    un = {rm: st.mean(x['recall'] for x in g[('unchanged', rm)])
          for rm in sorted(b for a, b in g if a == 'unchanged')}
    print('=' * 70)
    print('PHÁN GIẢ THUYẾT')
    print('=' * 70)
    if len(ex) >= 2:
        lo, hi = min(ex), max(ex)
        gain = ex[hi] - ex[lo]
        print(f'  exhaust: R_max {lo} -> {hi} cho recall {ex[lo]:.1f} -> {ex[hi]:.1f} '
              f'({gain:+.1f})')
        if gain > 2.0:
            print()
            print('  => GIẢ THUYẾT 1 ĐÚNG: R_max đang cắt exhaust.')
            print('     Phép so ở R_max=15 KHÔNG công bằng. Phải báo cáo ở trần đủ lớn,')
            print('     và bảng R_max trong bài phải quét lại cho điều kiện mới.')
            if hi in un and ex[hi] >= un[hi] - 1:
                print(f'     Ở R_max={hi}, exhaust {ex[hi]:.1f} so với unchanged '
                      f'{un[hi]:.1f} — đã bắt kịp.')
        else:
            print()
            print('  => GIẢ THUYẾT 2 ĐÚNG: nới trần KHÔNG khôi phục recall.')
            print('     Điều kiện Kademlia chuẩn hội tụ CHẶT hơn, nên tám probe lệch')
            print('     một bit trả về các tập node TRÙNG NHAU nhiều hơn, và tổng độ')
            print('     phủ hẹp lại. Với truy vấn tương tự, độ phủ mới là thứ cần.')
            print()
            print('     Kết luận: thuật toán cũ KHÔNG phải bug mà là một chiến lược')
            print('     tìm kiếm khác, hợp với tác vụ hơn Kademlia chuẩn. Việc cần làm')
            print('     là MÔ TẢ LẠI pseudocode cho đúng, không phải sửa code.')
    if any(x['nodes'] for rule in ('unchanged', 'exhaust')
           for rm in g if (rule, rm[1]) in g for x in g.get((rule, rm[1]), [])):
        print()
        print('  Cột "node" là bằng chứng phụ: nếu unchanged luôn chạm nhiều node')
        print('  phân biệt hơn ở mọi R_max thì độ phủ đúng là cơ chế.')
EOF
