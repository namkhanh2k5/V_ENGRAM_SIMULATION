#!/bin/bash
# ============================================================================
# CHẠY LẠI DISCRETE-EVENT SAU KHI SỬA ĐIỀU KIỆN DỪNG (nhận xét mục 1)
#
#   tmux new -s stoprule
#   source venv/bin/activate
#   PARALLEL=4 bash run_stoprule.sh 2>&1 | tee stoprule.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep main_simulation.py | grep -v grep | wc -l
#   ls sr_exhaust_*_s*.txt 2>/dev/null | wc -l        # A, cần 40
#   ls sr_cmp_*_s*.txt 2>/dev/null | wc -l            # B, cần 6
#   ls sr_cost_*.txt 2>/dev/null | wc -l              # C, cần 2
#   tail -5 stoprule.log
#
# XEM KẾT QUẢ
#   bash run_stoprule.sh 2>&1 | tail -40
#
# ---------------------------------------------------------------------------
# VẤN ĐỀ: điều kiện dừng cũ là "top-k không đổi giữa hai vòng". Mỗi vòng chỉ
# query alpha=3 node còn top-k=20, nên vòng lặp có thể dừng khi 14 trong 20 node
# của top-k CHƯA từng được query — mà bất kỳ node nào trong đó cũng có thể biết
# một node gần target hơn. B_t = B_{t-1} không kéo theo đã vét cạn.
#
# SỬA: điều kiện Kademlia chuẩn — chỉ query node NẰM TRONG top-k, và dừng khi
# top-k đã query hết. Tương đương B_t = B_{t-1} AND B_t ⊆ V.
#
# ĐO THỬ Ở N=2000, ba seed: recall Y HỆT ở cả ba (84,0 / 83,0 / 84,0), RPC tăng
# đều ~57%, node chạm không đổi. Tức các query thêm chỉ XÁC NHẬN điều đã biết.
# Nếu quy mô thật cũng vậy thì: con số recall trong bài ĐÚNG, con số RPC bị
# ĐẾM THIẾU, và chỉ bảng chi phí phải sửa.
#
# Sweep dùng oracle (main_simulation_v2.py) KHÔNG bị ảnh hưởng — chúng không đi
# walk, nên không cần chạy lại.
#
# Ước tính với PARALLEL=4: A 40 lần x ~35 phút = ~6 giờ (chậm hơn trước vì
# nhiều vòng hơn), B 6 lần ~1 giờ, C 2 lần ~2 giờ. TỔNG ~9 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2 3 4 5 6 7 8 9"

grep -q "STOP_RULE" src/routing.py || {
    echo "src/routing.py chưa có STOP_RULE — tải bản mới rồi chạy lại"; exit 1; }
echo "✓ routing.py có cờ STOP_RULE (mặc định exhaust)"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

# --------------------------------------------------------- A. MC1 chạy lại
echo ""
echo "########## A. MC1 HEADLINE VỚI ĐIỀU KIỆN ĐÚNG ##########"
run_a() {
    local mode=$1 seed=$2 extra=$3
    local f="sr_exhaust_${mode}_s${seed}.txt"
    [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] $f"; return; }
    env SKIP_PAYLOAD=1 STOP_RULE=exhaust ROUTING_MODE="$mode" $extra \
        timeout 10800 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$seed" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 500 \
        > "$f" 2>&1 || echo "  [LỖI] $mode s=$seed"
}
echo "-- semantic --"
for s in $SEEDS; do run_a semantic "$s" "" & wait_slot; done; wait

MATCH=$($PY - <<'EOF'
import glob, re, statistics as st
v = []
for f in glob.glob('sr_exhaust_semantic_s*.txt'):
    m = re.search(r'Unique nodes contacted\s+([\d,]+)', open(f, errors='ignore').read())
    if m:
        v.append(float(m.group(1).replace(',', '')))
print(int(round(st.mean(v))) if v else 504)
EOF
)
echo "   semantic chạm TB $MATCH node -> dùng cho random_unique"
for m in keyed_lookup random_slots random_unique; do
    echo "-- $m --"
    ex=""; [ "$m" = "random_unique" ] && ex="MATCH_UNIQUE_NODES=$MATCH"
    for s in $SEEDS; do run_a "$m" "$s" "$ex" & wait_slot; done; wait
done

# ------------------------------------------- B. đối chiếu hai điều kiện
echo ""
echo "########## B. ĐỐI CHIẾU unchanged vs exhaust (3 seed) ##########"
for s in 20235956 1 2; do
    for rule in unchanged exhaust; do
        f="sr_cmp_${rule}_s${s}.txt"
        [ -s "$f" ] && grep -q "Recall@5" "$f" && continue
        SKIP_PAYLOAD=1 STOP_RULE=$rule timeout 10800 $PY main_simulation.py \
            --dataset code --nodes $N --seed "$s" --k-query 20 --multi-probe 8 \
            --meta-anchors 1 --nq 500 > "$f" 2>&1 &
        wait_slot
    done
done
wait

# ------------------------------------------------- C. bảng chi phí (có payload)
echo ""
echo "########## C. BẢNG CHI PHÍ VỚI PAYLOAD ##########"
for rule in unchanged exhaust; do
    f="sr_cost_${rule}.txt"
    [ -s "$f" ] && grep -q "BẢNG CHI PHÍ" "$f" && { echo "  [skip] $f"; continue; }
    echo "  $rule (1-2 giờ) ..."
    PLACEMENT_MODE=deterministic PLACEMENT_K=20 FETCH_TOP=1 PARALLEL_ADC=1 \
        STOP_RULE=$rule R_MAX=20 timeout 14400 $PY main_simulation.py \
        --dataset code --nodes $N --seed 20235956 --k-query 20 --multi-probe 8 \
        --meta-anchors 1 --nq 100 > "$f" 2>&1 || echo "  [LỖI] cost $rule"
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, re, statistics as st

def read(f):
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    g = lambda p: (float(m.group(1).replace(',', ''))
                   if (m := re.search(p, t)) else None)
    return {'recall': g(r'Recall@5\s*:\s*([\d.]+)%'),
            'rpc': g(r'RPC/query\s*:\s*([\d,.]+)'),
            'rounds': g(r'Rounds/query\s*:\s*([\d,.]+)'),
            'nodes': g(r'Unique nodes contacted\s+([\d,]+)'),
            'p50': g(r'Latency p50 \(ms\)\s+([\d,.]+)')}

# ---- B: đối chiếu ----
print('=' * 74)
print('B. ĐỐI CHIẾU HAI ĐIỀU KIỆN DỪNG — code, N=10.000, 500 query, 3 seed')
print('=' * 74)
agg = {}
for rule in ('unchanged', 'exhaust'):
    v = [r for f in glob.glob(f'sr_cmp_{rule}_s*.txt') if (r := read(f)) and r['recall']]
    if v:
        agg[rule] = v
if len(agg) == 2:
    print(f"{'đại lượng':16s} {'unchanged':>12s} {'exhaust':>12s} {'đổi':>9s}")
    print('-' * 54)
    for k, lbl in [('recall', 'Recall@5 (%)'), ('rounds', 'vòng/query'),
                   ('rpc', 'RPC/query'), ('nodes', 'node chạm')]:
        a = st.mean(x[k] for x in agg['unchanged'] if x[k])
        b = st.mean(x[k] for x in agg['exhaust'] if x[k])
        print(f'{lbl:16s} {a:>12.1f} {b:>12.1f} {100*(b-a)/a:>+8.0f}%')
    ra = st.mean(x['recall'] for x in agg['unchanged'])
    rb = st.mean(x['recall'] for x in agg['exhaust'])
    sd = st.stdev([x['recall'] for x in agg['exhaust']]) if len(agg['exhaust']) > 1 else 0
    print()
    if abs(rb - ra) < max(0.5, sd):
        print(f'  => Recall KHÔNG đổi ({ra:.1f} -> {rb:.1f}, trong nhiễu {sd:.1f}).')
        print('     Điều kiện cũ tìm đúng vùng, chỉ ĐẾM THIẾU chi phí xác nhận.')
        print('     Kết luận về cơ chế giữ nguyên; chỉ bảng chi phí phải sửa.')
    else:
        print(f'  => Recall ĐỔI {rb-ra:+.1f} điểm. Điều kiện cũ tìm SAI vùng,')
        print('     nên mọi số discrete-event phải thay, kể cả MC1.')
else:
    print('  chưa đủ dữ liệu')

# ---- A: MC1 ----
print()
print('=' * 74)
print('A. MC1 VỚI ĐIỀU KIỆN ĐÚNG')
print('=' * 74)
IDEAL = {'semantic': 80.0, 'keyed_lookup': 33.4,
         'random_slots': 34.5, 'random_unique': 22.1}
OLD = {'semantic': 73.8, 'keyed_lookup': 33.1,
       'random_slots': 34.2, 'random_unique': 22.9}
print(f"{'chế độ':16s} {'n':>2s} {'mới':>13s} {'cũ':>7s} {'tham chiếu':>11s} {'RPC':>7s}")
print('-' * 62)
new = {}
for mode in ('semantic', 'keyed_lookup', 'random_slots', 'random_unique'):
    v = [r for f in glob.glob(f'sr_exhaust_{mode}_s*.txt') if (r := read(f)) and r['recall']]
    if not v:
        print(f'{mode:16s} {"(chưa chạy)":>16s}'); continue
    m = st.mean(x['recall'] for x in v)
    sd = st.stdev([x['recall'] for x in v]) if len(v) > 1 else 0
    rp = st.mean(x['rpc'] for x in v if x['rpc'])
    new[mode] = m
    print(f'{mode:16s} {len(v):>2} {m:>7.1f}±{sd:<4.1f} {OLD[mode]:>7.1f} '
          f'{IDEAL[mode]:>11.1f} {rp:>7.0f}')
if 'semantic' in new:
    print()
    print(f"{'baseline':18s} {'tỉ lệ mới':>10s} {'tỉ lệ cũ':>9s} {'lý tưởng':>9s}")
    print('-' * 50)
    for mode in ('keyed_lookup', 'random_slots', 'random_unique'):
        if mode in new and new[mode]:
            print(f'{mode:18s} {new["semantic"]/new[mode]:>9.2f}x '
                  f'{OLD["semantic"]/OLD[mode]:>8.2f}x '
                  f'{IDEAL["semantic"]/IDEAL[mode]:>8.2f}x')

# ---- C: chi phí ----
print()
print('=' * 74)
print('C. BẢNG CHI PHÍ')
print('=' * 74)
for rule in ('unchanged', 'exhaust'):
    r = read(f'sr_cost_{rule}.txt')
    if r and r['rpc']:
        p50 = f"{r['p50']/1000:.1f}s" if r['p50'] else '?'
        print(f"  {rule:11s} RPC {r['rpc']:>7,.0f} | vòng {r['rounds'] or 0:>7,.0f} "
              f"| p50 {p50:>6s}")
print()
print('  Bài đang ghi discovery 1.145 RPC. Nếu exhaust cho số cao hơn nhiều thì')
print('  bảng chi phí VÀ bảng hiệu suất per-RPC đều phải cập nhật.')
EOF
