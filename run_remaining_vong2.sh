#!/bin/bash
# ============================================================================
# NĂM MỤC CÒN LẠI: MC3, MC4, MC7, MC9, MC12
#
#   tmux new -s v2b
#   source venv/bin/activate
#   PARALLEL=4 bash run_remaining_vong2.sh 2>&1 | tee remaining.log
#
# MC3+MC12 — churn đo tại PHA NGẪU NHIÊN trong chu kỳ sửa chữa, và tách
#            physical / routable availability. Bản trước luôn đo ngay sau
#            repair nên availability luôn 100%; đo theo pha cho thấy min thấp
#            hơn mean, và min mới là con số một deployment phải chịu.
#
# MC4     — so IPFS công bằng. "r=20" trong bảng churn là L*r = 5x20 = 100 vị
#           trí, không phải 20 bản như IPFS. Muốn khớp thì L*r = 20, tức r=4.
#           Chạy CẢ HAI để bảng có dòng đúng nhãn.
#
# MC7     — baseline khớp NGÂN SÁCH TÍNH TOÁN. Các baseline hiện khớp số node,
#           nhưng node semantic chọn giữ nhiều record hơn nên semantic được
#           nhiều ADC computation hơn ở cùng số node.
#
# MC9     — latency 500 query x 5 seed thay vì 20 query 1 seed. p99 với 20 mẫu
#           gần như là max, không dùng được.
#
# Ước tính với PARALLEL=4:
#   A. churn theo pha (MC3+MC12)   3 rep x 3 seed        ~1,5 giờ
#   B. churn cho MC4               3 cấu hình x 3 seed   ~1,2 giờ
#   C. baseline equal-cand (MC7)   2 corpus x 5 seed     ~0,3 giờ
#   D. latency 500q x 5 seed (MC9) 5 lần                 ~2,5 giờ
#   TỔNG ~5 giờ
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000

for f in main_churn_engine.py main_simulation.py main_simulation_v2.py analyze_churn.py; do
    [ -f "$f" ] || { echo "THIẾU $f — tải về rồi chạy lại"; exit 1; }
done
grep -q "random_equalcand" main_simulation_v2.py || {
    echo "main_simulation_v2.py chưa có random_equalcand — bản cũ, tải lại"; exit 1; }
grep -q "phase" main_churn_engine.py || {
    echo "main_churn_engine.py chưa đo theo pha — bản cũ, tải lại"; exit 1; }
echo "✓ đủ công cụ"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

# ------------------------------------------------- A + B. churn (MC3/MC12/MC4)
echo ""
echo "########## A+B. CHURN theo pha, kèm r=4 cho so IPFS ##########"
run_churn() {
    local r=$1 rep=$2 seed=$3
    local m=""; [ "$rep" != "0" ] && m="l"
    local f="churn_code_N${N}_r${r}_ses120_weibull_rep${rep}${m}_s${seed}_nq200.json"
    [ -f "$f" ] && { echo "  [skip] r=$r rep=$rep s=$seed"; return; }
    local dur=720; local need=$((rep*3)); [ "$need" -gt "$dur" ] && dur=$need
    $PY main_churn_engine.py --dataset code --nodes $N --nq 200 \
        --median-session 120 --duration "$dur" --session-dist weibull \
        --meta-anchors "$r" --repair-interval "$rep" --seed "$seed" \
        > "churnlog_r${r}_rep${rep}_s${seed}.txt" 2>&1 \
        || echo "  [LỖI] r=$r rep=$rep s=$seed"
}
for s in 20235956 1 2; do
    for rep in 60 240 960; do
        run_churn 1 "$rep" "$s" & wait_slot          # đề xuất của bài
    done
    # MC4 — ba cấu hình để so IPFS cho đúng:
    #   r=4 + repair 1320ph = ĐÚNG cấu hình IPFS (20 bản, republish 22 giờ)
    #   r=4 không sửa       = cùng số bản nhưng bỏ repair, tách ảnh hưởng repair
    #   r=20 không sửa      = cấu hình cũ trong bài, giữ để đối chiếu
    #                         (đây là L*r = 100 vị trí, KHÔNG phải 20 bản như IPFS)
    run_churn 4 1320 "$s" & wait_slot
    run_churn 4    0 "$s" & wait_slot
    run_churn 20   0 "$s" & wait_slot
done
wait
$PY analyze_churn.py > churn_results_v2.txt 2>&1
echo "-> churn_results_v2.txt"

# ------------------------------------------------------- C. MC7 equal-candidate
echo ""
echo "########## C. MC7 — baseline khớp ngân sách ỨNG VIÊN ##########"
for s in 20235956 1 2 3 4; do
    for ds in code scifact; do
        f="result_${ds}_N${N}_L5_K20_MA1_T8_m512_RANDCAND_s${s}_nq500.json"
        [ -f "$f" ] && continue
        $PY main_simulation_v2.py --dataset "$ds" --nodes $N --nq 500 \
            --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
            --use-pq --pq-variant m512 --seed "$s" --routing random_equalcand \
            >/dev/null 2>&1 &
        wait_slot
    done
done
wait
MIN_NQ=500 $PY summarize.py > paper_tables_v2.txt 2>&1
echo "-> paper_tables_v2.txt"

# ------------------------------------------------------------ D. MC9 latency
echo ""
echo "########## D. MC9 — latency 500 query x 5 seed ##########"
for s in 20235956 1 2 3 4; do
    f="lat_s${s}.txt"
    [ -s "$f" ] && grep -q "Latency p50" "$f" && { echo "  [skip] $f"; continue; }
    PLACEMENT_MODE=deterministic PLACEMENT_K=20 FETCH_TOP=1 PARALLEL_ADC=1 \
        timeout 10800 $PY main_simulation.py --dataset code --nodes $N \
        --seed "$s" --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 500 \
        > "$f" 2>&1 || echo "  [LỖI] latency s=$s"
    echo "  xong seed $s"
done

echo ""
echo "########## TỔNG HỢP LATENCY ##########"
$PY - <<'EOF'
import glob, re, statistics as st
rows=[]
for f in sorted(glob.glob('lat_s*.txt')):
    t=open(f,errors='ignore').read()
    g=lambda p:(float(m.group(1).replace(',','')) if (m:=re.search(p,t)) else None)
    r={'p50':g(r'Latency p50 \(ms\)\s+([\d,.]+)'),
       'p95':g(r'Latency p95 \(ms\)\s+([\d,.]+)'),
       'p99':g(r'Latency p99 \(ms\)\s+([\d,.]+)'),
       'rpc':g(r'RPC/query\s*:\s*([\d,.]+)')}
    if r['p50']: rows.append(r)
if not rows:
    print('  chưa có lat_s*.txt')
else:
    print(f"  {len(rows)} seed x 500 query")
    for k in ('p50','p95','p99','rpc'):
        v=[r[k] for r in rows if r[k]]
        if not v: continue
        sd=st.stdev(v) if len(v)>1 else 0.0
        u='ms' if k!='rpc' else ''
        print(f"    {k:4s} {st.mean(v):>9,.0f} ± {sd:>7,.0f} {u}  "
              f"[{min(v):,.0f}, {max(v):,.0f}]")
    print()
    print("  p99 giờ dựa trên 2.500 query thay vì 20, nên dùng được.")
EOF

echo ""
echo "########## XONG — đọc: ##########"
echo "  churn_results_v2.txt   MC3+MC12+MC4"
echo "  paper_tables_v2.txt    MC7 (dòng RANDCAND)"
echo "  output ở trên          MC9 latency"