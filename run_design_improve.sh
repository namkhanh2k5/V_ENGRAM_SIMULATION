#!/bin/bash
# ============================================================================
# BA THÍ NGHIỆM CẢI THIỆN THIẾT KẾ + MỘT LẦN CHẠY LẠI BẮT BUỘC
#
#   tmux new -s improve
#   source venv/bin/activate
#   PARALLEL=4 bash run_design_improve.sh 2>&1 | tee improve.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep -E "main_simulation|main_churn" | grep -v grep | wc -l
#   ls result_*_LT*_*_nq500.json 2>/dev/null | wc -l          # A, cần 20
#   ls churn_code_*_ttl*_*.json 2>/dev/null | wc -l           # B, cần 12
#   ls result_*_L5_K*_MA1_T*_m512_s*_nq500.json | wc -l       # C
#   ls rmax_k20_R*.txt 2>/dev/null | wc -l                    # D, cần 4
#   tail -5 improve.log
#
# XEM KẾT QUẢ
#   bash run_design_improve.sh 2>&1 | tail -60
#
# ---------------------------------------------------------------------------
# A. NGÂN SÁCH TRẢ VỀ CỦA NODE (local top-κ) — CHƯA AI QUÉT
#
# Churn engine không có bước node lọc top-30 và cho recall ~85%; sweep simulator
# có bước đó và cho 80%. Chênh 5 điểm là cái giá của việc mỗi node chỉ trả về
# 30 ứng viên tốt nhất thay vì tất cả.
#
# Bước lọc tồn tại để giảm byte trả về, và κ=30 chưa bao giờ được biện minh
# bằng số. Nếu κ=50 hay 100 lấy lại phần lớn 5 điểm đó với chi phí byte nhỏ thì
# đó là cải thiện thiết kế thật, rẻ, chưa khai thác.
#
# ---------------------------------------------------------------------------
# B. TTL — THAM SỐ TỰ DO CHƯA TỐI ƯU
#
# Ở r=1, L=5 thì dấu vết danh nghĩa là 5 bản, nhưng đo được 6,2. Dôi 24% là do
# TTL = 2,2 lần chu kỳ sửa: khi anchor dịch sang node khác, bản cũ còn nán lại
# tới 2,2 chu kỳ. Mà Mục r* chứng minh dấu vết lớn hơn thì tỉ lệ thấp hơn.
#
# Hạ TTL xuống 1,2 hoặc 1,5 lần chu kỳ sẽ giảm dấu vết và TĂNG tỉ lệ, đổi lại
# rủi ro hết hạn trước khi kịp gia hạn. Chưa ai quét.
#
# ---------------------------------------------------------------------------
# C. (T, K) Ở CÙNG NGÂN SÁCH NODE — câu hỏi iso-cost cho discovery
#
# T và K đều mua recall bằng cách chạm thêm node. Bảng budget sweep quét từng
# cái riêng, chưa ai hỏi cặp nào hiệu quả nhất ở CÙNG số node chạm. Đây đúng là
# câu hỏi iso-traffic đã làm cho churn, nhưng cho tầng discovery.
#
# Ngân sách danh nghĩa L*K*T = 800: (K=20,T=8) hiện dùng, (K=40,T=4),
# (K=80,T=2), (K=10,T=16), (K=160,T=1).
#
# ---------------------------------------------------------------------------
# D. CHẠY LẠI BẢNG R_max — BẮT BUỘC
#
# Bảng R_max đo lookup nào chạm trần, mà PLACEMENT_K vừa hạ từ 300 xuống 20.
# Lookup xin 20 ứng viên hội tụ nhanh hơn hẳn lookup xin 300, nên bảng đó đo
# bằng cấu hình cũ và có thể đổi hẳn.
#
# Ước tính với PARALLEL=4: A 20 + B 12 + C 15 + D 4 = 51 lần chạy, ~3 giờ.
# ============================================================================
set -u
PY=python3
PARALLEL=${PARALLEL:-4}
N=10000
SEEDS="20235956 1 2 3 4"

for f in main_simulation_v2.py main_churn_engine.py main_simulation.py; do
    [ -f "$f" ] || { echo "THIẾU $f"; exit 1; }
done
grep -q "keep-alive vẫn tốn một message" main_churn_engine.py || {
    echo "main_churn_engine.py chưa vá bug lazy+TTL"; exit 1; }
echo "✓ đủ công cụ"

wait_slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done; }

# ------------------------------------------------------------------ A
echo ""
echo "########## A. QUÉT local top-κ ##########"
for s in $SEEDS; do
    for lt in 10 30 50 100; do
        f="result_code_N${N}_L5_K20_MA1_T8_m512_LT${lt}_s${s}_nq500.json"
        [ -f "$f" ] && continue
        $PY main_simulation_v2.py --dataset code --nodes $N --nq 500 \
            --num-tables 5 --k-query 20 --meta-anchors 1 --multi-probe 8 \
            --use-pq --pq-variant m512 --seed "$s" --local-topk "$lt" \
            --out "$f" >/dev/null 2>&1 &
        wait_slot
    done
done
wait

# ------------------------------------------------------------------ B
echo ""
echo "########## B. QUÉT TTL (chu kỳ sửa cố định 480ph) ##########"
for s in 20235956 1 2; do
    for ttlmult in 12 15 22 30; do
        ttl=$((480*ttlmult/10))
        f="churn_ttl${ttlmult}_s${s}.json"
        [ -f "$f" ] && continue
        $PY main_churn_engine.py --dataset code --nodes $N --nq 200 \
            --median-session 120 --duration 1440 --session-dist weibull \
            --meta-anchors 1 --repair-interval 480 --ttl "$ttl" \
            --seed "$s" --out "$f" > "ttllog_${ttlmult}_s${s}.txt" 2>&1 &
        wait_slot
    done
done
wait

# ------------------------------------------------------------------ C
echo ""
echo "########## C. (T,K) Ở CÙNG NGÂN SÁCH L*K*T = 800 ##########"
for s in 20235956 1 2; do
    for cfg in "10 16" "20 8" "40 4" "80 2" "160 1"; do
        set -- $cfg; K=$1; T=$2
        f="result_code_N${N}_L5_K${K}_MA1_T${T}_m512_s${s}_nq500.json"
        [ -f "$f" ] && continue
        $PY main_simulation_v2.py --dataset code --nodes $N --nq 500 \
            --num-tables 5 --k-query "$K" --meta-anchors 1 --multi-probe "$T" \
            --use-pq --pq-variant m512 --seed "$s" >/dev/null 2>&1 &
        wait_slot
    done
done
wait

# ------------------------------------------------------------------ D
echo ""
echo "########## D. CHẠY LẠI R_max VỚI PLACEMENT_K=20 ##########"
for R in 5 10 15 20; do
    f="rmax_k20_R${R}.txt"
    [ -s "$f" ] && grep -q "R_max" "$f" && { echo "  [skip] R=$R"; continue; }
    echo "  R_max=$R ..."
    PLACEMENT_MODE=deterministic PLACEMENT_K=20 FETCH_TOP=1 PARALLEL_ADC=1 \
        R_MAX=$R timeout 7200 $PY main_simulation.py --dataset code --nodes $N \
        --seed 20235956 --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 100 \
        > "$f" 2>&1 || echo "  [LỖI] R=$R"
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, json, re, statistics as st
from collections import defaultdict

def load(pat, keyfn):
    g = defaultdict(list)
    for f in glob.glob(pat):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        k = keyfn(f, d)
        if k is not None:
            g[k].append(d)
    return g

# ---- A ----
print('=' * 74)
print('A. NGÂN SÁCH TRẢ VỀ CỦA NODE (local top-κ)')
print('=' * 74)
g = load('result_code_N10000_L5_K20_MA1_T8_m512_LT*_s*_nq500.json',
         lambda f, d: d.get('local_topk'))
if g:
    print(f"{'κ':>4s} {'Recall@5':>12s} {'ứng viên':>10s} {'byte/query':>12s} "
          f"{'Δrecall':>8s}")
    print('-' * 52)
    base = None
    for k in sorted(g):
        v = g[k]
        r = st.mean(x['recall5'] for x in v)
        c = st.mean(x['mean_unique_candidates'] for x in v)
        nd = st.mean(x['mean_nodes_touched'] for x in v)
        by = nd * (512 + k * 24)          # query PQ gửi đi + κ tag trả về
        if base is None:
            base = r
        print(f'{k:>4} {r:>11.1f}% {c:>10.0f} {by/1024:>10.0f} KB {r-base:>+7.1f}')
    ks = sorted(g)
    if 30 in ks and max(ks) > 30:
        r30 = st.mean(x['recall5'] for x in g[30])
        rmx = st.mean(x['recall5'] for x in g[max(ks)])
        print()
        print(f'  κ=30 -> κ={max(ks)}: recall {r30:.1f}% -> {rmx:.1f}% ({rmx-r30:+.1f})')
        if rmx - r30 > 1.5:
            print(f'  => Nới κ đáng giá. Cân nhắc đổi mặc định và nêu trong bài.')
        else:
            print(f'  => κ=30 đã gần bão hoà. Lựa chọn hiện tại được biện minh.')
else:
    print('  chưa có dữ liệu')

# ---- B ----
print()
print('=' * 74)
print('B. TTL (chu kỳ sửa 480ph cố định)')
print('=' * 74)
gb = load('churn_ttl*_s*.json',
          lambda f, d: (m := re.match(r'churn_ttl(\d+)_s', f)) and int(m.group(1)))
if gb:
    print(f"{'TTL/chu kỳ':>11s} {'avail min':>10s} {'R@5':>7s} {'TỈ LỆ':>7s} "
          f"{'dấu vết':>8s} {'msg/22h':>9s}")
    print('-' * 58)
    for k in sorted(gb):
        v = gb[k]
        def a(fn):
            return st.mean(fn(d) for d in v)
        mn = a(lambda d: min(x['meta_physical'] for x in d['history']
                             if x.get('epoch', 0) > 0))
        last = [d['history'][-1] for d in v]
        print(f'{k/10:>10.1f}x {mn:>9.1f}% '
              f'{st.mean(x["final_r5"] for x in last):>6.1f}% '
              f'{st.mean(x["ratio"] for x in last):>6.2f}x '
              f'{st.mean(x["footprint"] for x in last):>8.2f} '
              f'{st.mean(x["repair_msgs"] for x in last)/20000*1320/1440:>9.1f}')
    ok = [k for k in gb
          if st.mean(min(x['meta_physical'] for x in d['history']
                         if x.get('epoch', 0) > 0) for d in gb[k]) >= 99.0]
    if ok:
        best = min(ok)
        cur = 22
        if best < cur:
            rb = st.mean(d['history'][-1]['ratio'] for d in gb[best])
            rc = st.mean(d['history'][-1]['ratio'] for d in gb[cur]) if cur in gb else 0
            print()
            print(f'  TTL nhỏ nhất còn giữ 99%: {best/10:.1f}x (hiện dùng 2,2x)')
            print(f'  Tỉ lệ {rc:.2f}x -> {rb:.2f}x nếu hạ TTL. '
                  f'{"ĐÁNG SỬA" if rb > rc + 0.05 else "không đáng"}')
else:
    print('  chưa có dữ liệu')

# ---- C ----
print()
print('=' * 74)
print('C. (T,K) Ở CÙNG NGÂN SÁCH DANH NGHĨA L·K·T = 800')
print('=' * 74)
gc = defaultdict(list)
for f in glob.glob('result_code_N10000_L5_K*_MA1_T*_m512_s*_nq500.json'):
    m = re.search(r'_K(\d+)_MA1_T(\d+)_', f)
    if not m:
        continue
    K, T = int(m.group(1)), int(m.group(2))
    if 5 * K * T != 800:
        continue
    try:
        gc[(K, T)].append(json.load(open(f)))
    except Exception:
        pass
if gc:
    print(f"{'K':>4s} {'T':>3s} {'Recall@5':>12s} {'node%':>7s} {'cand%':>7s} "
          f"{'R@5/node%':>10s}")
    print('-' * 50)
    best = None
    for (K, T) in sorted(gc):
        v = gc[(K, T)]
        r = st.mean(x['recall5'] for x in v)
        nd = st.mean(x['pct_network_touched'] for x in v)
        cd = st.mean(100 * x['mean_unique_candidates'] / 20000 for x in v)
        eff = r / nd if nd else 0
        cur = '  <- hiện dùng' if (K, T) == (20, 8) else ''
        print(f'{K:>4} {T:>3} {r:>11.1f}% {nd:>6.1f}% {cd:>6.1f}% {eff:>10.1f}{cur}')
        if best is None or r > best[1]:
            best = ((K, T), r)
    if best and best[0] != (20, 8):
        rc = st.mean(x['recall5'] for x in gc[(20, 8)]) if (20, 8) in gc else 0
        print()
        print(f'  Tốt nhất: K={best[0][0]}, T={best[0][1]} -> {best[1]:.1f}% '
              f'so với {rc:.1f}% ở cấu hình hiện tại ({best[1]-rc:+.1f})')
        if best[1] - rc > 1.5:
            print('  => Cấu hình hiện tại KHÔNG tối ưu ở cùng ngân sách. Đáng đổi.')
    elif best:
        print()
        print('  => Cấu hình hiện tại (K=20, T=8) là tốt nhất ở cùng ngân sách.')
        print('     Lựa chọn được biện minh bằng số, không phải bằng mặc định.')
else:
    print('  chưa có dữ liệu')

# ---- D ----
print()
print('=' * 74)
print('D. R_max VỚI PLACEMENT_K=20 (bảng cũ đo ở K=300)')
print('=' * 74)
rows = []
for f in sorted(glob.glob('rmax_k20_R*.txt')):
    m = re.match(r'rmax_k20_R(\d+)\.txt', f)
    t = open(f, errors='ignore').read()
    g2 = lambda p: (float(mm.group(1).replace(',', ''))
                    if (mm := re.search(p, t)) else None)
    rows.append((int(m.group(1)),
                 g2(r'Lookup ch[aạ]m tr[aầ]n\s*:\s*\d+/\d+\s*\(([\d.]+)%'),
                 g2(r'Recall@5\s*:\s*([\d.]+)%'),
                 g2(r'RPC/query\s*:\s*([\d.,]+)')))
if rows:
    print(f"{'R_max':>6s} {'lookup chạm trần':>18s} {'Recall@5':>10s} {'RPC':>8s}")
    print('-' * 46)
    for R, cap, rec, rpc in rows:
        print(f'{R:>6} {(cap if cap is not None else 0):>17.1f}% '
              f'{(rec or 0):>9.1f}% {(rpc or 0):>8.0f}')
    print()
    print('  Bài đang ghi: 65,5% chạm trần ở R=5, 1,2% ở R=10, 0% từ R=15.')
    print('  Nếu số mới khác thì bảng R_max phải cập nhật.')
else:
    print('  chưa có dữ liệu')
EOF
