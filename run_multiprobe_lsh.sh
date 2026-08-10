#!/bin/bash
# ============================================================================
# BASELINE MULTI-PROBE BUCKET-LSH — trả lời comment thêm của thầy
#
#   tmux new -s mplsh
#   source venv/bin/activate
#   bash run_multiprobe_lsh.sh 2>&1 | tee mplsh.log
#
# KIỂM TIẾN ĐỘ
#   ps aux | grep baselines.py | grep -v grep | wc -l
#   ls mplsh_*_T*_s*.txt 2>/dev/null | wc -l          # cần 30
#   tail -5 mplsh.log
#
# XEM KẾT QUẢ
#   bash run_multiprobe_lsh.sh 2>&1 | tail -35
#
# ---------------------------------------------------------------------------
# CÂU HỎI: bài so V-Engram (T=8 probe mỗi bảng) với Bucket-LSH chỉ tra bucket
# CHÍNH XÁC (tương đương T=1). Reviewer sẽ hỏi: "you compared against a
# deliberately weak exact-bucket LSH — what if it also multi-probes?"
#
# CÁCH LÀM: cho Bucket-LSH ĐÚNG cơ chế multi-probe mà V-Engram dùng — lật những
# bit có |projection| nhỏ nhất, tức bit nằm sát siêu phẳng. Quy tắc lấy nguyên
# từ src/routing.py dòng 96, không phải cơ chế khác.
#
# KHỚP NGÂN SÁCH ở hai trục:
#   lookup    : L*T tra cứu ở cả hai bên (5 bảng x 8 probe = 40)
#   candidate : quét b, chọn bề rộng cho pool gần V-Engram nhất
#
# MỘT KHÁC BIỆT CÒN LẠI, PHẢI NÊU TRONG BÀI: mỗi probe của V-Engram là một
# lookup Kademlia trả về K=20 node GẦN NHẤT theo XOR — tức truy vấn bán kính.
# Mỗi probe của Bucket-LSH là một tra cứu KHỚP CHÍNH XÁC một nhãn bucket. Nên
# kể cả khi khớp T, V-Engram vẫn phủ rộng hơn mỗi probe. Multi-probe đóng phần
# "nhiều probe", không đóng phần "bán kính so với khớp chính xác".
#
# Đó chính là điều đáng đo: nếu multi-probe đóng gần hết khoảng cách thì lợi thế
# của định tuyến DHT nhỏ hơn bài đang ngụ ý, và phải nói vậy.
#
# Ước tính: 3 mức T x 2 corpus x 5 seed = 30 lần chạy, ~1,5 giờ.
# ============================================================================
set -u
PY=python3
SEEDS="20235956 1 2 3 4"

grep -q "BUCKET_PROBES" baselines.py || {
    echo "baselines.py chưa có BUCKET_PROBES — tải bản mới rồi chạy lại"; exit 1; }
echo "✓ baselines.py có multi-probe"

for ds in code scifact; do
    vc=$([ "$ds" = "code" ] && echo 4510 || echo 1845)
    for T in 1 4 8; do
        for s in $SEEDS; do
            f="mplsh_${ds}_T${T}_s${s}.txt"
            [ -s "$f" ] && grep -q "Recall@5" "$f" && { echo "  [skip] $f"; continue; }
            echo "  $ds T=$T seed=$s ..."
            CORPUS=$ds BUCKET_PROBES=$T POOL_PER_TABLE=902 \
                VENGRAM_CANDIDATES=$vc RNG_SEED=$s LSH_SEED=$s \
                BUCKET_WIDTHS=8,10,12,14,16 \
                timeout 3600 $PY baselines.py > "$f" 2>&1 \
                || echo "    [LỖI] $ds T=$T s=$s"
        done
    done
done

echo ""
echo "########## TỔNG HỢP ##########"
$PY - <<'EOF'
import glob, re, statistics as st
from collections import defaultdict

# V-Engram ở cùng cấu hình, để so
VENGRAM = {'code': 80.0, 'scifact': 80.9}

g = defaultdict(list)
for f in glob.glob('mplsh_*_T*_s*.txt'):
    m = re.match(r'mplsh_(\w+)_T(\d+)_s(\d+)\.txt', f)
    if not m:
        continue
    t = open(f, encoding='utf-8', errors='ignore').read()
    # dòng bucket-LSH được chọn cho bảng chính
    mm = re.search(r'Bucket-LSH[^\n]*?([\d.]+)%[^\n]*?([\d.]+)%[^\n]*?pool[^\d]*([\d,]+)', t)
    if not mm:
        # thử lấy dòng b= có pool gần ngân sách nhất
        rows = re.findall(r'b=\s*(\d+) T=\d+: Recall@5=\s*([\d.]+)%.*?pool TB=\s*([\d,]+)', t)
        if not rows:
            continue
        vc = 4510 if m.group(1) == 'code' else 1845
        b, r, pool = min(rows, key=lambda x: abs(float(x[2].replace(',', '')) - vc))
        g[(m.group(1), int(m.group(2)))].append(
            {'recall': float(r), 'pool': float(pool.replace(',', '')), 'b': int(b)})
    else:
        g[(m.group(1), int(m.group(2)))].append(
            {'recall': float(mm.group(1)), 'pool': float(mm.group(3).replace(',', '')),
             'b': None})

if not g:
    print('  chưa có dữ liệu')
else:
    for ds in sorted({a for a, _ in g}):
        print()
        print('=' * 70)
        print(f'{ds}  —  V-Engram cùng cấu hình: {VENGRAM.get(ds, 0):.1f}%')
        print('=' * 70)
        print(f"{'T':>3s} {'n':>2s} {'Recall@5':>13s} {'pool':>8s} {'b':>4s} "
              f"{'tỉ lệ V-E':>10s}")
        print('-' * 50)
        base = None
        for T in sorted(b for a, b in g if a == ds):
            v = g[(ds, T)]
            r = st.mean(x['recall'] for x in v)
            sd = st.stdev([x['recall'] for x in v]) if len(v) > 1 else 0
            pool = st.mean(x['pool'] for x in v)
            bs = [x['b'] for x in v if x['b']]
            bb = f"{st.mode(bs)}" if bs else '--'
            if base is None:
                base = r
            ratio = VENGRAM.get(ds, 0) / r if r else 0
            print(f'{T:>3} {len(v):>2} {r:>7.1f}±{sd:<4.1f} {pool:>8.0f} {bb:>4s} '
                  f'{ratio:>9.2f}x')
        Ts = sorted(b for a, b in g if a == ds)
        if len(Ts) >= 2:
            r1 = st.mean(x['recall'] for x in g[(ds, Ts[0])])
            r8 = st.mean(x['recall'] for x in g[(ds, Ts[-1])])
            ve = VENGRAM.get(ds, 0)
            closed = 100 * (r8 - r1) / (ve - r1) if ve > r1 else 0
            print()
            print(f'  Multi-probe đóng {closed:.0f}% khoảng cách tới V-Engram '
                  f'({r1:.1f} -> {r8:.1f}, đích {ve:.1f}).')
            if r8 >= ve - 2:
                print('  => Baseline gần bắt kịp. Lợi thế của định tuyến DHT nhỏ hơn')
                print('     bài đang ngụ ý, và phải nói đúng như vậy.')
            elif closed > 50:
                print('  => Multi-probe đóng phần lớn khoảng cách nhưng chưa hết.')
                print('     Phần còn lại là truy vấn BÁN KÍNH (K node gần nhất) so với')
                print('     KHỚP CHÍNH XÁC một nhãn bucket — đó mới là thứ DHT mang lại.')
            else:
                print('  => Multi-probe đóng được ít. Khoảng cách chủ yếu đến từ')
                print('     cơ chế bán kính, không phải từ số probe.')
EOF
