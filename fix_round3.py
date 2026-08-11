#!/usr/bin/env python3
"""
Tính lại hai chỗ analyze_round3.py làm sai. Không chạy lại thí nghiệm.

    python3 fix_round3.py

1. Khối C dùng sai phép kiểm. Nó so chênh trung bình với 2 lần ĐỘ LỆCH CHUẨN
   của từng nhóm, trong khi phép kiểm đúng dùng SAI SỐ CHUẨN CỦA HIỆU. Và vì
   hai nhóm chạy trên CÙNG BỘ SEED nên phải ghép cặp — mạnh hơn hẳn.

2. Khối B trộn nhóm A vào dòng stable/all, vì nhóm A chạy với STOP_RULE và
   FRONTIER_SCOPE mặc định (đúng bằng stable/all) nên sinh cùng tên file JSON.
   Script này đếm riêng để biết mức nhiễm bẩn.
"""
import glob
import json
import math
import re
import statistics as st


def read(f):
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    m = re.search(r'Recall@5\s*:\s*([\d.]+)%', t)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- C
print('=' * 72)
print('C. MARGIN-RANKED PROBING — TÍNH LẠI BẰNG KIỂM ĐỊNH GHÉP CẶP')
print('=' * 72)

pairs = []
for f in sorted(glob.glob('r3_C_margin_s*.txt')):
    # re.search trả None khi không khớp; gọi thẳng .group() sẽ gãy. Đây là lỗi
    # đã sửa ở analyze_mc1.py, lặp lại vì mình chép mẫu cũ.
    ms = re.search(r'_s(\d+)\.txt$', f)
    if ms is None:
        print(f'  [bỏ qua] {f}: tên không có _s<seed>.txt')
        continue
    rm = read(f)
    rr = read(f.replace('_margin_', '_random_'))
    if rm is not None and rr is not None:
        pairs.append((ms.group(1), rm, rr))
    else:
        print(f'  [bỏ qua] seed {ms.group(1)}: thiếu một trong hai chế độ')

if not pairs:
    print('  chưa có đủ dữ liệu')
else:
    print(f"{'seed':>12s} {'margin':>8s} {'random':>8s} {'chênh':>8s}")
    print('-' * 40)
    for s, a, b in pairs:
        print(f'{s:>12s} {a:>7.1f}% {b:>7.1f}% {a-b:>+7.1f}')
    d = [a - b for _, a, b in pairs]
    n = len(d)
    md = st.mean(d)
    sdd = st.stdev(d) if n > 1 else 0.0
    se = sdd / math.sqrt(n) if n > 1 else float('nan')
    t = md / se if se else float('nan')
    print('-' * 40)
    print(f"{'trung bình':>12s} {'':>8s} {'':>8s} {md:>+7.1f}")
    print()
    print(f'  n = {n} cặp, độ lệch chuẩn của HIỆU = {sdd:.2f}, SE = {se:.2f}')
    print(f'  t ghép cặp = {t:.2f}, bậc tự do = {n-1}')
    # ngưỡng t hai phía, alpha = 0.05
    TCRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
             6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}
    tc = TCRIT.get(n - 1, 2.0)
    print(f'  t tới hạn (alpha=0,05, hai phía) = {tc}')
    print()
    if abs(t) > tc and md > 0:
        print('  => Margin HƠN ngẫu nhiên, có ý nghĩa thống kê.')
        print('     Giữ lập luận về margin trong bài, kèm con số này.')
    elif abs(t) > tc:
        print('  => Ngẫu nhiên HƠN margin, có ý nghĩa thống kê. Phải bỏ lập luận.')
    else:
        print('  => Chưa đạt ý nghĩa thống kê ở mức 0,05.')
        need = math.ceil((tc * sdd / max(abs(md), 1e-9)) ** 2) if md else 0
        print(f'     Cần khoảng {need} cặp seed để phát hiện hiệu {md:+.1f} điểm.')
        print('     Hoặc hạ giọng: nêu xu hướng, không nêu kết luận.')
    print()
    same = sum(1 for x in d if x > 0)
    print(f'  Dấu: margin thắng {same}/{n} seed.')
    if same == n:
        print('  Margin thắng ở MỌI seed — kiểm định dấu cho p = '
              f'{2**-n:.3f}, chặt hơn t-test khi n nhỏ.')

# ---------------------------------------------------------------- B
print()
print('=' * 72)
print('B. NGUỒN GỐC CÁC FILE stable/all')
print('=' * 72)

def _m(rows, k):
    """Trung bình an toàn: bỏ qua file thiếu khoá.

    random_slots và random_unique KHÔNG đi lookup nên không có peer set, do đó
    không có jaccard_mean. Truy cập x['jaccard_mean'] thẳng sẽ gãy — đúng lỗi
    vừa gặp."""
    xs = [x[k] for x in rows if x.get(k) is not None]
    return st.mean(xs) if xs else float('nan')


# nhóm B ghi ra termabl_*.json (bản script mới); nhóm A ghi result_full_*.json
grpB = []
for f in glob.glob('termabl_*.json'):
    try:
        grpB.append(json.load(open(f)))
    except Exception:
        pass

grpA = []
for f in glob.glob('result_full_code_N10000_*_nq500.json'):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get('stop_rule') == 'stable' and d.get('frontier_scope') == 'all':
        grpA.append(d)

print(f'  termabl_*.json (nhóm B, chạy riêng)      : {len(grpB)} file')
print(f'  result_full_*  (nhóm A, stable/all)      : {len(grpA)} file')
print()

if grpB:
    from collections import defaultdict
    by = defaultdict(list)
    for d in grpB:
        by[(d.get('stop_rule'), d.get('frontier_scope'))].append(d)
    print('  NHÓM B — dùng số này để tách hai trục:')
    print(f"    {'cấu hình':16s} {'n':>2s} {'recall':>8s} {'Jaccard':>8s} "
          f"{'XOR rank':>9s} {'RPC':>7s}")
    print('    ' + '-' * 56)
    for k in sorted(by):
        v = by[k]
        print(f"    {k[0]+'/'+k[1]:16s} {len(v):>2} {_m(v,'recall_at_5'):>7.1f}% "
              f"{_m(v,'jaccard_mean'):>8.3f} {_m(v,'xor_rank_mean'):>9.1f} "
              f"{_m(v,'disc_rpcs'):>7.0f}")
    if ('stable', 'all') in by and len(by) == 4:
        b = _m(by[('stable', 'all')], 'recall_at_5')
        print()
        print(f"    mốc stable/all           {b:>7.1f}%")
        for k, lbl in [(('exhaust', 'all'), 'chỉ đổi ĐIỀU KIỆN DỪNG'),
                       (('stable', 'topk'), 'chỉ đổi PHẠM VI HỎI'),
                       (('exhaust', 'topk'), 'đổi cả hai')]:
            if k in by:
                print(f"    {lbl:24s} {_m(by[k],'recall_at_5') - b:>+6.1f} điểm")
else:
    print('  Chưa có termabl_*.json — khối B chưa chạy bằng script mới.')
    print('  Các file result_full hiện có đều là nhóm A (chạy với STOP_RULE và')
    print('  FRONTIER_SCOPE mặc định), nên KHÔNG dùng để tách hai trục được.')
    print()
    print('  Chạy lại khối B:')
    print('    rm -f r3_B_*.txt')
    print('    PARALLEL=4 bash run_round3.sh 2>&1 | tee round3b.log')
    if grpA:
        print()
        print(f"  Tham khảo nhóm A (stable/all, {len(grpA)} file, gồm cả baseline):")
        print(f"    recall  {_m(grpA,'recall_at_5'):>6.1f}%")
        print(f"    Jaccard {_m(grpA,'jaccard_mean'):>6.3f}  "
              f"(random_slots/random_unique không có, đã bỏ qua)")