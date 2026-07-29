#!/usr/bin/env python3
"""
Phân tích thí nghiệm churn.

Câu hỏi: có mua được độ bền mà không phá định tuyến ngữ nghĩa không?

Đọc: trạng thái CUỐI của mỗi lần chạy (sau 3 lần median session), gộp qua seed.

    python3 analyze_churn.py
"""
import glob
import json
import statistics as st
from collections import defaultdict


def label(cfg):
    r = cfg['meta_anchors']
    rep = cfg['repair_interval']
    ses = cfg['median_session']
    if rep <= 0:
        return f'r={r}, không sửa'
    if abs(rep - ses / 4) < 0.51:
        return f'r={r}, sửa mỗi median/4'
    if abs(rep - ses) < 0.51:
        return f'r={r}, sửa mỗi median'
    return f'r={r}, sửa mỗi {rep:.0f}ph'


ORDER = ['r=1, không sửa', 'r=1, sửa mỗi median', 'r=1, sửa mỗi median/4',
         'r=20, không sửa']


def main():
    runs = defaultdict(list)          # (median_session, label) -> [history_cuối]
    firsts = defaultdict(list)
    for f in glob.glob('churn_*.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cfg, hist = d['config'], d['history']
        if not hist:
            continue
        # bỏ file sinh trước khi engine đo baseline ngẫu nhiên — thiếu khoá,
        # và quan trọng hơn là không so được vì thiếu đại lượng quyết định
        if 'random_r5' not in hist[-1] or 'footprint' not in hist[-1]:
            print(f'  [bỏ qua] {f} — sinh bởi engine bản cũ, thiếu random_r5')
            continue
        key = (cfg['median_session'], label(cfg))
        runs[key].append(hist[-1])
        firsts[key].append(hist[0])

    if not runs:
        print('Chưa có file churn_*.json nào.')
        return

    for ses in sorted({k[0] for k in runs}, reverse=True):
        print('=' * 104)
        print(f'MEDIAN SESSION = {ses:.0f} PHÚT   '
              f'(thời lượng mô phỏng {3*ses:.0f} phút = 3 lần thay lượt)')
        print('=' * 104)
        print(f"{'cấu hình':24s} {'n':>2s} {'meta avail':>11s} {'semantic':>10s} "
              f"{'random':>9s} {'TỈ LỆ':>8s} {'dấu vết':>9s} {'msg sửa':>11s}")
        print('-' * 104)
        rows = {}
        for lbl in ORDER:
            v = runs.get((ses, lbl), [])
            if not v:
                continue
            g = lambda k: st.mean(x[k] for x in v)
            sd = lambda k: st.stdev([x[k] for x in v]) if len(v) > 1 else 0.0
            rows[lbl] = {'meta': g('meta_avail'), 'sem': g('final_r5'),
                         'rnd': g('random_r5'), 'ratio': g('ratio'),
                         'fp': g('footprint'), 'msg': g('repair_msgs'),
                         'sem_sd': sd('final_r5'), 'n': len(v)}
            r = rows[lbl]
            mark = ''
            if r['ratio'] < 1.0:
                mark = '  <- NGẪU NHIÊN THẮNG'
            print(f"{lbl:24s} {r['n']:>2} {r['meta']:>10.1f}% "
                  f"{r['sem']:>7.1f}±{r['sem_sd']:<2.0f} {r['rnd']:>8.1f}% "
                  f"{r['ratio']:>7.2f}x {r['fp']:>9.2f} {r['msg']:>11,.0f}{mark}")

        # so trực tiếp hai lối
        a = rows.get('r=1, sửa mỗi median/4')
        b = rows.get('r=20, không sửa')
        c = rows.get('r=1, không sửa')
        if a and b:
            print()
            print(f'  SO HAI LỐI:')
            print(f'    độ bền   : r=1+sửa {a["meta"]:.1f}%  vs  r=20 {b["meta"]:.1f}%')
            print(f'    tỉ lệ    : r=1+sửa {a["ratio"]:.2f}x  vs  r=20 {b["ratio"]:.2f}x')
            if b['fp'] > 0:
                print(f'    dấu vết  : r=1+sửa {a["fp"]:.2f}  vs  r=20 {b["fp"]:.2f} '
                      f'(nhỏ hơn {b["fp"]/max(a["fp"],1e-9):.1f} lần)')
            print(f'    cái giá  : {a["msg"]:,.0f} message sửa chữa vs 0')
            if a['ratio'] >= 1.0 > b['ratio']:
                print(f'    => r=1+sửa GIỮ được lợi thế ngữ nghĩa, r=20 MẤT.')
            elif a['ratio'] > b['ratio'] * 1.2:
                print(f'    => r=1+sửa giữ lợi thế tốt hơn r=20 rõ rệt.')
        if c and a:
            print(f'    sửa chữa đáng giá: {a["sem"]-c["sem"]:+.1f} điểm recall '
                  f'so với không sửa gì')
        print()

    # bảng chi phí sửa chữa theo mức churn
    print('=' * 104)
    print('CHI PHÍ SỬA CHỮA theo mức churn (cấu hình r=1, sửa mỗi median/4)')
    print('=' * 104)
    print(f"{'median session':>16s} {'msg sửa':>12s} {'msg/phút':>11s} "
          f"{'msg/doc':>10s} {'tỉ lệ':>8s}")
    print('-' * 104)
    for ses in sorted({k[0] for k in runs}, reverse=True):
        v = runs.get((ses, 'r=1, sửa mỗi median/4'), [])
        if not v:
            continue
        msg = st.mean(x['repair_msgs'] for x in v)
        ratio = st.mean(x['ratio'] for x in v)
        dur = 3 * ses
        print(f'{ses:>13.0f}ph {msg:>12,.0f} {msg/dur:>11,.0f} '
              f'{msg/20000:>10.2f} {ratio:>7.2f}x')
    print()
    print('  msg/doc tính trên corpus 20.000. Đối chiếu IPFS: republish 20 bản mỗi')
    print('  22 giờ = 20 msg/doc mỗi 22 giờ, trong mạng mà 87,6% session dưới 8 giờ.')


if __name__ == '__main__':
    main()
