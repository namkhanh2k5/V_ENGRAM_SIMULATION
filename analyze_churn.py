#!/usr/bin/env python3
"""
Phân tích churn: sửa chữa thưa đến đâu thì hỏng?

Trục chính là CHU KỲ SỬA CHỮA so với median session. Không quét session time
vì mô phỏng không thứ nguyên — quét session mà buộc mọi tham số theo nó thì
ba mức cho kết quả y hệt.

    python3 analyze_churn.py
"""
import glob
import json
import statistics as st
from collections import defaultdict


def main():
    runs = defaultdict(list)          # (r, repair_interval) -> [trạng thái cuối]
    ses = None
    for f in glob.glob('churn_*.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cfg, hist = d['config'], d['history']
        if not hist or 'random_r5' not in hist[-1]:
            continue
        rec = dict(hist[-1])
        rec['_duration'] = cfg['duration']          # cần để chuẩn hoá lưu lượng
        # MC3: hist[-1] là MỘT pha cụ thể trong chu kỳ sửa chữa, không đại diện.
        # Giữ cả loạt để báo min/mean, vì min mới là con số deployment phải chịu.
        _m = [h for h in hist if h.get('epoch', 0) > 0]
        for key, out in [('meta_physical', 'phys'), ('meta_routable', 'route')]:
            v = [h[key] for h in _m if key in h]
            if v:
                rec[out + '_mean'] = sum(v) / len(v)
                rec[out + '_min'] = min(v)
        v = [h['final_r5'] for h in _m if 'final_r5' in h]
        if v:
            rec['r5_min'] = min(v)
        runs[(cfg['median_session'], cfg['meta_anchors'],
              cfg['repair_interval'])].append(rec)

    if not runs:
        print('Chưa có file churn_*.json hợp lệ.')
        return

    # File từ các lần chạy median session KHÁC NHAU không so được với nhau.
    # Chọn mức có nhiều dữ liệu nhất, và nói rõ nếu đang bỏ qua mức khác.
    by_ses = defaultdict(int)
    for (ms, _, _), v in runs.items():
        by_ses[ms] += len(v)
    ses = max(by_ses, key=lambda m: by_ses[m])
    if len(by_ses) > 1:
        bỏ = {m: n for m, n in by_ses.items() if m != ses}
        print(f'*** CẢNH BÁO: có file từ {len(by_ses)} mức median session khác nhau.')
        print(f'    Dùng {ses:.0f}ph ({by_ses[ses]} lần chạy), BỎ QUA: '
              + ', '.join(f'{m:.0f}ph ({n})' for m, n in sorted(bỏ.items())))
        print(f'    Các mức đó thuộc thí nghiệm khác, không so chung được.')
        print(f'    Xoá bằng: rm -f $(ls churn_*.json | grep -v "_ses{ses:.0f}_")')
        print()
    runs = {(r, rep): v for (ms, r, rep), v in runs.items() if ms == ses}

    print('=' * 100)
    print(f'CHURN: median session {ses:.0f} phút, Weibull (khớp Stutzbach & Rejaie IMC\'06)')
    print('Trục chính: CHU KỲ SỬA CHỮA so với tốc độ churn')
    print('=' * 100)
    print(f"{'cấu hình':26s} {'n':>2s} {'meta':>7s} {'semantic':>10s} {'random':>8s} "
          f"{'TỈ LỆ':>7s} {'dấu vết':>8s} {'msg/doc/22h':>12s}")
    print('-' * 100)
    print('  (lưu lượng CHUẨN HOÁ theo thời lượng: chu kỳ sửa dài cần mô phỏng dài,')
    print('   nên số message THÔ không so trực tiếp được)')

    def key_order(k):
        r, rep = k
        # r=1 có sửa trước (dày -> thưa), rồi r=1 không sửa, rồi r=20
        if r == 1 and rep > 0:
            return (0, rep)
        if r == 1:
            return (1, 0)
        return (2, r)

    rows = {}
    for k in sorted(runs, key=key_order):
        r, rep = k
        v = runs[k]
        g = lambda x: st.mean(y[x] for y in v)
        sd = lambda x: st.stdev([y[x] for y in v]) if len(v) > 1 else 0.0
        if rep > 0:
            # BUG cũ: biến này từng đặt tên 'k', ghi đè khoá vòng lặp cũng tên 'k',
            # nên rows[k] về sau dùng khoá float thay vì tuple (r, rep).
            mult = rep / ses                   # chu kỳ sửa TÍNH THEO median session
            lbl = (f'r={r}, sửa mỗi {rep:.0f}ph = {mult:.2g}x med' if mult >= 1
                   else f'r={r}, sửa mỗi {rep:.0f}ph = med/{1/mult:.0f}')
        else:
            lbl = f'r={r}, KHÔNG sửa'
        dur = g('_duration')
        row = {'meta': g('meta_avail'), 'sem': g('final_r5'), 'rnd': g('random_r5'),
               'phys_mean': g('phys_mean') if any('phys_mean' in x for x in v) else None,
               'phys_min': (min(x['phys_min'] for x in v if 'phys_min' in x)
                            if any('phys_min' in x for x in v) else None),
               'route_min': (min(x['route_min'] for x in v if 'route_min' in x)
                             if any('route_min' in x for x in v) else None),
               'ratio': g('ratio'), 'fp': g('footprint'), 'msg': g('repair_msgs'),
               'sem_sd': sd('final_r5'), 'n': len(v), 'rep': rep, 'r': r,
               'dur': dur,
               # chuẩn hoá: message mỗi doc trong 22 giờ, để so với IPFS
               'msg22': g('repair_msgs') / 20000 * 1320 / dur if dur > 0 else 0.0}
        rows[k] = row
        mark = '  <- NGẪU NHIÊN THẮNG' if row['ratio'] < 1.0 else ''
        if row['meta'] < 95 and rep > 0:
            mark = '  <- sửa KHÔNG KỊP'
        print(f"{lbl:26s} {row['n']:>2} {row['meta']:>6.1f}% "
              f"{row['sem']:>6.1f}±{row['sem_sd']:<3.0f} {row['rnd']:>7.1f}% "
              f"{row['ratio']:>6.2f}x {row['fp']:>8.2f} {row['msg22']:>12.2f}{mark}")

    # ---- MC3 + MC12: availability theo pha, và tách physical/routable ----
    has_phase = any(r.get('phys_min') is not None for r in rows.values())
    if has_phase:
        print()
        print('=' * 100)
        print('MC3 + MC12 — AVAILABILITY ĐO TẠI PHA NGẪU NHIÊN, TÁCH HAI ĐẠI LƯỢNG')
        print('=' * 100)
        print('  Bản trước chỉ đo ở mốc epoch, mà mốc đó luôn trùng lúc vừa repair,')
        print('  nên availability luôn 100%. Đo tại pha ngẫu nhiên cho biết con số')
        print('  một truy vấn đến bất kỳ lúc nào sẽ gặp.')
        print()
        print(f"  {'cấu hình':26s} {'phys mean':>10s} {'phys MIN':>9s} "
              f"{'route MIN':>10s} {'R@5 min':>8s}")
        print('  ' + '-' * 70)
        for k in sorted(rows, key=key_order):
            r = rows[k]
            if r.get('phys_min') is None:
                continue
            lbl = (f"r={r['r']}, sửa mỗi {r['rep']:.0f}ph" if r['rep'] > 0
                   else f"r={r['r']}, KHÔNG sửa")
            rmin = min(x['r5_min'] for x in runs[k] if 'r5_min' in x) \
                if any('r5_min' in x for x in runs[k]) else float('nan')
            print(f"  {lbl:26s} {r['phys_mean']:>9.1f}% {r['phys_min']:>8.1f}% "
                  f"{r['route_min']:>9.1f}% {rmin:>7.1f}%")
        print()
        print('  phys  = anchor còn giữ record (repair kiểm soát cái này)')
        print('  route = client ĐỊNH TUYẾN tới được (người dùng thấy cái này)')
        print('  Hai cột lệch nhau nghĩa là record còn sống nhưng lookup không tới.')

    # ---- tìm ngưỡng: chu kỳ sửa THƯA NHẤT còn giữ được availability ----
    repaired = sorted([v for k, v in rows.items() if v['r'] == 1 and v['rep'] > 0],
                      key=lambda x: x['rep'])
    # dùng MIN theo pha nếu có, vì giá trị cuối là một pha may mắn
    ok = [v for v in repaired
          if (v['phys_min'] if v.get('phys_min') is not None else v['meta']) >= 99.0]
    print()
    print('=' * 100)
    print('NGƯỠNG SỬA CHỮA')
    print('=' * 100)
    if ok:
        loosest = max(ok, key=lambda x: x['rep'])
        k = loosest['rep'] / ses
        print(f'  Chu kỳ sửa THƯA NHẤT đã thử mà còn giữ availability >= 99%: '
              f'{loosest["rep"]:.0f} phút = {k:.2g} lần median session')
        if loosest is max(repaired, key=lambda x: x['rep']):
            print(f'  *** ĐÂY LÀ MỨC THƯA NHẤT TRONG DẢI QUÉT — ngưỡng thật nằm NGOÀI. ***')
            print(f'      Để so với IPFS (republish mỗi 22 giờ = {1320/ses:.0f}x median),')
            print(f'      cần quét tới {int(1320)}ph. Thêm: --repair-interval 480, 960, 1440.')
        print(f'    availability {loosest["meta"]:.1f}% | recall {loosest["sem"]:.1f}% | '
              f'tỉ lệ {loosest["ratio"]:.2f}x | {loosest["msg"]:,.0f} message')
        tight = min(repaired, key=lambda x: x['rep']) if repaired else loosest
        if tight['rep'] < loosest['rep']:
            print(f'  So với sửa dày nhất ({tight["rep"]:.0f}ph): '
                  f'lưu lượng {tight["msg22"]:.1f} -> {loosest["msg22"]:.1f} msg/doc/22h '
                  f'(giảm {100*(1-loosest["msg22"]/max(tight["msg22"],1e-9)):.0f}%), '
                  f'tỉ lệ {tight["ratio"]:.2f}x -> {loosest["ratio"]:.2f}x')
    else:
        print('  KHÔNG chu kỳ nào giữ được availability >= 99%. Cần sửa dày hơn nữa.')

    # ---- so hai lối thiết kế ----
    r20 = next((v for k, v in rows.items() if v['r'] == 20), None)
    r1no = next((v for k, v in rows.items() if v['r'] == 1 and v['rep'] == 0), None)
    if ok and r20:
        best = max(ok, key=lambda x: x['rep'])
        print()
        print('=' * 100)
        print('HAI LỐI MUA ĐỘ BỀN')
        print('=' * 100)
        print(f"{'':28s} {'r=1 + sửa':>14s} {'r=20 không sửa':>16s}")
        print('-' * 62)
        for lbl, ka, kb, fmt in [('metadata availability', best['meta'], r20['meta'], '{:.1f}%'),
                                 ('Recall@5 semantic', best['sem'], r20['sem'], '{:.1f}%'),
                                 ('Recall@5 ngẫu nhiên', best['rnd'], r20['rnd'], '{:.1f}%'),
                                 ('TỈ LỆ sem/rand', best['ratio'], r20['ratio'], '{:.2f}x'),
                                 ('dấu vết (bản/doc)', best['fp'], r20['fp'], '{:.2f}'),
                                 ('message sửa chữa', best['msg'], r20['msg'], '{:,.0f}')]:
            print(f'{lbl:28s} {fmt.format(ka):>14s} {fmt.format(kb):>16s}')
        print()
        if best['ratio'] >= 1.0 > r20['ratio']:
            print('  => r=1+sửa GIỮ lợi thế ngữ nghĩa; r=20 MẤT (ngẫu nhiên thắng).')
            print('     Cùng độ bền, nhưng một bên còn cơ chế còn bên kia thì không.')
        elif best['ratio'] > r20['ratio'] * 1.3:
            print(f'  => r=1+sửa giữ lợi thế cao hơn {best["ratio"]/r20["ratio"]:.1f} lần.')
        if r20['fp'] > 0:
            print(f'     Dấu vết nhỏ hơn {r20["fp"]/max(best["fp"],1e-9):.1f} lần.')
        per_22h = best['msg22']
        print(f'     Cái giá: {best["msg"]:,.0f} message trong {best["dur"]:.0f} phút '
              f'= {per_22h:.1f} msg/doc mỗi 22 giờ.')
        print()
        print('  ĐỐI CHIẾU IPFS, quy về cùng đơn vị thời gian:')
        print(f'    V-Engram r=1 + sửa : {per_22h:.1f} msg/doc mỗi 22 giờ')
        print(f'    IPFS r=20 republish: 20.0 msg/doc mỗi 22 giờ')
        if per_22h < 20:
            print(f'    => lưu lượng ÍT HƠN {20/max(per_22h,1e-9):.1f} lần, '
                  f'dấu vết nhỏ hơn {r20["fp"]/max(best["fp"],1e-9):.1f} lần')
        else:
            print(f'    => lưu lượng chênh {100*(per_22h/20-1):+.0f}%, '
                  f'dấu vết nhỏ hơn {r20["fp"]/max(best["fp"],1e-9):.1f} lần')
        print(f'       và giữ được lợi thế {best["ratio"]:.2f}x thay vì {r20["ratio"]:.2f}x.')
    if r1no:
        print()
        print(f'  Đối chứng r=1 KHÔNG sửa: availability {r1no["meta"]:.1f}%, '
              f'recall {r1no["sem"]:.1f}% — cho biết sửa chữa đáng giá bao nhiêu.')


if __name__ == '__main__':
    main()