#!/usr/bin/env python3
"""
So các sweep dưới LOOKUP LÝ TƯỞNG HOÁ với WALK KADEMLIA THẬT.

Câu hỏi: chênh lệch 3 điểm đo ở một cấu hình có GIỮ NGUYÊN qua các cấu hình
không? Nếu không, các khác biệt mà sweep báo cáo bị nén và kết luận có thể đổi.

    python3 analyze_validate.py
"""
import glob
import re

# Số từ mô phỏng lý tưởng hoá (main_simulation_v2.py, 10 seed, 500 query)
IDEAL = {
    (5, 1, 20, 8):  {'sem': 80.0, 'rnd': 34.5, 'note': 'cấu hình chốt'},
    (5, 3, 20, 8):  {'sem': 80.4, 'rnd': 70.6, 'note': 'giữa dải r*'},
    (8, 5, 20, 8):  {'sem': 92.3, 'rnd': 95.6, 'note': 'QUA điểm giao r*'},
    (5, 1, 20, 1):  {'sem': 39.8, 'rnd': None, 'note': 'không thăm dò'},
    (5, 1, 100, 1): {'sem': 65.2, 'rnd': None, 'note': 'mở rộng thay thăm dò'},
    (12, 1, 20, 8): {'sem': 95.3, 'rnd': 89.6, 'note': 'L cao'},
}


def read_recall(path):
    """Lấy Recall@5 từ log của main_simulation.py."""
    try:
        txt = open(path, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    m = re.search(r'Recall@5\s*:\s*([\d.]+)%', txt)
    return float(m.group(1)) if m else None


def main():
    rows = []
    for cfg, ideal in IDEAL.items():
        L, r, K, T = cfg
        sem = read_recall(f'validate_L{L}_r{r}_K{K}_T{T}.txt')
        rnd = read_recall(f'validate_L{L}_r{r}_K{K}_T{T}_RAND.txt')
        rows.append((cfg, ideal, sem, rnd))

    import glob as _g, re as _re
    nq = '?'
    for f in _g.glob('validate_*.txt'):
        txt = open(f, errors='ignore').read()
        m = _re.search(r'(\d+)\s*truy v[aấ]n|Ch[aạ]y\s+(\d+)\s*query|nq[=: ]+(\d+)', txt)
        if m:
            nq = next(g for g in m.groups() if g)
            break
    print('=' * 104)
    print(f'LÝ TƯỞNG HOÁ vs WALK KADEMLIA THẬT   (nq={nq})')
    print('  Lý tưởng: 10 seed × 500 query | Walk thật: 1 seed × nq query')
    print('=' * 104)
    print(f"{'cấu hình':22s} {'semantic':>19s} {'random':>19s} "
          f"{'tỉ lệ lt':>9s} {'tỉ lệ thật':>11s}")
    print(f"{'':22s} {'lý tưởng  thật  mất':>19s} {'lý tưởng  thật  mất':>19s}")
    print('-' * 104)

    drops = []
    for (L, r, K, T), ideal, sem, rnd in rows:
        lbl = f'L={L} r={r} K={K} T={T}'
        if sem is None:
            print(f'{lbl:22s} {"(chưa chạy)":>19s}')
            continue
        d_sem = ideal['sem'] - sem
        drops.append((lbl, d_sem))
        s_str = f"{ideal['sem']:>7.1f} {sem:>6.1f} {d_sem:>+5.1f}"
        if rnd is not None and ideal['rnd'] is not None:
            d_rnd = ideal['rnd'] - rnd
            r_str = f"{ideal['rnd']:>7.1f} {rnd:>6.1f} {d_rnd:>+5.1f}"
            ratio_i = ideal['sem'] / ideal['rnd']
            ratio_r = sem / rnd if rnd > 0 else float('nan')
            mark = ''
            if (ratio_i >= 1.0) != (ratio_r >= 1.0):
                mark = '  <- ĐỔI PHÍA so với mốc 1!'
            print(f'{lbl:22s} {s_str:>19s} {r_str:>19s} '
                  f'{ratio_i:>8.2f}x {ratio_r:>10.2f}x{mark}')
        else:
            print(f'{lbl:22s} {s_str:>19s} {"--":>19s} {"--":>9s} {"--":>11s}')

    if not drops:
        print('\nChưa có kết quả nào.')
        return

    print()
    print('=' * 104)
    print('CHÊNH LỆCH CÓ ĐỀU KHÔNG?')
    print('=' * 104)
    vals = [d for _, d in drops]
    print(f'  Mất recall do định tuyến thật, theo cấu hình:')
    for lbl, d in drops:
        print(f'    {lbl:22s} {d:+.1f} điểm')
    spread = max(vals) - min(vals)
    print()
    print(f'  Trung bình {sum(vals)/len(vals):+.1f} điểm | '
          f'dải {min(vals):+.1f} đến {max(vals):+.1f} | chênh lệch {spread:.1f} điểm')
    print()
    # SO TỈ LỆ, KHÔNG so recall tuyệt đối.
    #
    # Bản trước lập luận sai: cho rằng random không định tuyến nên không mất gì
    # dưới walk thật, rồi dùng cột random làm "sàn nhiễu". Thực tế random CŨNG
    # mất, và mất nhiều hơn khi L·r lớn. Lý do: INGEST cũng dùng lookup xấp xỉ,
    # nên anchor bị đặt lệch và hai semantic key dễ rơi vào cùng node hơn — số
    # anchor PHÂN BIỆT (A) giảm, kéo P_rand giảm theo. Random không định tuyến
    # khi truy vấn, nhưng dữ liệu nó tìm đã được GHI bằng định tuyến xấp xỉ.
    #
    # Hệ quả: recall tuyệt đối không so được, nhưng TỈ LỆ thì có — tử và mẫu
    # cùng chịu một hiệu ứng.
    pairs = [(f'L={L} r={r} K={K} T={T}',
              ideal['sem'] / ideal['rnd'], sem / rnd)
             for (L, r, K, T), ideal, sem, rnd in rows
             if sem and rnd and ideal['rnd'] and rnd > 0]
    print()
    print('=' * 104)
    print('TỈ LỆ CÓ BỀN VỮNG KHÔNG?  (đại lượng mà mọi kết luận của bài dựa vào)')
    print('=' * 104)
    if not pairs:
        print('  Chưa đủ cặp semantic/random để so tỉ lệ.')
        return
    print(f"{'cấu hình':22s} {'lý tưởng':>10s} {'thật':>8s} {'đổi':>8s}  ghi chú")
    print('-' * 104)
    crossed = []
    for lbl, ri, rr in pairs:
        note = ''
        if (ri >= 1.0) != (rr >= 1.0):
            note = 'ĐỔI PHÍA qua mốc 1!'
            crossed.append(lbl)
        elif ri >= 1.0:
            note = 'trên mốc 1 ở cả hai'
        else:
            note = 'dưới mốc 1 ở cả hai'
        print(f'{lbl:22s} {ri:>9.2f}x {rr:>7.2f}x {rr-ri:>+7.2f}  {note}')

    shifts = [abs(rr - ri) for _, ri, rr in pairs]
    print()
    print(f'  Tỉ lệ đổi nhiều nhất: {max(shifts):.2f} | trung bình {sum(shifts)/len(shifts):.2f}')
    print()
    if crossed:
        print(f'  *** {len(crossed)} cấu hình ĐỔI PHÍA qua mốc 1: ' + ', '.join(crossed))
        print('      Bảng r* phải sửa — điểm giao dịch chuyển dưới định tuyến thật.')
    elif max(shifts) < 0.15:
        print('  => TỈ LỆ BỀN VỮNG. Định tuyến thật lấy đi recall ở CẢ HAI phía, nên')
        print('     tỉ lệ gần như không đổi, và không cấu hình nào đổi phía qua mốc 1.')
        print('     Mọi kết luận của bài dựa trên tỉ lệ GIỮ NGUYÊN dưới định tuyến thật.')
    else:
        print(f'  => Tỉ lệ đổi tới {max(shifts):.2f}. Đáng kể — phải nêu trong bài.')

    print()
    print('  Recall TUYỆT ĐỐI thì khác: định tuyến thật lấy đi '
          f'{min(v for _, v in drops):.1f}–{max(v for _, v in drops):.1f} điểm, '
          f'trung bình {sum(v for _, v in drops)/len(drops):.1f}.')
    print('  Mức mất TĂNG theo L·r, vì càng nhiều anchor càng nhiều cơ hội bị đặt')
    print('  lệch. Nên số recall trong các sweep là CẬN TRÊN, và cận trên đó lỏng')
    print('  hơn ở cấu hình dấu vết lớn.')


if __name__ == '__main__':
    main()