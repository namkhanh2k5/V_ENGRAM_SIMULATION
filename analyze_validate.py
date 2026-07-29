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

    print('=' * 104)
    print('LÝ TƯỞNG HOÁ vs WALK KADEMLIA THẬT')
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
    if spread < 3.0:
        print('  => Chênh lệch KHÁ ĐỀU qua các cấu hình. Các so sánh giữa cấu hình')
        print('     mà sweep báo cáo GIỮ NGUYÊN dưới định tuyến thật; chỉ giá trị')
        print('     tuyệt đối bị dịch xuống một lượng gần như cố định.')
    else:
        print('  => Chênh lệch THAY ĐỔI RÕ theo cấu hình. Các so sánh giữa cấu hình')
        print('     bị nén hoặc giãn dưới định tuyến thật. Phải nêu rõ trong bài rằng')
        print('     sweep chỉ cho biết THỨ TỰ, không cho biết độ lớn khác biệt.')
    print()
    print('  Lưu ý: baseline random_slots bốc node trực tiếp, KHÔNG định tuyến,')
    print('  nên về nguyên tắc nó không mất gì dưới walk thật. Chênh lệch ở cột')
    print('  random phản ánh nhiễu của mẫu 20 query, không phải hiệu ứng định tuyến.')


if __name__ == '__main__':
    main()
