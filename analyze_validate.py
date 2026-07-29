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
        m = _re.search(r'nq[= ]+(\d+)|--nq (\d+)', open(f, errors='ignore').read())
        if m:
            nq = m.group(1) or m.group(2)
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
    # ĐỐI CHỨNG NHIỄU: random_slots bốc node trực tiếp, KHÔNG định tuyến, nên
    # chênh lệch THẬT của nó phải bằng 0. Mọi lệch quan sát được ở cột random
    # là nhiễu thuần tuý — và nó cho biết sàn nhiễu của phép đo.
    rnd_drops = [ideal['rnd'] - rnd for (_, ideal, _, rnd) in rows
                 if rnd is not None and ideal['rnd'] is not None]
    print()
    if rnd_drops:
        noise = max(abs(min(rnd_drops)), abs(max(rnd_drops)))
        print(f'  SÀN NHIỄU (từ cột random, nơi chênh lệch thật = 0):')
        print(f'    lệch quan sát: ' + ', '.join(f'{d:+.1f}' for d in rnd_drops))
        print(f'    biên độ nhiễu ~{noise:.1f} điểm')
        print()
        if spread <= noise * 1.5:
            print(f'  => Tản của cột semantic ({spread:.1f}đ) KHÔNG vượt sàn nhiễu.')
            print(f'     Không phát hiện được suy giảm hệ thống nào do định tuyến thật.')
            print(f'     NHƯNG phép đo cũng KHÔNG ĐỦ ĐỘ PHÂN GIẢI để loại trừ hiệu ứng')
            print(f'     cỡ vài điểm. Cần nhiều query hơn nếu muốn kết luận chắc.')
        else:
            print(f'  => Tản semantic ({spread:.1f}đ) VƯỢT sàn nhiễu ({noise:.1f}đ).')
            print(f'     Có dấu hiệu định tuyến thật ảnh hưởng khác nhau theo cấu hình.')
    if any(d < 0 for _, d in drops):
        print()
        print('  *** CẢNH BÁO: có cấu hình mà walk THẬT cho recall CAO HƠN lý tưởng.')
        print('      Điều đó BẤT KHẢ về cơ chế — walk chỉ xấp xỉ tập K node gần nhất.')
        print('      Đây là bằng chứng trực tiếp rằng mẫu query quá nhỏ. Tăng --nq.')


if __name__ == '__main__':
    main()