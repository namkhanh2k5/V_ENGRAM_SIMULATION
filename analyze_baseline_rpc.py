#!/usr/bin/env python3
"""
So chi phí RPC ĐO ĐƯỢC với con số bài SUY RA (mục 9).

    python3 analyze_baseline_rpc.py
"""
import glob
import re

# Số bài đang dùng, trong bảng "Recall per unit of cost".
# Cột RPC của baseline là SUY RA, không đo — đó là chỗ mục 9 muốn đóng.
PAPER = {
    'semantic':      {'rpc': 1145, 'recall': 80.0, 'basis': 'đo được'},
    'keyed_lookup':  {'rpc': 1411, 'recall': 33.4, 'basis': 'suy: 641 lookup + 770 ADC'},
    'random_slots':  {'rpc':  800, 'recall': 34.5, 'basis': 'suy: chỉ ADC, 800 node'},
    'random_unique': {'rpc':  492, 'recall': 22.1, 'basis': 'suy: chỉ ADC, 492 node'},
}
ORDER = ['semantic', 'keyed_lookup', 'random_slots', 'random_unique']


def read(mode):
    f = f'rpc_{mode}.txt'
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    g = lambda pat: (float(m.group(1).replace(',', ''))
                     if (m := re.search(pat, t)) else None)
    return {
        'file': f,
        'recall': g(r'Recall@5\s*:\s*([\d.]+)%'),
        'rpc_total': g(r'RPC/query\s*:\s*([\d,.]+)'),
        'disc_rpc': g(r'RPCs?(?:/query)?\s+([\d,.]+)'),
        'nodes': g(r'Unique nodes contacted\s+([\d,.]+)'),
        # log in "Routing rounds/query   213.8  2896.2  3110.0" — phải cho
        # khớp cả '/query' và số thập phân, không chỉ \s+ và số nguyên
        'rounds': g(r'Routing rounds(?:/query)?\s+([\d,.]+)'),
    }


def main():
    rows = [(m, PAPER[m], read(m)) for m in ORDER]
    if all(r is None for _, _, r in rows):
        print('Chưa có file rpc_*.txt nào. Chạy run_measure_baseline_rpc.sh trước.')
        return

    print('=' * 96)
    print('CHI PHÍ RPC: ĐO ĐƯỢC vs BÀI SUY RA')
    print('=' * 96)
    print(f"{'chế độ':16s} {'bài suy':>9s} {'đo được':>9s} {'lệch':>8s} "
          f"{'node':>7s} {'recall':>8s}   căn cứ bài dùng")
    print('-' * 96)
    ok = True
    for mode, pv, r in rows:
        if r is None or r['rpc_total'] is None:
            print(f"{mode:16s} {pv['rpc']:>9,} {'(chưa chạy)':>9s}")
            continue
        meas = r['rpc_total']
        d = 100 * (meas - pv['rpc']) / pv['rpc']
        flag = '' if abs(d) < 15 else '  <-- LỆCH LỚN'
        if abs(d) >= 15:
            ok = False
        print(f"{mode:16s} {pv['rpc']:>9,} {meas:>9,.0f} {d:>+7.0f}% "
              f"{(r['nodes'] or 0):>7,.0f} {(r['recall'] or 0):>7.1f}% "
              f"  {pv['basis']}{flag}")

    # ---- kiểm giả định cốt lõi: khoá ngẫu nhiên tốn như khoá ngữ nghĩa? ----
    sem, key = read('semantic'), read('keyed_lookup')
    print()
    print('=' * 96)
    print('GIẢ ĐỊNH CỐT LÕI: khoá ngẫu nhiên tốn như khoá ngữ nghĩa để định tuyến?')
    print('=' * 96)
    if sem and key and sem.get('rounds') and key.get('rounds'):
        # cả hai chạy đúng L*T = 40 lookup, nên so vòng/lookup là so trực tiếp
        rs, rk = sem['rounds'] / 40, key['rounds'] / 40
        print(f"  vòng mỗi lookup: semantic {rs:.1f} | khoá ngẫu nhiên {rk:.1f} "
              f"({100*(rk-rs)/rs:+.0f}%)")
        if abs(rk - rs) / rs < 0.10:
            print('  => Giả định ĐÚNG. Độ dài lookup không phụ thuộc khoá, nên phép')
            print('     suy trong bài hợp lệ, và giờ có số đo hậu thuẫn.')
        else:
            print('  => Giả định SAI. Khoá ngẫu nhiên tốn khác khoá ngữ nghĩa, nên')
            print('     phải dùng số ĐO trong bảng, và nêu hiệu ứng này trong bài.')
    else:
        print('  Chưa đủ dữ liệu (cần cả rpc_semantic.txt và rpc_keyed_lookup.txt).')

    # ---- tính lại hiệu suất per-RPC bằng số đo ----
    print()
    print('=' * 96)
    print('HIỆU SUẤT PER-RPC TÍNH LẠI BẰNG SỐ ĐO')
    print('=' * 96)
    base = None
    print(f"{'chế độ':16s} {'R@5/10^3 RPC':>14s} {'tỉ lệ so semantic':>19s}")
    print('-' * 96)
    for mode, pv, r in rows:
        if r is None or not r['rpc_total'] or not r['recall']:
            continue
        eff = r['recall'] / r['rpc_total'] * 1000
        if mode == 'semantic':
            base = eff
        rel = f"{base/eff:.2f}x" if base and eff > 0 and mode != 'semantic' else '---'
        print(f"{mode:16s} {eff:>14.1f} {rel:>19s}")
    print()
    if ok:
        print('  Mọi số đo nằm trong 15% con số bài suy ra. Có thể đổi câu trong')
        print('  Threats từ "derived rather than measured" thành "measured", và')
        print('  cập nhật cột RPC bằng số đo.')
    else:
        print('  Có chỗ lệch quá 15%. PHẢI dùng số đo trong bảng và giải thích vì')
        print('  sao phép suy sai ở chỗ đó.')


if __name__ == '__main__':
    main()