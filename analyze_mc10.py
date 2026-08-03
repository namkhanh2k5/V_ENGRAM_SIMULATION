#!/usr/bin/env python3
"""
MC10 — phân tích lối A (sửa payload) và kiểm kê lối B (hạ payload xuống phụ).

    python3 analyze_mc10.py                    # lối A: so bốn biến thể
    python3 analyze_mc10.py --inventory a.tex  # lối B: bài mất gì nếu hạ payload
"""
import argparse
import glob
import json
import re


# ---------------------------------------------------------------- LỐI A
def read(f):
    try:
        t = open(f, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    g = lambda p: (float(m.group(1).replace(',', ''))
                   if (m := re.search(p, t)) else None)
    return {
        'disc_rpc': g(r'RPC/query\s+([\d,.]+)'),
        'pay_rpc': g(r'RPC/query\s+[\d,.]+\s+([\d,.]+)'),
        'tot_rpc': g(r'RPC/query\s+[\d,.]+\s+[\d,.]+\s+([\d,.]+)'),
        'p50': g(r'Latency p50 \(ms\)\s+([\d,.]+)'),
        'p95': g(r'Latency p95 \(ms\)\s+([\d,.]+)'),
        'recall': g(r'Recall@5\s*:\s*([\d.]+)%'),
        'depth': g(r'probe_depth_mean["\s:]+([\d.]+)'),
    }


def variants():
    V = [('scan', 300, 0, 'hiện tại (mốc so)'),
         ('deterministic', 300, 0, 'chỉ đổi tính tất định'),
         ('deterministic', 20, 0, 'thêm: hạ k'),
         ('deterministic', 20, 1, 'thêm: chỉ lấy top-1')]
    rows = []
    for mode, k, top, note in V:
        d = read(f'mc10_{mode}_k{k}_top{top}.txt')
        # depth nằm trong JSON, không phải log
        js = glob.glob(f'result_full_*_{mode}{k}*_nq*.json')
        if js:
            try:
                j = json.load(open(js[0]))
                if d:
                    d['depth'] = j.get('probe_depth_mean')
                    d['miss'] = j.get('shard_misses')
            except Exception:
                pass
        rows.append((mode, k, top, note, d))
    return rows


def show_a():
    rows = variants()
    if not any(d for *_, d in rows):
        print('Chưa có file mc10_*.txt. Chạy run_mc10_payload.sh trước.')
        return
    print('=' * 100)
    print('LỐI A — SỬA TẦNG PAYLOAD')
    print('=' * 100)
    print(f"{'biến thể':34s} {'sâu dò':>7s} {'disc':>7s} {'payload':>9s} "
          f"{'tổng':>9s} {'pay %':>6s} {'p50 (s)':>8s}")
    print('-' * 100)
    base = None
    for mode, k, top, note, d in rows:
        lbl = f'{mode} k={k}' + (f' top={top}' if top else '')
        if not d or d.get('tot_rpc') is None:
            print(f'{lbl:34s} {"(chưa chạy)":>7s}')
            continue
        if base is None:
            base = d['tot_rpc']
        pct = 100 * d['pay_rpc'] / d['tot_rpc'] if d['tot_rpc'] else 0
        dep = f"{d['depth']:.2f}" if d.get('depth') is not None else '?'
        p50 = f"{d['p50']/1000:.1f}" if d.get('p50') else '?'
        print(f'{lbl:34s} {dep:>7s} {d["disc_rpc"]:>7,.0f} {d["pay_rpc"]:>9,.0f} '
              f'{d["tot_rpc"]:>9,.0f} {pct:>5.0f}% {p50:>8s}')

    # đọc chỉ số quyết định
    d0 = rows[0][4]
    print()
    print('  CHỈ SỐ QUYẾT ĐỊNH — độ sâu dò ở chế độ scan:')
    if d0 and d0.get('depth') is not None:
        if d0['depth'] < 1.3:
            print(f'    {d0["depth"]:.2f} — client tìm thấy shard ngay ứng viên đầu.')
            print('    => Giả thuyết "mất tính tất định gây dò sâu" KHÔNG đúng.')
            print('       Chi phí đến từ PLACEMENT_CANDIDATES quá lớn, không từ dò.')
            print('       Sửa rẻ nhất: hạ k, KHÔNG cần đổi quy tắc đặt shard.')
        else:
            print(f'    {d0["depth"]:.2f} — phải dò sâu. Giả thuyết ĐÚNG.')
            print('    => Đổi sang đặt tất định là cần thiết, không chỉ hạ k.')
    else:
        print('    (chưa có số)')

    if len(rows) >= 4 and all(r[4] and r[4].get('tot_rpc') for r in rows):
        a, b, c, e = (r[4]['tot_rpc'] for r in rows)
        print()
        print('  PHÂN RÃ MỨC GIẢM:')
        print(f'    do tính tất định : {a:,.0f} -> {b:,.0f}  ({100*(1-b/a):+.0f}%)')
        print(f'    do hạ k          : {b:,.0f} -> {c:,.0f}  ({100*(1-c/b):+.0f}%)')
        print(f'    do chỉ lấy top-1 : {c:,.0f} -> {e:,.0f}  ({100*(1-e/c):+.0f}%)')


# ---------------------------------------------------------------- LỐI B
def show_b(tex):
    s = open(tex, encoding='utf-8').read()
    PAT = r'payload|shard|Reed[- ]Solomon|erasure|placement key'
    print('=' * 100)
    print('LỐI B — KIỂM KÊ: BÀI MẤT GÌ NẾU HẠ PAYLOAD XUỐNG THÀNH PHẦN PHỤ')
    print('=' * 100)

    secs = [(m.start(), m.group(1), m.group(2))
            for m in re.finditer(r'\\(section|subsection)\{([^}]+)\}', s)]
    print()
    print('MỤC PHỤ THUỘC PAYLOAD (số lần nhắc):')
    print(f"  {'mục':46s} {'nhắc':>5s}  mức phụ thuộc")
    print('  ' + '-' * 78)
    heavy = []
    for i, (pos, lvl, name) in enumerate(secs):
        end = secs[i + 1][0] if i + 1 < len(secs) else len(s)
        seg = s[pos:end]
        n = len(re.findall(PAT, seg, re.I))
        words = len(seg.split())
        if n < 3:
            continue
        dens = n / max(words, 1) * 1000
        lvl_s = '  ' if lvl == 'subsection' else ''
        verdict = ('VIẾT LẠI' if dens > 12 else
                   'sửa vài câu' if dens > 5 else 'gần như giữ')
        if dens > 12:
            heavy.append(name)
        print(f'  {lvl_s}{name[:44]:44s} {n:>5}  {verdict}')

    print()
    print('BẢNG / HÌNH phụ thuộc payload:')
    for m in re.finditer(r'\\label\{(tab|fig):([^}]+)\}', s):
        lbl = m.group(0)
        beg = s.rfind('\\begin{', 0, m.start())
        seg = s[beg:m.start()] if beg >= 0 else ''
        n = len(re.findall(PAT, seg, re.I))
        if n >= 2:
            print(f'  {m.group(1)}:{m.group(2):32s} {n:>3} lần nhắc')

    print()
    print('TUYÊN BỐ TRONG ABSTRACT / CONTRIBUTIONS / CONCLUSION:')
    for lbl, beg, end in [('Abstract', '\\begin{abstract}', '\\end{abstract}'),
                          ('Contributions', '\\begin{enumerate}', '\\end{enumerate}'),
                          ('Conclusion', '\\section{Conclusion}', None)]:
        i = s.find(beg)
        j = s.find(end, i) if end else len(s)
        if i < 0:
            continue
        seg = s[i:j]
        hits = [m.group(0) for m in re.finditer(PAT, seg, re.I)]
        print(f'  {lbl:16s} {len(hits):>3} lần')
        for sent in re.split(r'(?<=\.)\s+', re.sub(r'\s+', ' ', seg)):
            if re.search(PAT, sent, re.I):
                print(f'      "{sent.strip()[:110]}..."')

    print()
    print('=' * 100)
    print('ĐỌC KIỂM KÊ NÀY THẾ NÀO')
    print('=' * 100)
    print('  Mục ghi VIẾT LẠI là chỗ payload là chủ đề chính, không phải nhắc qua.')
    print('  Nếu số mục đó ít (2-3) thì lối B rẻ: đóng khung lại vài mục, sửa')
    print('  Abstract và Contributions, giữ nguyên phần còn lại.')
    print('  Nếu nhiều (>5) thì payload đã ăn sâu vào lập luận, hạ nó xuống sẽ')
    print('  để lại nhiều chỗ hụt — lúc đó lối A đáng hơn.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', metavar='TEX',
                    help='lối B: kiểm kê phụ thuộc payload trong file .tex')
    a = ap.parse_args()
    if a.inventory:
        show_b(a.inventory)
    else:
        show_a()
