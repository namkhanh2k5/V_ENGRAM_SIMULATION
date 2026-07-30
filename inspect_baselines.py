#!/usr/bin/env python3
"""
KIỂM LOG BASELINE — giải quyết mục 3.1 và 3.2 của bản rà soát.

HAI CÂU HỎI CHƯA TRẢ LỜI ĐƯỢC:

3.1 — Bảng baseline ghi candidate của Crypto-DHT là 3.505. Người rà soát chạy lại
      ở POOL_PER_TABLE=902 và được 4.122, nên đã sửa bảng. Nhưng nếu run gốc dùng
      P khác 902 thì 4.122 sai. Cần xác nhận P thật từ log.

3.2 — Bảng ghi "Multi-table LSH ceiling = 88,0%". Người rà soát đo pool LSH ở
      P=902 với rerank CHÍNH XÁC và được 99,7%. Chênh 11,7 điểm. Nếu chênh đó do
      PQ thì PQ đang mất 11,7 điểm ở pool của LSH, trong khi bảng PQ nói chỉ mất
      1,6 điểm ở pool của V-Engram. Hai số khó cùng đúng. Cần biết run gốc dùng
      RERANK = 'adc' hay 'exact'.

Script này đọc mọi log baseline_*.txt và muc18_*.txt, trích P và RERANK, rồi đối
chiếu với số trong bài. KHÔNG chạy lại gì — chỉ đọc.

    python3 inspect_baselines.py
"""
import glob
import re

# Số trong bài, để đối chiếu
PAPER = {
    'LSH ceiling':  {'recall': 88.0, 'cand': 3505},
    'Crypto-DHT':   {'recall': 22.5, 'cand': 4122},   # 4.122 sau khi rà soát sửa
    'V-Engram':     {'recall': 80.0, 'cand': 4510},
}


def parse(path):
    """Trích mọi thông tin nhận dạng được từ một log baselines.py."""
    try:
        txt = open(path, encoding='utf-8', errors='ignore').read()
    except OSError:
        return None
    out = {'file': path, 'corpus': None, 'pool_cfg': None, 'rerank': None,
           'rows': []}
    m = re.search(r'Corpus:\s*([\d,]+)\s*vectors', txt)
    if m:
        out['corpus'] = int(m.group(1).replace(',', ''))
    # POOL/table in ở dòng cấu hình
    for pat in (r'POOL[_/ ]?(?:PER[_ ])?TABLE\s*[=:]\s*(\d+)',
                r'POOL/table\s*[=:]\s*(\d+)',
                r'pool m[oỗ]i b[aả]ng\s*[=:]\s*(\d+)'):
        m = re.search(pat, txt, re.I)
        if m:
            out['pool_cfg'] = int(m.group(1)); break
    for pat in (r'RERANK\s*[=:]\s*[\'"]?(\w+)', r'rerank\s*[=:]\s*[\'"]?(adc|exact)'):
        m = re.search(pat, txt, re.I)
        if m:
            out['rerank'] = m.group(1).lower(); break
    # các dòng kết quả: tên, Recall@5, (pool nếu có)
    for line in txt.split('\n'):
        m = re.search(r'(Brute-Force|HNSW|Multi-Table LSH|Bucket-LSH|Crypto-DHT|'
                      r'Random-C|Random-5)[^\d]*?([\d.]+)\s*%', line)
        if m:
            pool = None
            mp = re.search(r'pool[^\d]*([\d,]+)', line, re.I)
            if mp:
                pool = int(mp.group(1).replace(',', ''))
            out['rows'].append((m.group(1), float(m.group(2)), pool))
    return out


def main():
    files = sorted(set(glob.glob('baseline_*.txt') + glob.glob('muc18_*.txt')))
    if not files:
        print('Không thấy log nào. Cần baseline_*.txt hoặc muc18_*.txt trong thư mục này.')
        print('Nếu đã xoá, chạy lại:')
        print('  CORPUS=code POOL_PER_TABLE=902 python3 baselines.py > muc18_code_P902.txt')
        return

    print('=' * 92)
    print(f'ĐỌC {len(files)} LOG BASELINE')
    print('=' * 92)
    print(f"{'file':30s} {'corpus':>8s} {'POOL/table':>11s} {'RERANK':>8s} {'#dòng':>6s}")
    print('-' * 92)
    parsed = []
    for f in files:
        d = parse(f)
        if not d:
            continue
        parsed.append(d)
        print(f"{d['file'][:30]:30s} {str(d['corpus'] or '?'):>8s} "
              f"{str(d['pool_cfg'] or '?'):>11s} {str(d['rerank'] or '?'):>8s} "
              f"{len(d['rows']):>6d}")

    # ---- 3.1: tìm log có candidate khớp bài ----
    print()
    print('=' * 92)
    print('3.1 — RUN NÀO SINH RA BẢNG BASELINE?')
    print('=' * 92)
    hit = False
    for d in parsed:
        for name, rec, pool in d['rows']:
            for pname, pv in PAPER.items():
                if pname.split()[0].lower() in name.lower():
                    dr = abs(rec - pv['recall'])
                    dp = abs(pool - pv['cand']) if pool else None
                    if dr < 0.6 or (dp is not None and dp < 30):
                        hit = True
                        print(f"  {d['file']}: {name} recall={rec:.1f}% "
                              f"pool={pool} | bài ghi {pv['recall']}% / {pv['cand']}")
                        print(f"      -> POOL/table = {d['pool_cfg'] or 'KHÔNG in trong log'}, "
                              f"RERANK = {d['rerank'] or 'KHÔNG in trong log'}")
    if not hit:
        print('  Không log nào khớp số trong bài.')
        print('  => Log của run gốc đã mất, hoặc baselines.py không in POOL/RERANK.')
        print('     Cách xác định: chạy lại và so, xem lệnh ở cuối script này.')

    # ---- 3.2: LSH ceiling adc vs exact ----
    print()
    print('=' * 92)
    print('3.2 — LSH CEILING: 88,0% (bài) vs 99,7% (đo lại với rerank exact)')
    print('=' * 92)
    lsh = [(d, r) for d in parsed for r in d['rows'] if 'Multi-Table LSH' in r[0]]
    if lsh:
        for d, (name, rec, pool) in lsh:
            print(f"  {d['file']}: {rec:.1f}% (pool {pool}), "
                  f"RERANK={d['rerank'] or '?'}, P={d['pool_cfg'] or '?'}")
        vals = sorted({round(r[1], 1) for _, r in lsh})
        print()
        if any(abs(v - 88.0) < 0.6 for v in vals) and any(v > 97 for v in vals):
            print('  => Có CẢ HAI giá trị trong log. Cái 88,0 phải là rerank ADC,')
            print('     cái >97 là rerank exact. Bài đang so LSH-với-ADC với')
            print('     V-Engram-với-ADC, tức đúng cùng điều kiện. KHÔNG mâu thuẫn,')
            print('     nhưng NÊN ghi rõ trong caption là "with ADC rerank".')
        elif any(abs(v - 88.0) < 0.6 for v in vals):
            print('  => Log chỉ có 88,0. Kiểm RERANK ở cột trên:')
            print('     nếu là "adc" thì bài đúng, chỉ cần ghi rõ vào caption.')
            print('     nếu là "exact" thì có mâu thuẫn thật, phải điều tra.')
        else:
            print('  => Không thấy giá trị 88,0 trong log nào. Số trong bài không')
            print('     truy được về log — phải chạy lại để xác nhận.')
    else:
        print('  Không log nào có dòng Multi-Table LSH.')

    print()
    print('=' * 92)
    print('NẾU LOG KHÔNG ĐỦ THÔNG TIN — chạy hai lệnh này để xác định dứt điểm')
    print('=' * 92)
    print("""
  # Đo LSH ceiling ở CẢ HAI chế độ rerank, cùng P, để biết PQ mất bao nhiêu
  for R in adc exact; do
      CORPUS=code POOL_PER_TABLE=902 RERANK=$R python3 baselines.py \\
          > lsh_rerank_$R.txt 2>&1
      echo -n "  RERANK=$R: "
      grep "Multi-Table LSH" lsh_rerank_$R.txt | grep -oE "[0-9.]+%" | head -1
  done

  # Nếu adc ~88 và exact ~99.7 thì bài đúng, chỉ cần ghi "with ADC rerank"
  # vào caption. Nếu adc cũng ~99.7 thì số 88,0 trong bài không truy được.
""")
    print("  LƯU Ý: baselines.py hiện đọc CORPUS và POOL_PER_TABLE từ env nhưng")
    print("  RERANK thì hardcode. Nếu lệnh trên không đổi được chế độ, sửa dòng")
    print("  RERANK trong baselines.py thành:")
    print("      RERANK = _os.environ.get('RERANK', 'adc')")


if __name__ == '__main__':
    main()
