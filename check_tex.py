#!/usr/bin/env python3
"""
KIỂM TRA NHẤT QUÁN FILE .TEX — chạy trước mỗi lần compile.

Vì sao cần: suốt các vòng sửa, cùng một lớp lỗi lọt qua nhiều lần — số cùng một
đại lượng ghi khác nhau ở hai bảng, bảng/hình không được \\ref, mục thực nghiệm
nằm trong Discussion, tuyên bố "chưa làm" cho việc đã làm. Con người đọc 54
trang không bắt được; script thì bắt được.

    python3 check_tex.py v-engram.tex
    python3 check_tex.py v-engram.tex --strict   # thoát mã 1 nếu có lỗi
"""
import argparse
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Các đại lượng phải NHẤT QUÁN toàn bài. Sửa danh sách này khi thêm bảng mới.
# (tên, regex bắt giá trị, giá trị đúng, dung sai)
# ---------------------------------------------------------------------------
INVARIANTS = [
    ('recall cấu hình chốt',   r'\$?80\.0',                80.0, 0.05),
    ('random cấu hình chốt',   r'\$?34\.5',                34.5, 0.05),
    ('node% cấu hình chốt',    r'\$?4\.9\\%',               4.9, 0.05),
]

# Cặp số KHÔNG được cùng xuất hiện (dấu hiệu làm tròn khác nhau cho cùng đại lượng)
CONFLICTS = [
    (r'22\.5\\%', r'22\.6\\%', 'candidate % cấu hình chốt'),
    (r'\$16\.3\$', r'\$16\.4\$', 'efficiency recall/node%'),
    (r'\$505\$ distinct', r'\$504\$ distinct', 'số node discrete-event chạm'),
    (r'4\{,\}414', r'4\{,\}465', 'unique candidates discrete-event'),
]

# Cụm từ báo hiệu tuyên bố có thể đã lỗi thời
STALE = [
    (r'left to future work', 'kiểm xem việc đó đã làm chưa'),
    (r'is not established here', 'kiểm xem đã có số chưa'),
    (r'not done here', 'kiểm xem đã làm chưa'),
    (r'the open question this paper leaves', 'kiểm xem mục sau có trả lời không'),
    (r'cannot answer it', 'kiểm xem mục sau có trả lời không'),
    (r'the earlier draft', 'ngôn ngữ quy trình, không nên có trong bài'),
    (r'the draft should', 'ngôn ngữ quy trình, không nên có trong bài'),
    (r'this paper opens', 'kiểm xem đã đóng chưa'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('--strict', action='store_true')
    a = ap.parse_args()
    s = open(a.tex, encoding='utf-8').read()
    err = warn = 0

    def E(m):
        nonlocal err; err += 1; print(f"  LỖI   {m}")

    def W(m):
        nonlocal warn; warn += 1; print(f"  cảnh báo  {m}")

    # ---- 1. Toàn vẹn LaTeX ----
    print("=" * 78)
    print("1. TOÀN VẸN LATEX")
    print("=" * 78)
    labels = set(re.findall(r'\\label\{([^}]+)\}', s))
    refs = set(re.findall(r'\\ref\{([^}]+)\}', s))
    for r in sorted(refs - labels):
        E(f"\\ref{{{r}}} không có \\label tương ứng")
    dup = [l for l in labels if s.count(f'\\label{{{l}}}') > 1]
    for l in dup:
        E(f"\\label{{{l}}} khai báo {s.count(f'label{{{l}}}')} lần")
    cites = set()
    for c in re.findall(r'\\cite\{([^}]+)\}', s):
        cites.update(x.strip() for x in c.split(','))
    bibs = set(re.findall(r'\\bibitem\{([^}]+)\}', s))
    for c in sorted(cites - bibs):
        E(f"\\cite{{{c}}} không có \\bibitem")
    for b in sorted(bibs - cites):
        W(f"\\bibitem{{{b}}} không được trích dẫn")
    for env in ('table', 'tabular', 'figure', 'equation', 'document',
                'thebibliography', 'abstract', 'algorithm', 'itemize', 'enumerate'):
        o, c = s.count('\\begin{'+env+'}'), s.count('\\end{'+env+'}')
        if o != c:
            E(f"môi trường {env} lệch: {o} begin / {c} end")
    # tàn dư lỗi escape
    for pat, desc in [(r'refsec:', 'thiếu dấu { } sau \\ref'),
                      (r'\\\\%', 'escape \\% lọt vào (kiểm cả nhãn hình)'),
                      (r'TODO', 'còn TODO')]:
        n = len(re.findall(pat, s))
        if n:
            E(f"{desc}: {n} chỗ")
    if err == 0:
        print("  ✓ sạch")

    # ---- 2. Bảng và hình phải được tham chiếu ----
    print()
    print("=" * 78)
    print("2. BẢNG / HÌNH CÓ ĐƯỢC THAM CHIẾU")
    print("=" * 78)
    e0 = err
    for pre, kind in (('tab:', 'Bảng'), ('fig:', 'Hình')):
        for l in sorted(x for x in labels if x.startswith(pre)):
            if f'\\ref{{{l}}}' not in s:
                E(f"{kind} {l} không có chỗ nào \\ref tới — reviewer sẽ để ý")
    if err == e0:
        print("  ✓ mọi bảng và hình đều được tham chiếu")

    # ---- 3. Mục thực nghiệm nằm đúng Section ----
    print()
    print("=" * 78)
    print("3. VỊ TRÍ MỤC")
    print("=" * 78)
    e0 = err
    secs = [(m.start(), m.group(1), m.group(2))
            for m in re.finditer(r'\\(section|subsection)\{([^}]+)\}', s)]
    cur = None
    EXPERIMENT_WORDS = ('sweep', 'weak scaling', 'churn', 'ablation', 'result',
                        'baseline', 'robustness', 'cost', 'load distribution',
                        'threshold formula')
    for pos, lvl, name in secs:
        if lvl == 'section':
            cur = name
        elif cur and 'discussion' in cur.lower():
            low = name.lower()
            if any(w in low for w in EXPERIMENT_WORDS) and 'threats' not in low:
                E(f'mục thực nghiệm "{name}" nằm trong Section "{cur}" — '
                  f'phải chuyển sang Experiments')
    if err == e0:
        print("  ✓ mục thực nghiệm nằm đúng Section")

    # ---- 4. Số mâu thuẫn ----
    print()
    print("=" * 78)
    print("4. SỐ MÂU THUẪN GIỮA CÁC BẢNG")
    print("=" * 78)
    e0 = err
    for p1, p2, desc in CONFLICTS:
        n1, n2 = len(re.findall(p1, s)), len(re.findall(p2, s))
        if n1 and n2:
            E(f"{desc}: cả hai giá trị cùng xuất hiện ({n1} và {n2} lần) — "
              f"thống nhất một chữ số")
    if err == e0:
        print("  ✓ không thấy số mâu thuẫn trong danh sách đã khai")

    # ---- 5. Tuyên bố có thể lỗi thời ----
    print()
    print("=" * 78)
    print("5. TUYÊN BỐ CẦN KIỂM LẠI (không tự động kết luận được)")
    print("=" * 78)
    found = False
    for pat, hint in STALE:
        for m in re.finditer(pat, s, re.I):
            found = True
            ctx = s[max(0, m.start()-90):m.start()+110].replace('\n', ' ')
            W(f"{hint}\n           ...{ctx.strip()}...")
    if not found:
        print("  ✓ không thấy cụm từ đáng nghi")

    # ---- 6. Bảng nào không khai số seed ----
    print()
    print("=" * 78)
    print("6. BẢNG CÓ KHAI SỐ SEED KHÔNG")
    print("=" * 78)
    e0 = warn
    # Không dùng regex bắt \caption{...} vì caption chứa dấu ngoặc lồng
    # (ví dụ $R_{\max}$) khiến non-greedy dừng sớm và cắt mất phần cuối.
    # Thay vào đó lấy toàn bộ đoạn từ \begin{table} tới \label.
    for m in re.finditer(r'\\label\{(tab:[^}]+)\}', s):
        lbl = m.group(1)
        beg = s.rfind('\\begin{table', 0, m.start())
        cap = s[beg:m.start()] if beg >= 0 else ''
        if not re.search(r'seed|Seed', cap):
            W(f"{lbl}: caption không nói số seed — chính chỗ này từng che bug Gini")
    if warn == e0:
        print("  ✓ mọi caption bảng đều khai số seed")

    # ---- 7. Bảng có số nhân bản sang make_figures.py phải có dòng ĐỒNG BỘ ----
    print()
    print("=" * 78)
    print("7. BẢNG NHÂN BẢN SỐ SANG make_figures.py")
    print("=" * 78)
    MIRRORED = ('tab:main_result', 'tab:budget_sweep', 'tab:normalised',
                'tab:rstar', 'tab:nsweep')
    e0 = warn
    for lbl in MIRRORED:
        i = s.find('\\label{' + lbl + '}')
        if i < 0:
            continue
        beg = s.rfind('\\begin{table}', 0, i)
        if beg < 0 or 'ĐỒNG BỘ' not in s[max(0, beg-300):beg]:
            W(f"{lbl}: thiếu dòng comment ĐỒNG BỘ trỏ về make_figures.py — "
              f"người sửa sau sẽ không biết có hai chỗ phải đổi")
    if warn == e0:
        print("  ✓ mọi bảng nhân bản đều có ghi chú đồng bộ")

    print()
    print("=" * 78)
    print(f"TỔNG: {err} lỗi, {warn} cảnh báo")
    print("=" * 78)
    if a.strict and err:
        sys.exit(1)


if __name__ == '__main__':
    main()