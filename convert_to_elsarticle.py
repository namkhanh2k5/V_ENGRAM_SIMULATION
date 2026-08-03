#!/usr/bin/env python3
r"""
CHUYỂN llncs -> elsarticle cho Computer Communications (Elsevier).

    python3 convert_to_elsarticle.py v-engram-revised.tex -o v-engram-cc.tex

Thay cho convert_to_snjnl.py: tạp chí đích đổi từ PPNA (Springer) sang
Computer Communications (Elsevier), nên class đổi từ sn-jnl sang elsarticle.

*** KHÔNG TEST ĐƯỢC TRONG SANDBOX vì thiếu elsarticle.cls. Bắt buộc compile
    thử và đọc kỹ phần cảnh báo cuối output. ***

CLASS: elsarticle có trong TeX Live/MiKTeX chuẩn, và Overleaf có sẵn template
"Elsevier Article (elsarticle)".

KHÁC BIỆT CHÍNH so với llncs:
  - title/author/abstract/keywords phải nằm trong \begin{frontmatter}
  - \author + \affiliation, KHÔNG phải \institute
  - abstract là MÔI TRƯỜNG (giống llncs, khác sn-jnl)
  - \keyword (số ít) trong frontmatter, phân cách bằng \sep
  - bibliography: elsarticle-num (số) hoặc elsarticle-harv (tên-năm)
  - Declaration of competing interest BẮT BUỘC
"""
import argparse
import re


def convert(s):
    notes, warns = [], []

    # ---------------------------------------------------------------- class
    s = re.sub(r'\\documentclass\[[^\]]*\]\{llncs\}',
               r'\\documentclass[preprint,12pt]{elsarticle}', s)
    s = re.sub(r'\\documentclass\{llncs\}',
               r'\\documentclass[preprint,12pt]{elsarticle}', s)
    notes.append("documentclass -> elsarticle [preprint,12pt]")
    warns.append("Nếu CC yêu cầu bản 2 cột khi nộp, đổi option thành "
                 "[final,5p,times,twocolumn]. Bản preprint dễ đọc hơn khi review.")

    # journal name
    if '\\journal{' not in s:
        s = s.replace('\\begin{document}',
                      '\\journal{Computer Communications}\n\n\\begin{document}', 1)
        notes.append("thêm \\journal{Computer Communications}")

    # ------------------------------------------------------------- citation
    # elsarticle dùng natbib nội bộ; giữ natbib sẽ xung đột
    s = re.sub(r'\\usepackage\[numbers,sort&compress\]\{natbib\}\n', '', s)
    s = re.sub(r'\\renewcommand\{\\bibname\}\{References\}\n', '', s)
    s = re.sub(r'\\renewcommand\{\\bibsection\}\{\\section\*\{\\bibname\}\}\n', '', s)
    notes.append("gỡ natbib (elsarticle tự nạp)")

    # ------------------------------------------------------------ title box
    m_t = re.search(r'\\title\{(.+?)\}\n', s, re.S)
    m_tr = re.search(r'\\titlerunning\{(.+?)\}\n', s)
    m_a = re.search(r'\\author\{(.+?)\}\n', s, re.S)
    m_i = re.search(r'\\institute\{(.+?)\}\n\n', s, re.S)
    m_ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S)

    title = m_t.group(1).strip() if m_t else 'TITLE'
    short = m_tr.group(1).strip() if m_tr else title[:60]
    names = [x.strip() for x in m_a.group(1).split('\\and')] if m_a else []
    emails = re.findall(r'\\email\{([^}]+)\}', m_i.group(1)) if m_i else []
    abstract = m_ab.group(1).strip() if m_ab else ''

    kw = 'Decentralized AI, Similarity Search, Distributed Hash Table, Kademlia, LSH'
    m_kw = re.search(r'pdfkeywords=\{([^}]+)\}', s)
    if m_kw:
        kw = m_kw.group(1)
    kw_tex = ' \\sep '.join(k.strip() for k in re.split(r'[,;]', kw) if k.strip())

    auth = []
    for k, nm in enumerate(names):
        star = '[label1]' if k else '[label1,cor1]'
        em = f'\n\\ead{{{emails[k]}}}' if k < len(emails) else ''
        auth.append(f'\\author{star}{{{nm}}}{em}')

    fm = (
        '\\begin{frontmatter}\n\n'
        f'\\title{{{title}}}\n\n'
        + '\n'.join(auth) + '\n\n'
        '\\affiliation[label1]{organization={School of Information and '
        'Communication Technology, Hanoi University of Science and Technology},\n'
        '            city={Hanoi},\n'
        '            country={Vietnam}}\n\n'
        '\\cortext[cor1]{Corresponding author}\n\n'
        f'\\begin{{abstract}}\n{abstract}\n\\end{{abstract}}\n\n'
        f'\\begin{{keyword}}\n{kw_tex}\n\\end{{keyword}}\n\n'
        '\\end{frontmatter}\n'
    )
    notes.append(f"dựng frontmatter: title, {len(names)} author, affiliation, "
                 f"abstract, keyword")
    warns.append("KIỂM \\affiliation: script điền cứng tên trường và địa chỉ. "
                 "Sửa nếu sai. Thêm ORCID nếu CC yêu cầu.")

    # gỡ các lệnh cũ và chèn frontmatter
    for m in (m_t, m_tr, m_a, m_i):
        if m:
            s = s.replace(m.group(0), '')
    if m_ab:
        s = s.replace(m_ab.group(0), '')
    s = re.sub(r'\\authorrunning\{[^}]*\}\n\n?', '', s)
    s = s.replace('\\maketitle\n', '')

    i = s.find('\\begin{document}')
    j = s.find('\n', i) + 1
    s = s[:j] + '\n' + fm + s[j:]
    notes.append("gỡ maketitle/titlerunning/authorrunning (elsarticle không dùng)")

    # ------------------------------------------------------- declarations
    i = s.find('\\begin{thebibliography}')
    if i >= 0:
        decl = r"""
\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

\section*{Data availability}
The corpora are derived from public datasets (CodeSearchNet and SciFact). The
simulator, analysis scripts, and configuration files needed to reproduce every
table and figure are available at
\url{https://github.com/namkhanh2k5/V_ENGRAM_SIMULATION}.
% TODO: cân nhắc gán DOI Zenodo cho bản snapshot dùng trong bài.

\section*{CRediT authorship contribution statement}
% TODO: điền theo mẫu CRediT, ví dụ:
% \textbf{Tên A}: Conceptualization, Methodology, Software, Writing -- original draft.
% \textbf{Tên B}: Supervision, Writing -- review \& editing.

\section*{Acknowledgements}
% TODO: điền nguồn tài trợ nếu có, hoặc bỏ mục này.

"""
        s = s[:i] + decl + s[i:]
        notes.append("chèn Declaration of competing interest, Data availability, "
                     "CRediT, Acknowledgements (3 TODO)")
        warns.append("Declaration of competing interest là BẮT BUỘC ở Elsevier. "
                     "CRediT thường bắt buộc. Data availability statement bắt buộc.")

    # ---------------------------------------------------------------- misc
    for pkg in ('inputenc', 'fontenc'):
        s = re.sub(r'\\usepackage\[[^\]]*\]\{' + pkg + r'\}\n', '', s)
    if '\\raggedbottom' in s:
        s = s.replace('\\raggedbottom\n', '')
    notes.append("gỡ inputenc/fontenc/raggedbottom")

    if '\\usepackage{algorithm}' in s:
        warns.append("elsarticle có thể xung đột với algorithm/algpseudocode. "
                     "Nếu lỗi, thử \\usepackage[section]{algorithm}.")
    warns.append("elsarticle mặc định KHÔNG dùng \\subsubsection sâu. Kiểm mục lục "
                 "sau khi compile.")

    return s, notes, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('-o', '--out', default=None)
    a = ap.parse_args()
    out, notes, warns = convert(open(a.tex, encoding='utf-8').read())
    dst = a.out or a.tex.replace('.tex', '-cc.tex')
    open(dst, 'w', encoding='utf-8').write(out)

    print('=' * 78)
    print(f'ĐÃ CHUYỂN: {a.tex} -> {dst}')
    print('=' * 78)
    for i, n in enumerate(notes, 1):
        print(f'  {i:>2}. {n}')
    print()
    print('=' * 78)
    print('BẮT BUỘC KIỂM TAY — script KHÔNG test được vì thiếu elsarticle.cls')
    print('=' * 78)
    for i, w in enumerate(warns, 1):
        print(f'  {i}. {w}')
    print()
    print('  Bản llncs GIỮ NGUYÊN, không bị ghi đè.')
    print('  Sau khi compile được:  python3 check_tex.py ' + dst)


if __name__ == '__main__':
    main()
