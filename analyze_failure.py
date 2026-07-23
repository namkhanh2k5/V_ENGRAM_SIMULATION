#!/usr/bin/env python3
"""
Phân tích node failure (mục 16).

Gộp hai nguồn:
  A. result_*_loss*.json  -> Reachable/Final Recall@5 dưới node loss
  B. failure_code_r*_s*.txt -> Metadata availability + Payload recovery

Câu hỏi: r mua được bao nhiêu độ bền? Mục 14 đã cho thấy r không tăng recall và
phá tỉ lệ sem/rand. Nếu r=1 vẫn trụ được dưới node loss thì không có lý do tăng r.

    python3 analyze_failure.py
"""
import glob
import json
import re
import statistics as st
from collections import defaultdict


def load_recall():
    """Đọc kết quả recall dưới node loss."""
    data = defaultdict(list)
    for f in glob.glob('result_code_N10000_L5_K20_MA*_T8_m512*_nq500.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('n_query', 0) < 500 or d.get('zipf', 0) > 0:
            continue
        mode = d.get('routing_mode') or ('random_slots' if d.get('random_routing') else 'semantic')
        if mode not in ('semantic', 'random_slots'):
            continue
        r = d['meta_anchors']
        loss = d.get('node_loss', 0.0)
        data[(r, loss, mode)].append((d['reachable_recall5'], d['recall5']))
    return data


def load_churn():
    """Đọc log churn: bảng 'Node loss | Metadata | Payload | End-to-end'."""
    out = defaultdict(lambda: defaultdict(list))   # r -> loss -> [(meta, pay, e2e)]
    for f in glob.glob('failure_code_r*_s*.txt'):
        m = re.search(r'_r(\d+)_s', f)
        if not m:
            continue
        r = int(m.group(1))
        for line in open(f, encoding='utf-8', errors='ignore'):
            mm = re.match(r'\s*(\d+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%', line)
            if mm:
                loss = int(mm.group(1)) / 100.0
                out[r][loss].append(tuple(float(mm.group(i)) for i in (2, 3, 4)))
    return out


def main():
    rec = load_recall()
    chu = load_churn()

    print("=" * 92)
    print("MỤC 16 — NODE FAILURE: r ∈ {1,2,3} × loss ∈ {0,10,20,30}%")
    print("=" * 92)
    print(f"{'r':>2} {'loss':>5} {'n':>3} {'Reach R@5':>11} {'Final R@5':>11} "
          f"{'Random R@5':>11} {'tỉ lệ':>6} | {'Meta avail':>11} {'Payload':>9} {'E2E':>7}")
    print("-" * 92)

    losses = [0.0, 0.1, 0.2, 0.3]
    for r in (1, 2, 3):
        for loss in losses:
            sem = rec.get((r, loss, 'semantic'), [])
            rnd = rec.get((r, loss, 'random_slots'), [])
            if not sem:
                continue
            reach = st.mean(x[0] for x in sem)
            final = st.mean(x[1] for x in sem)
            rnd_m = st.mean(x[1] for x in rnd) if rnd else float('nan')
            ratio = final / rnd_m if rnd and rnd_m > 0 else float('nan')
            # phần B
            cb = chu.get(r, {}).get(loss, [])
            if cb:
                meta = st.mean(x[0] for x in cb)
                pay = st.mean(x[1] for x in cb)
                e2e = st.mean(x[2] for x in cb)
                bstr = f"| {meta:>10.1f}% {pay:>8.1f}% {e2e:>6.1f}%"
            else:
                bstr = f"| {'--':>10} {'--':>8} {'--':>6}"
            rs = f"{ratio:>6.2f}" if ratio == ratio else f"{'--':>6}"
            print(f"{r:>2} {100*loss:>4.0f}% {len(sem):>3} {reach:>10.1f}% {final:>10.1f}% "
                  f"{rnd_m:>10.1f}% {rs} {bstr}")
        print()

    # kết luận: r mua được bao nhiêu độ bền?
    print("=" * 92)
    print("r MUA ĐƯỢC BAO NHIÊU ĐỘ BỀN?")
    print("=" * 92)
    print(f"{'r':>2} {'Final R@5 @0%':>14} {'@30%':>9} {'mất':>8}   {'Meta @30%':>10}")
    print("-" * 60)
    for r in (1, 2, 3):
        s0 = rec.get((r, 0.0, 'semantic'), [])
        s3 = rec.get((r, 0.3, 'semantic'), [])
        if not (s0 and s3):
            continue
        f0 = st.mean(x[1] for x in s0)
        f3 = st.mean(x[1] for x in s3)
        cb = chu.get(r, {}).get(0.3, [])
        mstr = f"{st.mean(x[0] for x in cb):>9.1f}%" if cb else f"{'--':>10}"
        print(f"{r:>2} {f0:>13.1f}% {f3:>8.1f}% {f0-f3:>+7.1f}đ   {mstr}")
    print()
    print("CÁCH ĐỌC:")
    print("  Mục 14 đã cho thấy r KHÔNG tăng recall và PHÁ tỉ lệ sem/rand.")
    print("  Thứ duy nhất r mua là độ bền. Bảng này định lượng nó.")
    print("  Nếu r=1 mất ít điểm ở 30% node chết => tăng r là trả giá vô ích,")
    print("  và độ bền nên mua bằng REPAIR (re-anchor) chứ không bằng nhân bản.")


if __name__ == '__main__':
    main()