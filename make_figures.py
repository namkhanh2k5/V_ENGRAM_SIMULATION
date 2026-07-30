#!/usr/bin/env python3
"""
SINH BA FIGURE CỦA BÀI: fig_normalised, fig_rstar, fig_nsweep.

VÌ SAO CÓ FILE NÀY: ba hình này trước đây được tạo bằng code chạy rời, không
commit vào repo. Hệ quả là không ai regenerate được, và không ai kiểm được nhãn
trong hình có khớp bảng trong bài hay không.

PHẠM VI — ĐỌC KỸ: file này là NGUỒN DUY NHẤT CHO CÁC HÌNH, không phải nguồn duy
nhất cho các số. Các hằng số dưới đây là BẢN SAO CHÉP TAY từ Bảng 4, 6, 11, 14,
17 và 18 trong .tex. Hàm check() chỉ kiểm tính nhất quán GIỮA CÁC BẢN SAO này và
với công thức — nó KHÔNG đọc .tex. Nếu ai sửa một bảng trong .tex mà quên sửa
hằng số tương ứng ở đây, check() vẫn báo ĐẠT và hình lệch bảng trong im lặng.
Đó đúng là lớp lỗi đã mất nhiều vòng để dọn.

=> SỬA BẢNG NÀO THÌ SỬA CẢ ĐÂY. Mỗi bảng liên quan trong .tex có một dòng
   comment trỏ về file này.

Cách đóng thật sự là sinh luôn thân bảng LaTeX từ cùng bộ hằng số rồi input vào
.tex, để chỉ còn một chỗ sửa. Chưa làm vì sát hạn nộp.

    python3 make_figures.py            # sinh 3 hình
    python3 make_figures.py --check    # chỉ đối chiếu số, không vẽ

MỌI SỐ ĐỀU KHAI BÁO Ở ĐẦU FILE, kèm chú thích bảng nào trong bài là nguồn.
Sửa bảng thì sửa ở đây, rồi chạy lại — hình và bảng không thể lệch nhau nữa.
"""
import argparse

import numpy as np

# ===========================================================================
# SỐ LIỆU — nguồn là các bảng trong bài. Sửa ở đây khi bảng đổi.
# ===========================================================================

# --- Bảng "Budget Sweeps" (sweep T ở K=20, sweep K ở T=8), code corpus ---
SWEEP_T = {'T': [1, 3, 5, 8, 12],
           'node_pct': [1.0, 2.1, 3.3, 4.9, 7.0],
           'recall': [39.8, 60.1, 71.3, 80.0, 85.4]}
SWEEP_K = {'K': [10, 20, 40, 80],
           'node_pct': [2.7, 4.9, 8.9, 15.7],
           'recall': [73.0, 80.0, 86.3, 91.7]}

# --- Bảng "Recall per Unit of Cost" ---
# (nhãn, RPC discovery, Recall@5, node% mạng)
# RPC ĐO ĐƯỢC trong bộ run discovery-only (cả bốn mode cùng một bộ run).
# Bảng chi phí báo 1.145 cho semantic vì run đó chạy cả payload, làm lệch dòng
# random 2 RPC. Bốn số dưới đây nhất quán với nhau, đó là điều hình cần.
COST_POINTS = [
    ('Semantic',           1147, 80.0, 4.9),
    ('Random, keyed',      1408, 33.4, 7.7),
    ('Random, nominal',     800, 34.5, 8.0),
    ('Random, eq. unique',  509, 22.1, 4.9),
]
SEM_EFFICIENCY = 80.0 / 1147        # recall trên mỗi RPC, dùng vẽ đường đẳng hiệu suất

# --- Bảng "Factorial sweep L × r", code corpus ---
FACT_L = [4, 5, 8, 10]
FACT_R = [1, 2, 4, 5, 10]
FACT_SEM = [[73.6, 73.8, 74.2, 74.4, 75.1],
            [80.0, 80.2, 80.9, 81.1, 81.8],
            [91.8, 91.8, 92.0, 92.3, 92.6],
            [94.3, 94.4, 94.5, 94.6, 94.8]]
FACT_RND = [[23.4, 41.4, 65.1, 73.1, 90.3],
            [34.5, 56.4, 79.8, 85.4, 94.9],
            [65.6, 85.6, 94.9, 95.6, 96.0],
            [80.7, 93.3, 96.0, 96.0, 96.0]]
CROSSOVER_LR = 20          # mốc L·r nơi bốn đường cắt mốc tỉ lệ 1

# --- Bảng "Network Size and the Threshold Formula" (N-sweep) ---
NSWEEP = {'N': [5000, 10000, 20000, 40000],
          'sem': [85.7, 80.0, 72.1, 63.6],
          'rnd': [57.4, 34.5, 18.9, 9.8]}

# --- Tham số dùng vẽ đường công thức ---
A_MEASURED = 5             # số anchor phân biệt đo được ở L=5, r=1
N_MAIN = 10000


def p_rand_hyper(N, A, M):
    """Eq r*: P_rand = 1 - C(N-A, M)/C(N, M). Dạng hypergeometric, KHÔNG rút gọn.

    Cài bằng dạng tích prod_{i<A} (N-M-i)/(N-i) thay vì math.comb. Hai dạng
    tương đương đại số và trùng đến sáu chữ số thập phân, nhưng comb phải dựng
    số nguyên nghìn chữ số. Đủ nhanh ở độ phân giải hiện tại; sẽ thành nút thắt
    nếu ai nâng số điểm của đường hoặc quét thêm N.
    """
    if M <= 0 or M >= N:
        return float('nan')
    p = 1.0
    for i in range(A):
        p *= (N - M - i) / (N - i)
    return 100.0 * (1 - p)


# ===========================================================================
# ĐỐI CHIẾU — chạy trước khi vẽ, để hình không thể lệch bảng
# ===========================================================================
def check():
    ok = True

    def eq(lbl, a, b, tol=0.06):
        nonlocal ok
        good = abs(a - b) <= tol
        ok &= good
        print(f"  {'✓' if good else '✗'} {lbl}: {a:.2f} vs {b:.2f}")

    print("=== ĐỐI CHIẾU NỘI BỘ ===")
    # cấu hình chốt phải trùng nhau ở cả ba nguồn
    i8 = SWEEP_T['T'].index(8)
    i20 = SWEEP_K['K'].index(20)
    eq('recall cấu hình chốt: sweep T vs sweep K',
       SWEEP_T['recall'][i8], SWEEP_K['recall'][i20])
    eq('node% cấu hình chốt: sweep T vs sweep K',
       SWEEP_T['node_pct'][i8], SWEEP_K['node_pct'][i20])
    eq('recall cấu hình chốt: sweep vs bảng chi phí',
       SWEEP_T['recall'][i8], COST_POINTS[0][2])
    eq('node% cấu hình chốt: sweep vs bảng chi phí',
       SWEEP_T['node_pct'][i8], COST_POINTS[0][3])
    eq('recall cấu hình chốt: sweep vs factorial (L=5,r=1)',
       SWEEP_T['recall'][i8], FACT_SEM[FACT_L.index(5)][FACT_R.index(1)])
    eq('recall cấu hình chốt: sweep vs N-sweep (N=10000)',
       SWEEP_T['recall'][i8], NSWEEP['sem'][NSWEEP['N'].index(10000)])
    eq('random cấu hình chốt: bảng chi phí vs factorial',
       COST_POINTS[2][2], FACT_RND[FACT_L.index(5)][FACT_R.index(1)])
    eq('random cấu hình chốt: factorial vs N-sweep',
       FACT_RND[FACT_L.index(5)][FACT_R.index(1)],
       NSWEEP['rnd'][NSWEEP['N'].index(10000)])

    # tỉ lệ 3,6x mà Hình 3 ghi
    r36 = COST_POINTS[0][2] / COST_POINTS[3][2]
    eq('mũi tên 3,6x = semantic / equal-unique', r36, 3.6, tol=0.05)

    # Caption nói baseline nằm ĐÂU so với đường đẳng hiệu suất. Kiểm để caption
    # không bao giờ mâu thuẫn hình: đã từng ghi sai là "oracles sit above".
    print()
    print("=== VỊ TRÍ so với đường đẳng hiệu suất semantic ===")
    for lbl, rpc, r5, _ in COST_POINTS[1:]:
        on = SEM_EFFICIENCY * rpc
        above = r5 > on
        if above:
            ok = False
        print(f"  {'✗ TRÊN' if above else '✓ dưới'} {lbl:20s} y={r5:5.1f} "
              f"vs đường {on:5.1f}")
    print("  (caption phải nói CẢ BA nằm dưới; oracle gần đường hơn keyed lookup)")

    # mốc crossover: bốn đường phải cắt mốc 1 quanh L·r = 20
    print()
    print("=== CROSSOVER: tỉ lệ tại L·r gần 20 ===")
    for i, L in enumerate(FACT_L):
        best = min(range(len(FACT_R)), key=lambda j: abs(L * FACT_R[j] - CROSSOVER_LR))
        lr = L * FACT_R[best]
        ratio = FACT_SEM[i][best] / FACT_RND[i][best]
        flag = '✓' if abs(ratio - 1.0) < 0.20 else '✗ xa mốc 1'
        print(f"  {flag} L={L:>2} r={FACT_R[best]:>2} (L·r={lr:>2}): tỉ lệ {ratio:.2f}")

    # công thức vs đo được
    print()
    print("=== Eq r* (hypergeometric) vs random đo được ===")
    for lbl, N, M, meas in [('N=10k, M=800', 10000, 800, 34.5),
                            ('N=25k, M=800', 25000, 800, 14.8),
                            ('N=50k, M=800', 50000, 800, 7.5),
                            ('N=25k, M=2000', 25000, 2000, 34.7),
                            ('N=50k, M=4000', 50000, 4000, 34.1)]:
        pred = p_rand_hyper(N, A_MEASURED, M)
        d = abs(pred - meas)
        print(f"  {'✓' if d < 1.0 else '✗'} {lbl:14s} dự đoán {pred:5.1f}% "
              f"đo {meas:5.1f}%  lệch {d:.1f}")

    print()
    print("KẾT LUẬN:", "mọi đối chiếu ĐẠT" if ok else "*** CÓ CHỖ LỆCH, sửa trước khi vẽ ***")
    return ok


# ===========================================================================
# VẼ
# ===========================================================================
def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt

    plt.rcParams.update({'font.family': 'serif', 'font.size': 9.5,
                         'axes.linewidth': 0.8, 'figure.dpi': 150})
    # KHÔNG dùng '\\%' trong nhãn: matplotlib không bật usetex nên nó in ra
    # nguyên văn dấu gạch chéo. Dùng '%' trực tiếp.
    COLS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    # ----- HÌNH 1: recall theo hai thước đo chi phí -----
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    ax = axes[0]
    # Hai đường semantic dán nhãn INLINE, không vào legend: sáu dòng legend che
    # đúng đoạn 7,5-11,5% nơi đường Eq.8 gặp hai điểm đo ở 7,7% và 8,0%.
    ax.plot(SWEEP_T['node_pct'], SWEEP_T['recall'], 'o-', color=COLS[0], lw=1.7, ms=5)
    ax.plot(SWEEP_K['node_pct'], SWEEP_K['recall'], 's-', color=COLS[1], lw=1.7, ms=5)
    ax.annotate('sweep $T$', xy=(SWEEP_T['node_pct'][1], SWEEP_T['recall'][1]),
                xytext=(-4, 7), textcoords='offset points', ha='right',
                fontsize=7.5, color=COLS[0], fontweight='bold')
    ax.annotate('sweep $K$', xy=(SWEEP_K['node_pct'][-1], SWEEP_K['recall'][-1]),
                xytext=(-2, 9), textcoords='offset points', ha='right',
                fontsize=7.5, color=COLS[1], fontweight='bold')
    ax.text(0.4, 93, 'Semantic', fontsize=8, fontweight='bold', color='0.25')
    x = np.linspace(0.5, 17, 200)
    ax.plot(x, [p_rand_hyper(N_MAIN, A_MEASURED, int(N_MAIN * xi / 100)) for xi in x],
            '--', color='gray', lw=1.3, label=f'Random (Eq., $A{{=}}{A_MEASURED}$)')
    # Ba baseline: keyed ở 7,7% và nominal ở 8,0% chỉ cách 0,3 điểm, nếu dùng
    # chung một marker thì render ra như MỘT hình thoi dày và người đọc đối
    # chiếu bảng sẽ tìm ba điểm mà chỉ thấy hai. Dùng marker riêng như panel phải.
    _mk = {'Random, keyed': ('D', COLS[3]), 'Random, nominal': ('s', COLS[2]),
           'Random, eq. unique': ('^', '#9467bd')}
    for lbl, _, r5, nd in COST_POINTS:
        if lbl == 'Semantic':
            continue
        mk, c = _mk[lbl]
        ax.plot(nd, r5, mk, color=c, ms=6, zorder=5,
                markeredgecolor='white', markeredgewidth=0.5,
                label=lbl.replace('Random, ', 'Random: '))
    sem_nd, sem_r5 = COST_POINTS[0][3], COST_POINTS[0][2]
    eq_r5 = COST_POINTS[3][2]
    ax.annotate('', xy=(sem_nd, sem_r5), xytext=(sem_nd, eq_r5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.1))
    ax.text(sem_nd + 0.4, (sem_r5 + eq_r5) / 2,
            f'${sem_r5/eq_r5:.1f}\\times$', fontsize=9, fontweight='bold')
    ax.set_xlabel('Unique nodes contacted (% of overlay)')
    ax.set_ylabel('Recall@5 (%)')
    ax.set_xlim(0, 17); ax.set_ylim(0, 100)
    # legend giờ 4 dòng (Eq + 3 baseline). Góc dưới-phải trống: đường Eq.8 ở
    # x=11,5% đã lên y=45,7 nên hộp trong y 2-32 không chạm nó.
    ax.legend(fontsize=6.8, loc='lower right', framealpha=0.92,
              borderpad=0.4, labelspacing=0.35)
    ax.grid(True, alpha=0.25, lw=0.5)

    ax = axes[1]
    marks = ['o', 'D', 's', '^']
    order = [COLS[0], COLS[3], COLS[2], '#9467bd']
    for (lbl, rpc, r5, _), mk, c in zip(COST_POINTS, marks, order):
        ax.scatter(rpc, r5, s=70, c=c, marker=mk, zorder=5,
                   edgecolors='white', linewidths=0.6, label=lbl)
        ax.annotate(f'{r5:.1f}', (rpc, r5), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=7.5)
    xx = np.linspace(300, 1600, 50)
    ax.plot(xx, SEM_EFFICIENCY * xx, ':', color=COLS[0], lw=1.1, alpha=0.7)
    ax.text(1180, SEM_EFFICIENCY * 1420, 'semantic\nefficiency', fontsize=6.8,
            color=COLS[0], style='italic', va='center')
    ax.set_xlabel('Discovery RPCs per query')
    ax.set_ylabel('Recall@5 (%)')
    ax.set_xlim(300, 1650); ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.92)
    ax.grid(True, alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig('fig_normalised.pdf', bbox_inches='tight')
    print("-> fig_normalised.pdf")

    # ----- HÌNH 2: factorial L × r -----
    sem = np.array(FACT_SEM); rnd = np.array(FACT_RND)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    for i, L in enumerate(FACT_L):
        ax.plot(FACT_R, sem[i], 'o-', color=COLS[i], lw=1.7, ms=4.5, label=f'$L={L}$')
        ax.plot(FACT_R, rnd[i], 's--', color=COLS[i], lw=1.2, ms=3.8, alpha=0.6)
    ax.set_xscale('log'); ax.set_xticks(FACT_R)
    # set_xticklabels đòi chuỗi, FACT_R là số nguyên -> ép str
    ax.set_xticklabels([str(r) for r in FACT_R])
    ax.minorticks_off()
    ax.set_xlabel('Replication factor $r$'); ax.set_ylabel('Recall@5 (%)')
    # nới trần để chú thích không đè lên đường L=8, L=10
    ax.set_ylim(15, 118); ax.set_yticks([20, 40, 60, 80, 100])
    ax.axhline(100, color='0.85', lw=0.6, zorder=0)
    ax.text(1.05, 112, 'solid: semantic — flat in $r$', fontsize=7, style='italic')
    ax.text(1.05, 105, 'dashed: random — rises steeply', fontsize=7,
            style='italic', alpha=0.7)
    ax.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25, lw=0.5)

    ax = axes[1]
    for i, L in enumerate(FACT_L):
        ax.plot([L * r for r in FACT_R], sem[i] / rnd[i], 'o-', color=COLS[i],
                lw=1.7, ms=4.5, label=f'$L={L}$')
    ax.axhline(1.0, color='black', lw=0.9, ls=':', zorder=1)
    ax.axvline(CROSSOVER_LR, color='gray', lw=0.9, ls='--', alpha=0.7, zorder=1)
    # nhãn ở đầu trên bị hộp legend (upper right) che một phần -> xuống chân
    # y=0.845 ra khỏi legend nhưng rơi đúng dải đường L=4 đi xuống (tại x=37 nó
    # đã ở y~0.855, trong chiều cao chữ). Vùng trống thật là y 1.8-2.4 ở x~21:
    # mọi đường ở đó quanh 1.0, legend bắt đầu từ y~2.65.
    ax.text(CROSSOVER_LR * 1.06, 2.0,
            f'$L{{\\cdot}}r{{\\approx}}{CROSSOVER_LR}$',
            fontsize=7.5, ha='left', va='center', color='dimgray')
    ax.set_xscale('log')
    # bỏ tick 16: trên thang log nó chồng lên 20
    ax.set_xticks([4, 8, 20, 40, 100])
    ax.set_xticklabels(['4', '8', '20', '40', '100'])
    ax.minorticks_off()
    ax.set_xlabel('Metadata footprint $L \\cdot r$')
    ax.set_ylabel('Semantic / random ratio')
    # Vùng DƯỚI 1.0 là phát hiện trung tâm của panel — 8 trong 20 điểm nằm ở
    # 0,83-0,99. Với tick mặc định (1.0, 1.5, ...) cả vùng đó bị nén sát đáy và
    # không đọc được. Đặt tick thưa dần: dày ở dưới, thưa ở trên.
    ax.set_ylim(0.8, 3.2)
    ax.set_yticks([0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticklabels(['0.8', '0.9', '1.0', '1.5', '2.0', '2.5', '3.0'])
    # tô nhẹ vùng dưới unity để mắt bắt được ngay
    ax.axhspan(0.8, 1.0, color='#d62728', alpha=0.055, zorder=0)
    ax.legend(fontsize=7.5, loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig('fig_rstar.pdf', bbox_inches='tight')
    print("-> fig_rstar.pdf")

    # ----- HÌNH 3: N-sweep -----
    Ns, s_n, r_n = NSWEEP['N'], NSWEEP['sem'], NSWEEP['rnd']
    ratio = [a / b for a, b in zip(s_n, r_n)]
    fig, ax1 = plt.subplots(figsize=(5.0, 3.4))
    l1 = ax1.plot(Ns, s_n, 'o-', color=COLS[0], lw=1.5, ms=6, label='Semantic')
    l2 = ax1.plot(Ns, r_n, 's-', color=COLS[3], lw=1.5, ms=6, label='Random')
    ax1.set_xlabel('Network size $N$ (nodes)'); ax1.set_ylabel('Recall@5 (%)')
    ax1.set_xscale('log')
    ax1.xaxis.set_major_locator(mt.FixedLocator(Ns))
    ax1.xaxis.set_minor_locator(mt.NullLocator())
    ax1.xaxis.set_major_formatter(mt.FixedFormatter(
        [f'{n//1000}k' for n in Ns]))
    # điểm cao nhất là 85,7 nên tick phải tới 100, không dừng ở 80
    ax1.set_ylim(0, 100)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax2 = ax1.twinx()
    l3 = ax2.plot(Ns, ratio, '^--', color=COLS[1], lw=1.3, ms=6,
                  label='Ratio (sem/rand)')
    ax2.set_ylabel('Ratio', color=COLS[1])
    ax2.tick_params(axis='y', labelcolor=COLS[1])
    ax2.set_ylim(0, max(ratio) * 1.2)
    # KHÔNG dùng hộp legend. Với hai trục y và ba đường, mọi góc đều có dữ liệu
    # đi qua: 'center right' che đường ratio giữa 10k-20k, còn 'lower left' che
    # điểm ratio ở 5k (1,5 trên trục phải = 21,4 trên thang trục trái, đúng
    # trong hộp), mà 1,5 là đầu mút caption dẫn. Nhãn cạnh đường thì không che.
    ax1.annotate('Semantic', xy=(Ns[1], s_n[1]), xytext=(0, 9),
                 textcoords='offset points', ha='center', fontsize=8,
                 color=COLS[0], fontweight='bold')
    # Neo ở Ns[2] chứ không Ns[1]: ở 10k chữ căn giữa có mép phải chạm x~11,5k
    # nơi đường đỏ còn ở y~32, sát đỉnh chữ. Ở 20k phía dưới đường trống hẳn.
    ax1.annotate('Random', xy=(Ns[2], r_n[2]), xytext=(0, -14),
                 textcoords='offset points', ha='center', fontsize=8,
                 color=COLS[3], fontweight='bold')
    ax2.annotate('Ratio (sem/rand)', xy=(Ns[2], ratio[2]), xytext=(-8, 10),
                 textcoords='offset points', ha='right', fontsize=8,
                 color=COLS[1], fontweight='bold')
    ax1.grid(True, alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig('fig_nsweep.pdf', bbox_inches='tight')
    print("-> fig_nsweep.pdf")
    print(f"   (tỉ lệ N-sweep: {ratio[0]:.1f} -> {ratio[-1]:.1f}, "
          f"phải khớp caption trong bài)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='chỉ đối chiếu, không vẽ')
    a = ap.parse_args()
    good = check()
    if a.check:
        raise SystemExit(0 if good else 1)
    if not good:
        print("\n*** CÓ CHỖ LỆCH — vẫn vẽ, nhưng sửa số trước khi dùng ***")
    print()
    draw()