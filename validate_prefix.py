"""
validate_prefix.py — Bằng chứng thực nghiệm cho CƠ CHẾ cốt lõi (mục 4.5).

Đặt ở gốc repo: python validate_prefix.py
Phụ thuộc: numpy (bắt buộc), matplotlib (tuỳ chọn, để vẽ hình).

Kiểm hai phương trình trong mục 3.2 trên embedding THẬT, dùng ĐÚNG ma trận SRP
của V-Engram (import từ src/routing.py):

  (Eq collision)  P[bit trùng | góc θ] = 1 − θ/π
  (Eq prefix)     P[c bit đầu cùng trùng | θ] = (1 − θ/π)^c

Test trên cặp doc–doc (cùng không gian embedding với query). Tính chất SRP đối xứng
nên kết luận giống hệt cặp query–doc, mà KHÔNG cần nạp model.

Ý nghĩa: nếu đường thực nghiệm bám sát đường lý thuyết, đây là bằng chứng TRỰC TIẾP
rằng tín hiệu định tuyến của V-Engram đúng như mô hình — không phải rò rỉ ground truth.
"""
import os
import csv
import numpy as np

# ======================= CẤU HÌNH =======================
CORPUS      = "code"     # "code" | "scifact"
EMB_PATHS   = {"code": "./data/embeddings_20k.npy", "scifact": "./data/scifact_embeddings.npy"}
EMB_PATH    = EMB_PATHS[CORPUS]
LSH_SEED    = 20235956   # PHẢI trùng DEFAULT_LSH_SEED trong src/routing.py
NUM_PROJ    = 5          # L (gộp số liệu trên cả L bảng độc lập)
N_ANCHORS   = 500        # số doc mốc; mỗi mốc ghép với toàn bộ N doc
BIN_DEG     = 5          # độ rộng bin góc (độ)
C_VALUES    = [4, 8, 12, 16, 20]   # các độ dài prefix để kiểm Eq prefix
MIN_PAIRS   = 50         # bỏ qua bin có quá ít cặp khi tính sai lệch tổng hợp
RNG_SEED    = 7
CSV_OUT     = f"prefix_validation_{CORPUS}.csv"
PLOT_OUT    = f"prefix_validation_{CORPUS}.png"
# ========================================================

from src.routing import initialize_lsh_projections
import src.routing as routing


def main():
    assert os.path.isdir("src"), "Hãy chạy ở thư mục gốc repo (cần thấy src/)."
    E = np.load(EMB_PATH).astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)   # chuẩn hoá cho chắc
    N = E.shape[0]
    print(f"[*] CORPUS={CORPUS}  N={N:,}  L={NUM_PROJ}  anchors={N_ANCHORS}")

    # Ma trận SRP TRÙNG V-Engram -> bit sketch cho mọi doc, mỗi bảng (N,160) bool
    initialize_lsh_projections(LSH_SEED)
    projs = routing.PROJECTION_MATRICES
    n_bits = projs[0].shape[1]
    doc_bits = [(E @ projs[t]) > 0 for t in range(NUM_PROJ)]
    print(f"[*] Sketch: {NUM_PROJ} bảng × ({N},{n_bits})")

    rng = np.random.RandomState(RNG_SEED)
    anchors = rng.choice(N, size=min(N_ANCHORS, N), replace=False)

    edges = np.arange(0, 180 + BIN_DEG, BIN_DEG, dtype=float)  # 0..180 (cosine có thể âm)
    nb = len(edges) - 1
    mids = (edges[:-1] + edges[1:]) / 2.0                      # nhãn bin (độ)

    perbit_agree = np.zeros(nb)     # tổng số bit trùng
    bit_total    = np.zeros(nb)     # tổng số lần so bit
    prefix_ok    = {c: np.zeros(nb) for c in C_VALUES}
    pair_total   = np.zeros(nb)     # tổng số cặp
    theta_sum    = np.zeros(nb)     # tổng θ (độ) để lấy trung bình mỗi bin

    for ai, a in enumerate(anchors):
        cos = np.clip(E @ E[a], -1.0, 1.0)
        theta_deg = np.degrees(np.arccos(cos))                # 0..180
        keep = np.ones(N, dtype=bool); keep[a] = False        # bỏ cặp tự thân (θ=0)
        td = theta_deg[keep]
        b = np.clip(np.digitize(td, edges) - 1, 0, nb - 1)    # chỉ số bin mỗi doc
        theta_sum += np.bincount(b, weights=td, minlength=nb)
        for t in range(NUM_PROJ):
            agree = ~(doc_bits[t][keep] ^ doc_bits[t][a])     # (N-1,160) bool: bit trùng
            perbit_agree += np.bincount(b, weights=agree.sum(1), minlength=nb)
            bit_total    += np.bincount(b, minlength=nb) * n_bits
            pair_total   += np.bincount(b, minlength=nb)
            for c in C_VALUES:
                ok = agree[:, :c].all(axis=1)
                prefix_ok[c] += np.bincount(b, weights=ok.astype(float), minlength=nb)
        if (ai + 1) % 100 == 0:
            print(f"    ...{ai+1}/{len(anchors)} anchors")

    # Tỉ lệ thực nghiệm + lý thuyết (lý thuyết tính từ θ TRUNG BÌNH mỗi bin)
    with np.errstate(invalid="ignore", divide="ignore"):
        emp_bit   = perbit_agree / bit_total
        emp_pref  = {c: prefix_ok[c] / pair_total for c in C_VALUES}
        mean_theta = theta_sum / pair_total       # θ trung bình thực tế trong bin (độ)
    theta_rad = np.radians(mean_theta)
    th_bit    = 1.0 - theta_rad / np.pi
    th_pref   = {c: (1.0 - theta_rad / np.pi) ** c for c in C_VALUES}

    valid = pair_total >= MIN_PAIRS

    # ---- In bảng 1: per-bit agreement vs 1 - θ/π ----
    print("\n=== Eq (collision):  P[bit trùng | θ]  vs  1 - θ/π ===")
    print(f"{'θ-bin(°)':>9} {'#cặp':>10} {'emp':>8} {'lý thuyết':>10} {'|sai|':>8}")
    for i in range(nb):
        if pair_total[i] == 0: continue
        mark = "" if valid[i] else "  (ít cặp)"
        print(f"{edges[i]:3.0f}-{edges[i+1]:<3.0f}  {int(pair_total[i]):>10,} "
              f"{emp_bit[i]:8.4f} {th_bit[i]:10.4f} {abs(emp_bit[i]-th_bit[i]):8.4f}{mark}")
    md = np.nanmax(np.abs(emp_bit[valid] - th_bit[valid]))
    print(f"--> Sai lệch tuyệt đối lớn nhất (bin đủ cặp): {md:.4f}")

    # ---- In bảng 2: prefix c-bit vs (1-θ/π)^c ----
    print("\n=== Eq (prefix):  P[c bit đầu trùng | θ]  vs  (1 - θ/π)^c ===")
    head = f"{'θ-bin(°)':>9}" + "".join(f"  c={c}: emp/lt" for c in C_VALUES)
    print(head)
    for i in range(nb):
        if not valid[i]: continue
        row = f"{edges[i]:3.0f}-{edges[i+1]:<3.0f}"
        for c in C_VALUES:
            row += f"  {emp_pref[c][i]:.3f}/{th_pref[c][i]:.3f}"
        print(row)

    # ---- CSV ----
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["theta_lo", "theta_hi", "mean_theta", "n_pairs", "emp_perbit", "theory_perbit"]
        for c in C_VALUES: header += [f"emp_prefix_c{c}", f"theory_prefix_c{c}"]
        w.writerow(header)
        for i in range(nb):
            if pair_total[i] == 0: continue
            r = [edges[i], edges[i+1], round(float(mean_theta[i]), 3), int(pair_total[i]),
                 round(float(emp_bit[i]), 5), round(float(th_bit[i]), 5)]
            for c in C_VALUES:
                r += [round(float(emp_pref[c][i]), 5), round(float(th_pref[c][i]), 5)]
            w.writerow(r)
    print(f"\n[*] Đã ghi: {CSV_OUT}")

    # ---- Hình (tuỳ chọn) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        m = valid
        xs = mean_theta
        order = np.argsort(xs[m]); xm = xs[m][order]
        ax[0].plot(xm, th_bit[m][order], "k-", label=r"theory $1-\theta/\pi$")
        ax[0].plot(xm, emp_bit[m][order], "o", ms=5, label="empirical")
        ax[0].set_xlabel(r"angle $\theta$ (deg)"); ax[0].set_ylabel("P[bit agrees]")
        ax[0].set_title("Per-bit collision"); ax[0].legend(); ax[0].grid(alpha=.3)
        for c in C_VALUES:
            line, = ax[1].plot(xm, emp_pref[c][m][order], "o-", ms=4, label=f"emp c={c}")
            ax[1].plot(xm, th_pref[c][m][order], "--", color=line.get_color(), alpha=.6)
        ax[1].set_xlabel(r"angle $\theta$ (deg)"); ax[1].set_ylabel("P[c-bit prefix agrees]")
        ax[1].set_title(r"Prefix collision (dashed = $(1-\theta/\pi)^c$)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(PLOT_OUT, dpi=150)
        print(f"[*] Đã ghi hình: {PLOT_OUT}")
    except Exception as e:
        print(f"[i] Bỏ qua vẽ hình ({e}). CSV đã đủ để dựng bảng/biểu đồ.")


if __name__ == "__main__":
    main()
