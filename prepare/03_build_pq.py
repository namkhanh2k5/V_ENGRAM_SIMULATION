#!/usr/bin/env python3
"""
Rebuild PQ codebooks from existing embeddings. Runs anywhere, no Colab, no GPU.

    python3 prepare/03_build_pq.py --data-dir data --corpus code
    python3 prepare/03_build_pq.py --data-dir data --corpus code --m 256

Why this exists as a plain script while the other two are notebooks: the
notebooks encode and download several gigabytes and need a GPU, so they are kept
as the Colab artefacts they are. Quantizer training reads embeddings that are
already on disk, takes a few minutes on a CPU, and is the step most likely to be
rerun -- so it is worth having in a form that runs without Colab.

Verifies against the shipped codebook when one is present, so a rebuild can be
checked rather than assumed equivalent.
"""
import argparse
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("faiss not installed. Try: pip install faiss-cpu")

_FAISS_VER = getattr(faiss, "__version__", "unknown")


def build(data_dir, corpus, m, nbits, seed):
    src = os.path.join(data_dir, f"{corpus}_corpus_embeddings.npy")
    if not os.path.exists(src):
        sys.exit(f"missing {src} -- build corpora first (see prepare/README.md)")

    E = np.load(src).astype("float32")
    n, d = E.shape
    if d % m:
        sys.exit(f"dimension {d} is not divisible by m={m}")
    d_sub = d // m

    norms = np.linalg.norm(E, axis=1)
    print(f"faiss       {_FAISS_VER}")
    print(f"corpus      {n:,} x {d}")
    print(f"norms       mean {norms.mean():.4f}, min {norms.min():.4f}, "
          f"max {norms.max():.4f}")
    if abs(norms.mean() - 1.0) > 0.01:
        print("  WARNING: embeddings are not unit-norm. Inner-product ADC "
              "assumes they are.")

    print(f"training    m={m}, d_sub={d_sub}, {nbits} bits "
          f"({1 << nbits} centroids per subquantizer, {m} bytes per object)")
    t0 = time.time()
    # Both of these are version-dependent: older faiss builds expose neither
    # omp_set_num_threads nor ProductQuantizer.cp. Neither is required for a
    # correct codebook, so degrade rather than fail.
    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(os.cpu_count() or 1)

    pq = faiss.ProductQuantizer(d, m, nbits)
    if hasattr(pq, "cp") and hasattr(pq.cp, "seed"):
        # Seeds the internal k-means so a rebuild is reproducible rather than
        # merely equivalent.
        pq.cp.seed = seed
    else:
        print("            note: this faiss build does not expose the clustering "
              "seed, so a rebuild will differ slightly from a previous one")
    pq.train(E)
    codes = pq.compute_codes(E)
    cb = faiss.vector_to_array(pq.centroids).reshape(m, 1 << nbits, d_sub)
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    mse = float(((pq.decode(codes) - E) ** 2).sum(axis=1).mean())
    print(f"            done in {time.time() - t0:.0f}s, "
          f"reconstruction MSE {mse:.4f}")

    suffix = "_m512" if m == 512 else ""
    cb_path = os.path.join(data_dir, f"{corpus}_pq_codebook{suffix}.npy")
    cd_path = os.path.join(data_dir, f"{corpus}_pq_codes{suffix}.npy")

    # compare against whatever is already there before overwriting
    if os.path.exists(cb_path):
        old_cb = np.load(cb_path)
        if old_cb.shape != cb.shape:
            print(f"  existing codebook has shape {old_cb.shape}, "
                  f"new one {cb.shape} -- different configuration")
        else:
            old_cd = np.load(cd_path)
            rec_old = np.concatenate(
                [old_cb[j][old_cd[:, j]] for j in range(m)], axis=1)
            mse_old = float(((rec_old - E) ** 2).sum(axis=1).mean())
            agree = float((old_cd == codes).mean())
            print(f"  existing    MSE {mse_old:.4f}, "
                  f"codes agree with rebuild on {100*agree:.1f}% of positions")
            print("  Code assignments need not match exactly: k-means is only "
                  "locally optimal. Comparable MSE is the check that matters.")

    np.save(cb_path, cb)
    np.save(cd_path, codes)
    print(f"wrote       {cb_path}  {cb.shape}")
    print(f"            {cd_path}  {codes.shape} {codes.dtype}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--corpus", default="code",
                    help="code, scifact, code50k, code100k")
    ap.add_argument("--m", type=int, default=512,
                    help="subquantizers; 512 is what the paper uses")
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20235956)
    a = ap.parse_args()
    build(a.data_dir, a.corpus, a.m, a.nbits, a.seed)