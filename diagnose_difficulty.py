"""
diagnose_difficulty.py — Vì sao HNSW & Centralized-LSH ra 100%?
Chạy ở gốc repo: python diagnose_difficulty.py
KHÔNG cần model/faiss — chỉ đọc GT json (đã có sẵn cosine_similarity).

Mục đích: xác nhận giả thuyết "corpus quá dễ" (near-duplicate / query gần trùng doc)
để biết phải đổi benchmark hay chỉ cần trình bày minh bạch.
"""
import json
import numpy as np

GT_PATH = "./data/code_ground_truth.json"

gt = json.load(open(GT_PATH, "r", encoding="utf-8"))
n = len(gt)
print(f"[*] {GT_PATH} — {n} query\n")

# 1) Cosine của top-1: nếu nhiều cái ~1.0 => query gần trùng 1 doc => task tầm thường
top1 = np.array([it["top_5_results"][0]["cosine_similarity"] for it in gt])
print("Cosine top-1: mean=%.4f  median=%.4f  min=%.4f" % (top1.mean(), np.median(top1), top1.min()))
for th in (0.999, 0.99, 0.95, 0.90):
    print("   %% query có cosine top-1 > %.3f : %5.1f%%" % (th, 100 * (top1 > th).mean()))

# 2) GT top-5 là dãy index liên tiếp => cụm near-duplicate tuần tự
def is_consecutive(idxs):
    s = sorted(idxs)
    return all(s[i + 1] - s[i] == 1 for i in range(len(s) - 1))
runs = sum(is_consecutive([r["index"] for r in it["top_5_results"]]) for it in gt)
print("\n%% GT top-5 toàn index liên tiếp (near-dup cluster): %.1f%%" % (100 * runs / n))

# 3) Độ "dễ" tổng thể: cosine trung bình của cả top-5
allcos = np.array([[r["cosine_similarity"] for r in it["top_5_results"]] for it in gt])
print("Cosine trung bình GT top-5: %.4f" % allcos.mean())
print("Khoảng cách cosine top1 - top5 (mean): %.4f" % (allcos[:, 0] - allcos[:, 4]).mean())

print("\nDIỄN GIẢI:")
print(" - Nếu phần lớn cosine top-1 > 0.99 và/hoặc nhiều top-5 liên tiếp:")
print("   => corpus near-duplicate, mọi ANN bão hoà 100%, benchmark không phân biệt được.")
print("   => Nên: (a) dùng SciFact làm benchmark chính, và/hoặc (b) khử trùng lặp corpus code.")
print(" - Nếu cosine top-1 phân tán (nhiều cái < 0.95):")
print("   => corpus đủ khó; 100% là do tín hiệu LSH thật sự mạnh, trình bày minh bạch là đủ.")