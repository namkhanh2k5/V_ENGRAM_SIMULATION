# `prepare/` — corpus and embedding construction

These scripts build every input the simulator reads from `data/`.

The two notebooks were run in Google Colab on a T4 GPU and are kept as
notebooks, unedited, because the data in the paper came from exactly this code
and `!pip` / `google.colab.drive` cells are not valid Python outside Jupyter. To
run them elsewhere, delete the mount and install cells and set `OUT_DIR` to a
local path; nothing else depends on Colab.

`03_build_pq.py` is a plain script that runs anywhere, no GPU required.

## What runs in what order

| step | produces | time |
|---|---|---|
| `01_build_corpora.ipynb` | `code` (20k) and `scifact` (5.2k): embeddings, ground truth, PQ codebooks at `m=256` and `m=512` | 40 min, GPU |
| `02_extend_weakscaling.ipynb` | `code50k` and `code100k`, each with its own ground truth and codebook | 90 min, GPU |
| `03_build_pq.py` | rebuilds a PQ codebook from embeddings already on disk | 5 min, CPU |

`02` requires the output of `01`: it appends new functions to the existing
20,000 and asserts byte-for-byte that the first 20,000 embeddings are unchanged,
so the smaller corpus stays a strict prefix of the larger one.

`03` is separate because quantizer training is the step most likely to be rerun
— it reads embeddings that already exist, takes minutes on a CPU, and needs no
Colab. It also compares against whatever codebook is already in `data/` and
reports both reconstruction errors, so a rebuild can be checked rather than
assumed equivalent:

```bash
python3 prepare/03_build_pq.py --data-dir data --corpus code
python3 prepare/03_build_pq.py --data-dir data --corpus code --m 256   # the older variant
```

Code assignments need not match a previous build bit for bit, since k-means is
only locally optimal; comparable reconstruction MSE is the check that matters.

## Fixed settings

| setting | value |
|---|---|
| seed | `20235956` for sampling, shuffling and query selection |
| embedding model | `BAAI/bge-large-en-v1.5`, 1024 dimensions |
| normalization | `normalize_embeddings=True` at encode time |
| BGE retrieval instruction | **not** prepended to queries |
| tokenizer, max length, truncation | model defaults, not overridden |
| similarity | inner product, which equals cosine on normalized vectors |
| ground truth | exact top-10 by `faiss.IndexFlatIP` |
| PQ | `faiss.ProductQuantizer`, `m=512`, `d_sub=2`, 8 bits, 256 centroids per subquantizer |

The paper uses the `m=512` variant. The `m=256` variant exists because it was
tried first and lost eight to twelve Recall@5 points to quantization error; both
are kept so that the comparison can be rerun.

## How the code corpus avoids self-retrieval

A query is the first sentence of a function's docstring. The document is the
same function **with that docstring removed** — by direct substring deletion
where the docstring appears verbatim, otherwise by removing the first
triple-quoted block. A function is admitted only if its docstring has at least
five whitespace-separated tokens and its stripped body is at least forty
characters after whitespace collapsing. Exact-duplicate bodies are dropped by
string equality; near-duplicate detection is not applied.

Both scripts print a diagnostic after building each corpus:

```
>>> GÓC query→GT5: mean=43.4° median=41.9° p90=57.2°
```

This is the angle between a query embedding and its ground-truth neighbours. A
mean near 43° is what a semantically related but textually distinct pair should
give. Values below roughly 35° would indicate that query text still appears in
the documents, and the script warns when that happens.

## Output files the simulator expects in `data/`

For each corpus `{name}` in `code`, `scifact`, `code50k`, `code100k`:

```
{name}_corpus_embeddings.npy   (N, 1024) float32, L2-normalized
{name}_query_embeddings.npy    (500, 1024) float32, L2-normalized
{name}_corpus_texts.json       corpus strings, index-aligned with embeddings
{name}_ground_truth.json       exact top-10 per query, with cosine similarities
{name}_pq_codebook_m512.npy    (512, 256, 2) float32
{name}_pq_codes_m512.npy       (N, 512) uint8
```

The `.npy` files are not in the repository: `code100k_corpus_embeddings.npy`
alone is 410 MB, above GitHub's hard per-file limit. Rebuild them with these
scripts, or request the archive from the corresponding author.

## The SQuAD block

`01_build_corpora.ipynb` ends with a cell that builds a SQuAD corpus. It is not
used anywhere in the paper and is kept only because removing it would change
the file from what was actually run.