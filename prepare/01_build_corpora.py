# =====================================================================
# 01_build_corpora.py — builds the code and SciFact corpora from scratch
#
# ORIGIN: Google Colab notebook, exported verbatim. It contains Colab-only
# calls (`!pip`, `google.colab.drive`) and will not run unmodified outside
# Colab. It is included because it is the script that actually produced the
# data used in the paper, and reproducing that data exactly matters more than
# packaging convenience.
#
# WHAT IT PRODUCES, per corpus {code, scifact}:
#   {name}_corpus_embeddings.npy   (N, 1024) float32, L2-normalized
#   {name}_query_embeddings.npy    (500, 1024) float32, L2-normalized
#   {name}_corpus_texts.json       corpus strings, index-aligned
#   {name}_ground_truth.json       exact top-10 per query by FAISS IndexFlatIP
#   {name}_pq_codebook.npy         (256, 256, 4) — m=256 variant
#   {name}_pq_codes.npy            (N, 256) uint8
#
# The m=512 codebook used by the paper is produced by the second cell block
# in this file; it retrains PQ on the same embeddings without re-encoding.
#
# KEY SETTINGS (all fixed, see prepare/README.md for rationale):
#   SEED = 20235956            sampling and shuffling
#   model = BAAI/bge-large-en-v1.5, 1024-dim, normalize_embeddings=True
#   no BGE retrieval instruction prepended to queries
#   code: CodeSearchNet Python train split, first shard, 20,000 functions
#         query = first sentence of docstring; document = function body with
#         the docstring removed. Filters: docstring >= 5 tokens, body >= 40
#         characters after whitespace collapsing, exact-duplicate bodies
#         dropped by string equality.
#   scifact: BeIR/scifact, corpus = "title. text", queries = 500 shuffled
#         claims of >= 3 tokens.
#   ground truth: exact inner product (= cosine on normalized vectors)
#
# The SQuAD block at the end is NOT used by the paper.
# =====================================================================


from google.colab import drive
drive.mount('/content/drive')

import os
OUT_DIR = '/content/drive/MyDrive/v_engram_final_data'
os.makedirs(OUT_DIR, exist_ok=True)
print('Thư mục đích:', OUT_DIR)

!pip -q install faiss-cpu sentence-transformers "datasets>=2.19" 2>/dev/null
!pip uninstall -y hf_xet hf-xet 2>/dev/null
print('Cài xong.')

# =====================================================================
#  CELL BỔ SUNG cho notebook Colab — sinh PQ codebook m=512 (nén nhẹ hơn)
#
#  VÌ SAO: PQ hiện tại m=256, d_sub=4 (256 byte/doc, nén 16x) đang ăn mất
#  8-12 điểm Recall@5. Số đo từ run_full:
#      L=5  T=8:  PQ on 76.8%  |  PQ off 84.5%  ->  mất 7.7đ
#      L=8  T=8:  PQ on 83.4%  |  PQ off 94.0%  ->  mất 10.6đ
#      L=12 T=8:  PQ on 86.7%  |  PQ off 99.1%  ->  mất 12.4đ
#
#  PQ là trục DUY NHẤT không ảnh hưởng tỉ lệ semantic/random, vì cả semantic
#  lẫn random routing đều dùng cùng một PQ. Giảm nén nâng recall cho CẢ HAI,
#  tỉ lệ giữ nguyên. Khác hẳn L và r — hai cái đó nhân metadata nên kéo random
#  lên theo và giết luận điểm.
#
#  m=512, d_sub=2 -> 512 byte/doc (nén 8x). Sai số lượng tử hoá giảm ~10 lần
#  (MSE 0.078 -> 0.008 trên dữ liệu Gaussian chuẩn hoá).
#
#  KHÔNG cần embed lại. Chỉ train PQ mới trên embeddings đã có. ~5 phút/bộ.
#
#  CHẠY: mở notebook cũ, chạy CELL 0 (mount Drive), rồi dán cell này vào chạy.
# =====================================================================
import os, time
import numpy as np
import faiss

OUT_DIR = '/content/drive/MyDrive/v_engram_final_data'
PQ_NBITS = 8          # 8-bit -> 256 centroid/subquantizer (giữ nguyên)
NEW_M = 512           # m mới: 1024/512 = 2 chiều mỗi subvector

for name in ['code', 'scifact', 'squad']:
    src = f'{OUT_DIR}/{name}_corpus_embeddings.npy'
    if not os.path.exists(src):
        print(f'[{name}] BỎ QUA — không thấy {src}')
        continue

    t0 = time.time()
    E = np.load(src).astype('float32')
    d = E.shape[1]
    d_sub = d // NEW_M
    assert d % NEW_M == 0, f'{d} không chia hết cho {NEW_M}'

    print(f'[{name}] corpus={E.shape} -> PQ m={NEW_M}, d_sub={d_sub} '
          f'({NEW_M} byte/doc, nén {d*4//NEW_M}x)')

    pq = faiss.ProductQuantizer(d, NEW_M, PQ_NBITS)
    pq.train(E)
    codes = pq.compute_codes(E)                                   # (N, 512) uint8
    cb = faiss.vector_to_array(pq.centroids).reshape(NEW_M, 1 << PQ_NBITS, d_sub)
    cb = np.ascontiguousarray(cb, dtype=np.float32)               # (512, 256, 2)

    np.save(f'{OUT_DIR}/{name}_pq_codebook_m512.npy', cb)
    np.save(f'{OUT_DIR}/{name}_pq_codes_m512.npy', codes)

    # So sánh sai số với codebook m=256 cũ
    recon_new = pq.decode(codes)
    mse_new = float(((recon_new - E) ** 2).sum(axis=1).mean())
    old_cb_path = f'{OUT_DIR}/{name}_pq_codebook.npy'
    msg_old = ''
    if os.path.exists(old_cb_path):
        old_cb = np.load(old_cb_path)                             # (256,256,4)
        old_codes = np.load(f'{OUT_DIR}/{name}_pq_codes.npy')
        m_old, _, ds_old = old_cb.shape
        recon_old = np.concatenate(
            [old_cb[j][old_codes[:, j]] for j in range(m_old)], axis=1)
        mse_old = float(((recon_old - E) ** 2).sum(axis=1).mean())
        msg_old = f' | m=256 cũ MSE={mse_old:.4f} -> giảm {mse_old/mse_new:.1f}x'

    print(f'[{name}] ✓ {time.time()-t0:.0f}s | codebook{cb.shape} codes{codes.shape} '
          f'| MSE={mse_new:.4f}{msg_old}')

print('''
=================== XONG ===================
File mới trên Drive:
  {code,scifact,squad}_pq_codebook_m512.npy   (512, 256, 2)
  {code,scifact,squad}_pq_codes_m512.npy      (N, 512) uint8

Tải về, bỏ vào ./data/ của repo, rồi chạy:
  python3 main_simulation_v2.py --dataset code --nodes 10000 --seed 20235956 \\
      --num-tables 5 --multi-probe 8 --meta-anchors 1 --k-query 20 \\
      --pq-variant m512 --nq 200 --use-pq

So với bản m=256 cùng cấu hình (đã đo: 76.8%). Nếu m512 đưa lên ~82-84% mà
tỉ lệ sem/rand vẫn ~2.2x -> đó là cấu hình cho paper.
============================================''')

import os
MIRROR = 'https://hf-mirror.com/BAAI/bge-large-en-v1.5/resolve/main'
MODEL_DIR = '/content/bge'
os.makedirs(MODEL_DIR + '/1_Pooling', exist_ok=True)

files = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.txt', 'special_tokens_map.json',
         'sentence_bert_config.json', 'config_sentence_transformers.json',
         'modules.json', 'model.safetensors', '1_Pooling/config.json']

for f in files:
    dst = f'{MODEL_DIR}/{f}'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = os.system(f'wget -q --show-progress -O "{dst}" "{MIRROR}/{f}"')
    print(('OK  ' if r == 0 else 'LỖI ') + f)

print('\nKiểm tra model.safetensors (phải ~1.3G, KHÔNG được vài KB):')
os.system(f'ls -lh {MODEL_DIR}/model.safetensors')

# Sao lưu vào Drive để lần sau khỏi tải lại:
os.system(f'cp -r {MODEL_DIR} /content/drive/MyDrive/bge_model 2>/dev/null')
print('Đã sao lưu model vào Drive/bge_model (lần sau trỏ thẳng vào đó).')

# %% CELL 1 — Imports + nạp model TỪ ĐĨA + helper (chạy sau CELL A)
import os
# Cho DATASET (CELL 2/3/4) cũng đi qua mirror. Đặt TRƯỚC khi import HF.
os.environ['HF_HUB_DISABLE_XET'] = '1'

import json, random, re, time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

MODEL_DIR = '/content/drive/MyDrive/bge_model'   # nạp model từ Drive (đã sao lưu ở CELL A)
PQ_M, PQ_NBITS = 256, 8
GT_TOPK = 10
SEED = 20235956

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)
_model = SentenceTransformer(MODEL_DIR, device=DEVICE)   # <-- đọc từ đĩa
print('Model nạp từ đĩa OK.')

def encode_norm(texts, bs=64):
    emb = _model.encode(list(texts), batch_size=bs, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return np.ascontiguousarray(emb, dtype=np.float32)

def robust_load(*args, **kw):
    from datasets import load_dataset
    for attempt in range(4):
        try:
            return load_dataset(*args, **kw)
        except Exception as e:
            print(f'  load thử {attempt+1} lỗi: {str(e)[:110]}... thử lại sau 5s')
            time.sleep(5)
    raise RuntimeError('load_dataset thất bại sau 4 lần.')

def build_and_save(name, corpus_texts, query_texts, query_source_idx=None):
    t0 = time.time()
    print('\n' + '=' * 70)
    print(f'[{name}] corpus={len(corpus_texts):,}  queries={len(query_texts):,}')
    print('=' * 70)

    print(f'[{name}] Embedding corpus...')
    Ec = encode_norm(corpus_texts)
    print(f'[{name}] Embedding queries...')
    Eq = encode_norm(query_texts)
    assert Ec.shape[1] == 1024 and Eq.shape[1] == 1024

    index = faiss.IndexFlatIP(1024); index.add(Ec)
    D, I = index.search(Eq, GT_TOPK)
    gt = []
    for i in range(len(query_texts)):
        results = [{
            'rank': r + 1, 'index': int(I[i][r]),
            'cosine_similarity': float(D[i][r]),
            'code_snippet': str(corpus_texts[int(I[i][r])]).replace('\n', ' ')[:80],
        } for r in range(GT_TOPK)]
        item = {'query_id': i + 1, 'query_text': str(query_texts[i]),
                'top_5_results': results[:5], 'top_10_results': results,
                'margin_top1_top10': float(D[i][0] - D[i][-1])}
        if query_source_idx is not None:
            src = int(query_source_idx[i])
            item['source_corpus_index'] = src
            item['source_in_gt10'] = bool(src in set(int(x) for x in I[i]))
        gt.append(item)

    print(f'[{name}] Train PQ...')
    pq = faiss.ProductQuantizer(1024, PQ_M, PQ_NBITS)
    pq.train(Ec)
    codes = pq.compute_codes(Ec)
    codebook = faiss.vector_to_array(pq.centroids).reshape(PQ_M, 1 << PQ_NBITS, 1024 // PQ_M)
    codebook = np.ascontiguousarray(codebook, dtype=np.float32)

    p = lambda s: os.path.join(OUT_DIR, f'{name}_{s}')
    np.save(p('corpus_embeddings.npy'), Ec)
    np.save(p('query_embeddings.npy'), Eq)
    np.save(p('pq_codebook.npy'), codebook)
    np.save(p('pq_codes.npy'), codes)
    with open(p('ground_truth.json'), 'w', encoding='utf-8') as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)
    with open(p('corpus_texts.json'), 'w', encoding='utf-8') as f:
        json.dump([str(x) for x in corpus_texts], f, ensure_ascii=False)

    angs = []
    for i in range(len(query_texts)):
        idxs = [r['index'] for r in gt[i]['top_5_results']]
        cos = Ec[idxs] @ Eq[i]
        angs.extend(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    angs = np.array(angs)
    print(f'\n[{name}] >>> GÓC query->GT5: mean={angs.mean():.1f}°  '
          f'median={np.median(angs):.1f}°  p90={np.percentile(angs,90):.1f}°')
    p_bit = 1 - angs / 180.0
    for c in (12, 14, 16):
        pc = (p_bit ** c).mean()
        print(f'[{name}]     prefix c={c}: p_bảng~{pc:.3f}  ->  Hit@5(L=5) dự đoán ~ {1-(1-pc)**5:.3f}')
    if query_source_idx is not None:
        rate = np.mean([g['source_in_gt10'] for g in gt])
        print(f'[{name}]     (hàm nguồn của query nằm trong GT-top10: {rate*100:.1f}%)')

    print(f'\n[{name}] Xong trong {time.time()-t0:.0f}s. Đã lưu 6 file.')
    return {'name': name, 'corpus': len(corpus_texts), 'queries': len(query_texts),
            'angle_mean': float(angs.mean()), 'dim': 1024}

MANIFEST = []
print('CELL 1 xong.')

# %% CELL 2 — BỘ CODE (CodeSearchNet Python) — query = docstring
import re, random
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

N_CORPUS_CODE, N_QUERY_CODE = 20000, 500
random.seed(SEED)

print('[code] Tìm file train parquet (qua mirror)...')
REPO = 'code-search-net/code_search_net'
files = list_repo_files(REPO, repo_type='dataset')
train_files = [f for f in files
               if f.endswith('.parquet') and 'train' in f.lower() and 'python' in f.lower()]
print('[code] Ứng viên:', train_files)
assert train_files, 'Không thấy train python. Parquet: ' + str([f for f in files if f.endswith(".parquet")])
fp = hf_hub_download(repo_id=REPO, filename=train_files[0], repo_type='dataset')
ds = pd.read_parquet(fp).to_dict('records')
print(f'[code] Đã tải {len(ds):,} hàm.')

def strip_docstring(code_str, doc_str):
    if doc_str and doc_str.strip() and doc_str.strip() in code_str:
        return code_str.replace(doc_str.strip(), ' ', 1), True
    stripped = re.sub(r'("""|\'\'\')(.*?)(\1)', ' ', code_str, count=1, flags=re.DOTALL)
    return stripped, (stripped != code_str)

def first_sentence(doc):
    doc = re.sub(r'\s+', ' ', str(doc)).strip()
    m = re.split(r'(?<=[.!?])\s', doc)
    return (m[0] if m else doc)[:200]

corpus_texts, corpus_docs, seen, n_ok = [], [], set(), 0
for row in ds:
    code = row.get('func_code_string') or ''
    doc  = row.get('func_documentation_string') or ''
    if len(doc.split()) < 5:
        continue
    code_wo, ok = strip_docstring(code, doc)
    code_wo = re.sub(r'\s+', ' ', code_wo).strip()
    if len(code_wo) < 40 or code_wo in seen:
        continue
    seen.add(code_wo); n_ok += int(ok)
    corpus_texts.append(code_wo); corpus_docs.append(doc)
    if len(corpus_texts) >= N_CORPUS_CODE:
        break
print(f'[code] Corpus: {len(corpus_texts):,} | gỡ docstring OK: {n_ok:,}')

cand = [i for i in range(len(corpus_texts)) if len(first_sentence(corpus_docs[i]).split()) >= 4]
q_idx = random.sample(cand, N_QUERY_CODE)
query_texts = [first_sentence(corpus_docs[i]) for i in q_idx]
leak = sum(1 for q in query_texts if q in set(corpus_texts))
print(f'[code] Query trùng nguyên văn corpus (phải = 0): {leak}')
print('[code] 3 query mẫu:', [q[:60] for q in query_texts[:3]])

MANIFEST.append(build_and_save('code', corpus_texts, query_texts, query_source_idx=q_idx))

# %% CELL 3 — BỘ SCIFACT (corpus=abstract, query=claim)
import random
random.seed(SEED)
print('[scifact] Tải BEIR/scifact (qua mirror)...')
corpus_ds = robust_load('BeIR/scifact', 'corpus', split='corpus')
query_ds  = robust_load('BeIR/scifact', 'queries', split='queries')

sci_corpus = []
for row in corpus_ds:
    t = (row.get('title') or '').strip(); x = (row.get('text') or '').strip()
    sci_corpus.append((t + '. ' + x).strip() if t else x)
print(f'[scifact] Corpus: {len(sci_corpus):,} (kỳ vọng ~5183)')

claims = [str(r.get('text') or '').strip() for r in query_ds]
claims = [c for c in claims if len(c.split()) >= 3]
random.shuffle(claims)
sci_query = claims[:500]
leak = sum(1 for q in sci_query if q in set(sci_corpus))
print(f'[scifact] Claim trùng corpus (nên = 0): {leak}')
print('[scifact] 3 claim mẫu:', [q[:60] for q in sci_query[:3]])

MANIFEST.append(build_and_save('scifact', sci_corpus, sci_query))

# %% CELL 4 — BỘ SQUAD/RAG (KHÔNG có trong paper; hỏi thầy trước khi dùng)
import random
random.seed(42)
print('[squad] Tải SQuAD (qua mirror)...')
sq = robust_load('rajpurkar/squad', split='train')
squad_corpus = sorted(set(sq['context']))
print(f'[squad] Context duy nhất: {len(squad_corpus):,} (kỳ vọng ~18891)')
idx = list(range(len(sq))); random.shuffle(idx)
squad_query = [sq[i]['question'] for i in idx[:500]]
print('[squad] 3 câu hỏi mẫu:', [q[:60] for q in squad_query[:3]])

MANIFEST.append(build_and_save('squad', squad_corpus, squad_query))

import json, os
import numpy as np
print('=' * 70 + '\nKIỂM TRA FILE\n' + '=' * 70)
for name in ['code', 'scifact', 'squad']:
    try:
        Ec = np.load(os.path.join(OUT_DIR, f'{name}_corpus_embeddings.npy'))
        Eq = np.load(os.path.join(OUT_DIR, f'{name}_query_embeddings.npy'))
        cb = np.load(os.path.join(OUT_DIR, f'{name}_pq_codebook.npy'))
        cd = np.load(os.path.join(OUT_DIR, f'{name}_pq_codes.npy'))
        gt = json.load(open(os.path.join(OUT_DIR, f'{name}_ground_truth.json')))
        print(f'{name:8s} | corpus {Ec.shape} | query {Eq.shape} | '
              f'codebook {cb.shape} | codes {cd.shape} {cd.dtype} | GT {len(gt)}')
        assert cb.shape == (256, 256, 4) and cd.shape[1] == 256 and cd.dtype == np.uint8
    except Exception as e:
        print(f'{name}: CHƯA CÓ / LỖI ({e})')
with open(os.path.join(OUT_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
    json.dump(MANIFEST, f, ensure_ascii=False, indent=2)
print('\nGóc trung bình mỗi bộ:')
for m in MANIFEST:
    print(f"  {m['name']:8s} góc={m['angle_mean']:.1f}°  corpus={m['corpus']:,}")
print('\n>>> GỬI THẦY bảng GÓC + Hit@5 dự đoán ở CELL 2/3 (checkpoint 1) trước khi chạy 5 seed.')