# =====================================================================
# 02_extend_weakscaling.py — extends the code corpus to 50k and 100k
#
# ORIGIN: Google Colab notebook, exported verbatim. Colab-only calls apply,
# same as 01_build_corpora.py.
#
# WHY SEPARATE: the weak-scaling experiment needs corpora of 50,000 and
# 100,000 objects. This script appends new CodeSearchNet functions to the
# existing 20,000 and asserts byte-for-byte that the first 20,000 embeddings
# are unchanged, so the smaller corpus remains a strict prefix of the larger.
#
# WHAT IT PRODUCES, per scale {code50k, code100k}:
#   same six files as 01_build_corpora.py, plus pq_codes_m512 / codebook_m512
#
# The product quantizer is RETRAINED for each scale rather than reused from
# the 20k corpus. Ground truth is recomputed for each scale, since the top-10
# neighbours of a query change when the corpus grows.
#
# Deduplication is against all previously admitted bodies, including the
# original 20,000, so no function appears twice across scales.
# =====================================================================


# =====================================================================
# CELL 1 — Cài đặt, mount Drive, kiểm tra điều kiện
# =====================================================================
!pip -q install faiss-cpu sentence-transformers "datasets>=2.19" 2>/dev/null
!pip uninstall -y hf_xet hf-xet 2>/dev/null

import os
os.environ['HF_HUB_DISABLE_XET'] = '1'      # đặt TRƯỚC khi import HF

from google.colab import drive
drive.mount('/content/drive')

import json, re, time, random, gc
import numpy as np
import pandas as pd
import faiss
import torch

OUT_DIR   = '/content/drive/MyDrive/v_engram_final_data'   # data cũ, KHÔNG đổi
MODEL_DIR = '/content/drive/MyDrive/bge_model'             # model đã sao lưu
SEED      = 20235956
TARGET    = 100_000        # corpus đích. Giảm xuống nếu không lấy đủ mẫu.
GT_TOPK   = 10
PQ_NBITS  = 8
NEW_M     = 512            # PQ m512, đúng cấu hình paper

random.seed(SEED)

print('=' * 70)
print('KIỂM TRA ĐIỀU KIỆN')
print('=' * 70)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available()
      else '*** KHÔNG CÓ GPU — đổi Runtime type sang T4 rồi chạy lại ***')

assert os.path.isdir(OUT_DIR), f'Không thấy {OUT_DIR}'
need = ['code_corpus_embeddings.npy', 'code_query_embeddings.npy',
        'code_corpus_texts.json', 'code_ground_truth.json']
for f in need:
    ok = os.path.exists(f'{OUT_DIR}/{f}')
    print(f'  {"✓" if ok else "✗"} {f}')
    assert ok, f'Thiếu {OUT_DIR}/{f}'

if os.path.isdir(MODEL_DIR) and os.path.exists(f'{MODEL_DIR}/model.safetensors'):
    sz = os.path.getsize(f'{MODEL_DIR}/model.safetensors') / 1e9
    print(f'  ✓ model có sẵn trong Drive ({sz:.2f} GB)')
    assert sz > 1.0, 'model.safetensors quá nhỏ — tải lại ở CELL 1b'
else:
    print('  ✗ CHƯA có model trong Drive — chạy CELL 1b để tải qua mirror')
print('\nCELL 1 xong.')

# =====================================================================
# CELL 2 — Nạp model và corpus 20K hiện có
# =====================================================================
from sentence_transformers import SentenceTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_model = SentenceTransformer(MODEL_DIR, device=DEVICE)
print(f'Model nạp từ đĩa OK (device={DEVICE})')

def encode_norm(texts, bs=64):
    emb = _model.encode(list(texts), batch_size=bs, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return np.ascontiguousarray(emb, dtype=np.float32)

# --- nạp lại 20K cũ ---
E_old     = np.load(f'{OUT_DIR}/code_corpus_embeddings.npy')
Eq        = np.load(f'{OUT_DIR}/code_query_embeddings.npy')
texts_old = json.load(open(f'{OUT_DIR}/code_corpus_texts.json'))
gt_old    = json.load(open(f'{OUT_DIR}/code_ground_truth.json'))

print(f'\ncorpus cũ : {E_old.shape}')
print(f'query     : {Eq.shape}')
print(f'texts cũ  : {len(texts_old):,}')
print(f'norm corpus: {np.linalg.norm(E_old, axis=1).mean():.4f} (phải ~1.0)')

# source index của 500 query — giữ nguyên quan hệ query ↔ doc gốc
q_src = [g.get('source_corpus_index', -1) for g in gt_old]
print(f'query có source index: {sum(1 for x in q_src if x >= 0)}/{len(q_src)}')
assert max(q_src) < len(texts_old), 'source index vượt corpus cũ'

seen = set(texts_old)
print('\nCELL 2 xong.')

# =====================================================================
# CELL 3 — Lấy thêm hàm từ CodeSearchNet cho đủ 100K
#
# Tách docstring GIỐNG HỆT notebook cũ để tránh self-retrieval:
#     query  = câu đầu của docstring
#     corpus = thân hàm ĐÃ XOÁ docstring
# =====================================================================
from huggingface_hub import hf_hub_download, list_repo_files

def strip_docstring(code_str, doc_str):
    """Y nguyên bản trong notebook cũ."""
    if doc_str and doc_str.strip() and doc_str.strip() in code_str:
        return code_str.replace(doc_str.strip(), ' ', 1), True
    stripped = re.sub(r'("""|\'\'\')(.*?)(\1)', ' ', code_str, count=1, flags=re.DOTALL)
    return stripped, (stripped != code_str)

REPO = 'code-search-net/code_search_net'
files_all = list_repo_files(REPO, repo_type='dataset')
train_files = sorted(f for f in files_all
                     if f.endswith('.parquet') and 'train' in f.lower()
                     and 'python' in f.lower())
print(f'Có {len(train_files)} file parquet train/python:')
for f in train_files:
    print('   ', f)
assert train_files, 'Không thấy parquet train python'

new_texts, new_docs = [], []
n_ok = n_dup = n_short = 0
for fi, fname in enumerate(train_files):
    if len(texts_old) + len(new_texts) >= TARGET:
        break
    print(f'\n[{fi+1}/{len(train_files)}] tải {fname} ...')
    fp = hf_hub_download(repo_id=REPO, filename=fname, repo_type='dataset')
    ds = pd.read_parquet(fp).to_dict('records')
    print(f'    {len(ds):,} hàm trong file')
    for row in ds:
        if len(texts_old) + len(new_texts) >= TARGET:
            break
        code = row.get('func_code_string') or ''
        doc  = row.get('func_documentation_string') or ''
        if len(doc.split()) < 5:
            n_short += 1; continue
        code_wo, ok = strip_docstring(code, doc)
        code_wo = re.sub(r'\s+', ' ', code_wo).strip()
        if len(code_wo) < 40:
            n_short += 1; continue
        if code_wo in seen:            # không trùng 20K cũ, cũng không tự trùng
            n_dup += 1; continue
        seen.add(code_wo)
        n_ok += int(ok)
        new_texts.append(code_wo); new_docs.append(doc)
    print(f'    tổng đã gom: {len(texts_old)+len(new_texts):,}/{TARGET:,}')
    del ds; gc.collect()

TOTAL = len(texts_old) + len(new_texts)
print(f'\n{"="*70}')
print(f'Lấy thêm {len(new_texts):,} hàm  ->  tổng {TOTAL:,}')
print(f'  gỡ docstring thành công : {n_ok:,}')
print(f'  bỏ vì trùng             : {n_dup:,}')
print(f'  bỏ vì quá ngắn          : {n_short:,}')
if TOTAL < TARGET:
    print(f'\n*** CHỈ ĐƯỢC {TOTAL:,}, KHÔNG ĐỦ {TARGET:,} ***')
    print('    Bộ code100k sẽ có đúng số này thay vì 100.000 — vẫn dùng được,')
    print('    chỉ cần đổi số node ở run_weakscaling.sh cho khớp tỉ lệ 2 doc/node.')
print('\nCELL 3 xong.')

# =====================================================================
# CELL 4 — Embed số doc mới (CHẬM NHẤT: 30-45 phút trên T4)
# =====================================================================
t0 = time.time()
print(f'Embedding {len(new_texts):,} doc mới ...')
E_new = encode_norm(new_texts, bs=64)
print(f'  {E_new.shape} trong {time.time()-t0:.0f}s')
print(f'  norm trung bình: {np.linalg.norm(E_new, axis=1).mean():.4f} (phải ~1.0)')

# GHÉP: 20K cũ TRƯỚC, mới SAU
E_all = np.ascontiguousarray(np.concatenate([E_old, E_new], axis=0), dtype=np.float32)
texts_all = list(texts_old) + new_texts
print(f'\ncorpus tổng: {E_all.shape}')
assert np.array_equal(E_all[:len(E_old)], E_old), '20K đầu KHÔNG khớp — DỪNG LẠI'
print('✓ Xác nhận 20.000 dòng đầu khớp chính xác corpus cũ')

del E_new; gc.collect()
print('\nCELL 4 xong.')

# =====================================================================
# CELL 5 — Sinh bộ dữ liệu cho từng quy mô
# =====================================================================
def build_scale(S, name):
    t0 = time.time()
    print('\n' + '=' * 70)
    print(f'[{name}] {S:,} doc')
    print('=' * 70)
    E = np.ascontiguousarray(E_all[:S])
    texts = texts_all[:S]

    print(f'[{name}] ground truth (FAISS exact top-{GT_TOPK}) ...')
    index = faiss.IndexFlatIP(1024); index.add(E)
    D, I = index.search(Eq, GT_TOPK)
    gt = []
    for i in range(len(Eq)):
        results = [{'rank': r + 1, 'index': int(I[i][r]),
                    'cosine_similarity': float(D[i][r]),
                    'code_snippet': str(texts[int(I[i][r])]).replace('\n', ' ')[:80]}
                   for r in range(GT_TOPK)]
        gt.append({'query_id': gt_old[i]['query_id'],
                   'query_text': gt_old[i]['query_text'],
                   'top_5_results': results[:5], 'top_10_results': results,
                   'margin_top1_top10': float(D[i][0] - D[i][-1]),
                   'source_corpus_index': int(q_src[i]),
                   'source_in_gt10': bool(int(q_src[i]) in set(int(x) for x in I[i]))})

    print(f'[{name}] train PQ m={NEW_M} ...')
    d_sub = 1024 // NEW_M
    pq = faiss.ProductQuantizer(1024, NEW_M, PQ_NBITS)
    pq.train(E)
    codes = pq.compute_codes(E)
    cb = faiss.vector_to_array(pq.centroids).reshape(NEW_M, 1 << PQ_NBITS, d_sub)
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    mse = float(((pq.decode(codes) - E) ** 2).sum(axis=1).mean())
    print(f'    codes{codes.shape} codebook{cb.shape} MSE={mse:.4f}')

    # --- chẩn đoán chất lượng, giống CELL 2 notebook cũ ---
    angs = []
    for i in range(len(Eq)):
        idxs = [r['index'] for r in gt[i]['top_5_results']]
        angs.extend(np.degrees(np.arccos(np.clip(E[idxs] @ Eq[i], -1, 1))))
    angs = np.array(angs)
    src_rate = float(np.mean([g['source_in_gt10'] for g in gt]))
    print(f'[{name}] >>> GÓC query→GT5: mean={angs.mean():.1f}° '
          f'median={np.median(angs):.1f}° p90={np.percentile(angs,90):.1f}°')
    print(f'[{name}]     doc gốc trong GT-top10: {src_rate*100:.1f}%')
    p_bit = 1 - angs / 180.0
    for c in (12, 14, 16):
        pc = (p_bit ** c).mean()
        print(f'[{name}]     prefix c={c}: p_bảng~{pc:.3f} -> Hit@5(L=5) dự đoán ~{1-(1-pc)**5:.3f}')
    if angs.mean() < 35:
        print(f'[{name}] *** CẢNH BÁO: góc {angs.mean():.1f}° quá nhỏ, nghi self-retrieval ***')

    # --- lưu vào ĐÚNG thư mục data cũ ---
    p = lambda sfx: os.path.join(OUT_DIR, f'{name}_{sfx}')
    np.save(p('corpus_embeddings.npy'), E)
    np.save(p('query_embeddings.npy'), Eq)
    np.save(p('pq_codes_m512.npy'), codes)
    np.save(p('pq_codebook_m512.npy'), cb)
    # simulator tìm *_pq_codes.npy khi KHÔNG truyền --pq-variant.
    # Lưu cùng nội dung m512. node.py đọc m và d_sub từ codebook.shape nên đúng.
    np.save(p('pq_codes.npy'), codes)
    np.save(p('pq_codebook.npy'), cb)
    json.dump([str(x) for x in texts], open(p('corpus_texts.json'), 'w'),
              ensure_ascii=False)
    json.dump(gt, open(p('ground_truth.json'), 'w'), ensure_ascii=False, indent=2)
    print(f'[{name}] ✓ xong {time.time()-t0:.0f}s')
    return {'name': name, 'corpus': S, 'queries': len(Eq),
            'angle_mean': float(angs.mean()), 'dim': 1024}

entries = [build_scale(50_000, 'code50k'),
           build_scale(min(100_000, len(texts_all)), 'code100k')]
print('\nCELL 5 xong.')

# =====================================================================
# CELL 6 — Cập nhật manifest và kiểm tra
# =====================================================================
mf_path = f'{OUT_DIR}/manifest.json'
mf = json.load(open(mf_path)) if os.path.exists(mf_path) else []
for e in entries:
    mf = [x for x in mf if x['name'] != e['name']]
    mf.append(e)
json.dump(mf, open(mf_path, 'w'), ensure_ascii=False, indent=2)

print('MANIFEST:')
for e in mf:
    print(f"  {e['name']:10s} corpus={e['corpus']:>7,} góc={e['angle_mean']:.1f}°")

print('\nKIỂM TRA FILE:')
for nm in ['code', 'code50k', 'code100k']:
    try:
        e = np.load(f'{OUT_DIR}/{nm}_corpus_embeddings.npy', mmap_mode='r')
        c = np.load(f'{OUT_DIR}/{nm}_pq_codes_m512.npy', mmap_mode='r')
        g = json.load(open(f'{OUT_DIR}/{nm}_ground_truth.json'))
        print(f'  {nm:10s} emb={e.shape} codes={c.shape} gt={len(g)}')
    except Exception as ex:
        print(f'  {nm:10s} LỖI: {ex}')

print('\nXÁC NHẬN 20K ĐẦU KHỚP CORPUS CŨ:')
a = np.load(f'{OUT_DIR}/code_corpus_embeddings.npy')
b = np.asarray(np.load(f'{OUT_DIR}/code100k_corpus_embeddings.npy',
                       mmap_mode='r')[:20_000])
print('  khớp byte-for-byte:', np.array_equal(a, b))
print('\nCELL 6 xong.')

# =====================================================================
# CELL 7 — Nén và tải về máy
#
# KHÔNG commit lên git: code100k_corpus_embeddings.npy = 410 MB,
# GitHub chặn cứng 100 MB mỗi file.
# =====================================================================
!cd {OUT_DIR} && tar czf /content/weakscaling.tar.gz code50k_* code100k_* manifest.json
!ls -lh /content/weakscaling.tar.gz

from google.colab import files
files.download('/content/weakscaling.tar.gz')

print('''
======================================================================
TẢI XONG. BƯỚC TIẾP THEO — xem file huong-dan-weakscaling.md
======================================================================
Tóm tắt:
  1. Trên MÁY BẠN:
       scp weakscaling.tar.gz research@103.67.203.71:~/V_ENGRAM_SIMULATION/
  2. Trên SERVER:
       cd ~/V_ENGRAM_SIMULATION
       tar xzf weakscaling.tar.gz -C data/
       ls -la data/code100k_*
       rm weakscaling.tar.gz
  3. Lấy code mới (2 file) rồi chạy:
       git pull
       tmux new -s weak
       source venv/bin/activate
       bash run_weakscaling.sh 2>&1 | tee weakscaling.log
======================================================================''')