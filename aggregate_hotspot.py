#!/usr/bin/env python3
"""
GỘP THỐNG KÊ HOTSPOT QUA NHIỀU SEED — giải quyết mục 2.4 của bản rà soát.

VẤN ĐỀ: Bảng "ba góc nhìn về tải" đang dùng số single-seed:
    max/mean records = 220x, Gini work = 0,92, max/mean work = 838x
Seed 20235956 tình cờ cho tail XẤU NHẤT trong 10 seed. Bài đang trích đúng seed
tệ nhất mà không khai. Luận điểm hotspot vẫn đứng (Gini luôn 0,72-0,84), nên sửa
lại không mất gì mà bỏ được chỗ bị bắt lỗi cherry-pick.

DỮ LIỆU ĐÃ CÓ: main_simulation_v2.py đã log sẵn metadata_gini, rpc_gini,
work_gini, work_max, work_mean, rpc_p99, rpc_max. Chỉ cần gộp.

    python3 aggregate_hotspot.py
    python3 aggregate_hotspot.py --dataset scifact
"""
import argparse
import glob
import json
import statistics as st
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None, help='lọc theo corpus (mặc định: tất cả)')
    ap.add_argument('--nq', type=int, default=500)
    a = ap.parse_args()

    # cấu hình chốt: L=5 K=20 r=1 T=8 m512, query ĐỀU (zipf=0), không node loss
    groups = defaultdict(list)
    for f in glob.glob('result_*_L5_K20_MA1_T8_m512_s*_nq*.json'):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('n_query', 0) < a.nq or d.get('node_loss', 0) > 0:
            continue
        if d.get('zipf', 0) > 0:
            continue
        mode = d.get('routing_mode') or ('random_slots' if d.get('random_routing') else 'semantic')
        if mode != 'semantic' or not d.get('use_pq', True):
            continue
        ds = d['dataset']
        if a.dataset and ds != a.dataset:
            continue
        groups[(ds, d.get('nodes', 10000))].append(d)

    if not groups:
        print('Không thấy file phù hợp. Cần result_*_L5_K20_MA1_T8_m512_s*_nq500.json')
        print('(chạy nhóm A của run_paper.sh hoặc run_baselines_muc1.sh trước)')
        return

    # Đọc từ manifest nếu có, để code50k/code100k không bị gán nhầm 20.000
    CORPUS = {'code': 20000, 'scifact': 5183, 'squad': 18891,
              'code50k': 50000, 'code100k': 100000}
    try:
        for e in json.load(open('data/manifest.json')):
            CORPUS[e['name']] = e['corpus']
    except Exception:
        pass

    for (ds, N), runs in sorted(groups.items()):
        n_docs = CORPUS.get(ds, 20000)
        print('=' * 96)
        print(f'{ds} — N={N:,}, corpus {n_docs:,}, {len(runs)} seed')
        print('=' * 96)

        def agg(key, scale=1.0):
            v = [r[key] * scale for r in runs if key in r]
            if not v:
                return None
            return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, min(v), max(v))

        rows = [
            ('Gini lưu trữ (records)', 'metadata_gini', 1.0, '{:.3f}'),
            ('Gini tải RPC',            'rpc_gini',      1.0, '{:.3f}'),
            ('Gini công tính toán',     'work_gini',     1.0, '{:.3f}'),
        ]
        print(f"{'đại lượng':26s} {'mean':>9s} {'std':>8s} {'min':>9s} {'max':>9s}")
        print('-' * 96)
        for lbl, key, sc, fmt in rows:
            r = agg(key, sc)
            if r is None:
                print(f'{lbl:26s} {"(chưa log)":>9s}')
                continue
            m, sd, lo, hi = r
            print(f'{lbl:26s} {fmt.format(m):>9s} {fmt.format(sd):>8s} '
                  f'{fmt.format(lo):>9s} {fmt.format(hi):>9s}')

        # tỉ số max/mean — đại lượng bài đang trích single-seed
        print()
        print(f"{'tỉ số max/mean':26s} {'mean':>9s} {'std':>8s} {'min':>9s} {'max':>9s}")
        print('-' * 96)
        for lbl, num, den in [('records', 'metadata_max', 'metadata_mean_per_node'),
                              ('RPC', 'rpc_max', 'rpc_mean'),
                              ('công tính toán', 'work_max', 'work_mean')]:
            v = [r[num] / r[den] for r in runs
                 if r.get(num) is not None and r.get(den, 0) > 0]
            if not v:
                # metadata_max có thể chưa được log; suy từ P99 nếu có
                print(f'{lbl:26s} {"(chưa log " + num + ")":>9s}')
                continue
            m = st.mean(v); sd = st.stdev(v) if len(v) > 1 else 0.0
            print(f'{lbl:26s} {m:>9.0f} {sd:>8.0f} {min(v):>9.0f} {max(v):>9.0f}')

        # seed nào cho tail xấu nhất
        wm = [(r.get('work_max', 0) / max(r.get('work_mean', 1), 1e-9), r['seed'])
              for r in runs if r.get('work_max')]
        if wm:
            wm.sort(reverse=True)
            print()
            print(f'  Seed cho tail công tính toán XẤU NHẤT: {wm[0][1]} '
                  f'({wm[0][0]:.0f}x mean)')
            print(f'  Seed cho tail NHẸ NHẤT              : {wm[-1][1]} '
                  f'({wm[-1][0]:.0f}x mean)')
            if len(wm) > 1 and wm[0][0] > 1.5 * st.median([x[0] for x in wm]):
                print(f'  *** Tail của seed xấu nhất gấp '
                      f'{wm[0][0]/st.median([x[0] for x in wm]):.1f} lần trung vị.')
                print(f'      Báo cáo single-seed sẽ bị xem là cherry-pick. Dùng mean ± std. ***')
        print()

    print('=' * 96)
    print('DÙNG SỐ NÀO CHO BÀI')
    print('=' * 96)
    print('  Thay các số single-seed bằng mean ± std, và ghi số seed vào caption.')
    print('  Luận điểm hotspot không dựa vào giá trị tail mà dựa vào việc Gini LUÔN')
    print('  cao ở mọi seed — nên báo mean ± std vẫn giữ nguyên lập luận, mà bỏ được')
    print('  chỗ reviewer bắt lỗi.')
    print()
    print('  Nếu cột nào in "(chưa log)": chạy lại nhóm A của run_paper.sh với bản')
    print('  main_simulation_v2.py hiện tại, vì các field work_* thêm sau.')


if __name__ == '__main__':
    main()