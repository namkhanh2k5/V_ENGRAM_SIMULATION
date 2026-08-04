#!/usr/bin/env python3
"""
CHURN ENGINE — node ra vào TRONG LÚC hệ đang phục vụ.

KHÁC main_churn_test.py: bản đó tắt một loạt node rồi mới query (static failure).
Bản này mô phỏng THỜI GIAN — node rời mạng khi hết session, node mới tham gia với
ID mới và bộ nhớ rỗng, metadata trên node đã rời MẤT theo, và cơ chế sửa chữa
(nếu bật) tái neo record về node còn sống.

BA CẤU HÌNH CẦN SO (thí nghiệm quyết định của paper):
    r=1  + repair    : lối V-Engram đề xuất — nhân bản thấp, sửa nhanh
    r=20 không repair: lối IPFS — nhân bản cao, sửa chậm
    r=1  không repair: đối chứng, cho biết sửa chữa đáng giá bao nhiêu

MÔ HÌNH SESSION:
    weibull (mặc định) — khớp đo thật của Stutzbach & Rejaie, IMC 2006:
        "session lengths are not exponential", phân bố Weibull/lognormal,
        đa số node ổn định còn thiểu số ra vào rất nhanh.
        shape k=0.5 cho đuôi nặng; median = scale * (ln2)^(1/k).
    exponential — quy ước mô phỏng của Li et al. (IPTPS'04) và Bamboo
        (USENIX'04). Giữ để so được với văn liệu, dù biết là không khớp thực tế.

MỐC THAM CHIẾU TỪ HỆ THẬT:
    IPFS  : 87,6% session dưới 8 giờ (467.134 session đo được, SIGCOMM 2022)
            r=20 replica, republish mỗi 22 giờ
    Li    : mũ, trung bình 1 giờ, 1.024 node
    Bamboo: median 1,4–47 phút (quét stress)

CHẠY:
    python3 main_churn_engine.py --dataset code --nodes 10000 \\
        --median-session 60 --duration 180 --meta-anchors 1 --repair-interval 30

    python3 main_churn_engine.py --dataset code --nodes 10000 \\
        --median-session 60 --duration 180 --meta-anchors 20 --repair-interval 0
"""
import argparse
import json
import os
import random
import time

import numpy as np

TTL_INF = float('inf')


# ---------------------------------------------------------------------------
# Mạng có churn
# ---------------------------------------------------------------------------
class ChurnNetwork:
    """Mạng Kademlia rút gọn, node ra vào theo thời gian.

    Giữ SỐ NODE không đổi: mỗi node rời đi thì một node mới tham gia. Đây là điều
    các nghiên cứu đo được — churn cao nhưng tổng số peer khá ổn định.
    """

    def __init__(self, num_nodes, seed, median_session, dist='weibull', shape=0.5):
        self.rnd = random.Random(seed)
        self.rng = np.random.RandomState(seed)
        self.num_nodes = num_nodes
        self.node_ids = np.array([self.rnd.getrandbits(63) for _ in range(num_nodes)],
                                 dtype=np.int64)
        # node_idx -> {tag: thời điểm HẾT HẠN}. Không lưu code vì truy hồi chỉ
        # cần biết node NÀO giữ tag nào; code lấy từ mảng codes toàn cục.
        self.ram = [dict() for _ in range(num_nodes)]
        self.now = 0.0
        self.dist = dist
        self.shape = shape
        self.median_session = float(median_session)
        # Weibull: median = scale * (ln2)^(1/k)  =>  scale = median / (ln2)^(1/k)
        self.scale = median_session / (np.log(2) ** (1.0 / shape)) if dist == 'weibull' \
            else median_session / np.log(2)                 # mũ: median = mean*ln2
        # thời gian còn lại của phiên hiện tại, mỗi node một giá trị
        self.remaining = self._draw_sessions(num_nodes)
        # thống kê
        self.total_departures = 0
        self.repair_msgs = 0

    def _draw_sessions(self, n):
        if self.dist == 'weibull':
            return self.scale * self.rng.weibull(self.shape, size=n)
        return self.rng.exponential(self.scale, size=n)

    def knn(self, key, k):
        d = np.bitwise_xor(self.node_ids, key)
        k = min(k, self.num_nodes)
        idx = np.argpartition(d, k - 1)[:k]
        return idx

    @staticmethod
    def key63(vector, proj):
        bits = (np.asarray(vector).flatten() @ proj > 0).astype(int)[:63]
        return np.int64(int(''.join(map(str, bits)), 2))

    @staticmethod
    def probe_keys63(vector, proj, T, c=16):
        pr = np.asarray(vector).flatten() @ proj
        bits = (pr > 0).astype(int)[:63]
        base = int(''.join(map(str, bits)), 2)
        keys = [np.int64(base)]
        if T <= 1:
            return keys
        c = min(c, 63)
        weak = np.argsort(np.abs(pr[:c]))
        for j in weak[:T - 1]:
            keys.append(np.int64(base ^ (1 << (62 - int(j)))))
        return keys

    def purge_expired(self):
        """Xoá record đã hết hạn.

        KHÔNG CÓ BƯỚC NÀY thì sửa chữa chỉ THÊM bản sao mà không bao giờ bớt:
        mỗi vòng sửa ghi vào node gần nhất HIỆN TẠI, còn bản cũ trên node vẫn
        sống thì nằm lại mãi. Sau vài vòng, "r=1 + sửa" âm thầm thành r=5 hoặc
        hơn — đúng vùng mà Mục r* chứng minh là mất hết lợi thế. Đo mà không
        purge là tự cho mình điểm.

        IPFS làm đúng vậy: Provide Validity 48 giờ, republish mỗi 22 giờ.
        """
        n = 0
        for d in self.ram:
            if not d:
                continue
            dead = [t for t, exp in d.items() if exp <= self.now]
            for t in dead:
                del d[t]
            n += len(dead)
        return n

    def step(self, dt):
        """Tiến dt phút. Node hết phiên rời đi, node mới thay chỗ.

        Node mới có ID NGẪU NHIÊN MỚI và bộ nhớ RỖNG — nó không kế thừa gì từ
        node cũ. Đây là điểm khác cốt lõi so với mô hình tắt node tĩnh.
        """
        self.now += dt
        self.purge_expired()
        self.remaining -= dt
        dead = np.flatnonzero(self.remaining <= 0)
        if len(dead) == 0:
            return 0
        for i in dead:
            self.ram[i] = {}                                # metadata MẤT theo
            self.node_ids[i] = self.rnd.getrandbits(63)     # ID mới, vị trí mới
        self.remaining[dead] = self._draw_sessions(len(dead))
        self.total_departures += len(dead)
        return len(dead)


# ---------------------------------------------------------------------------
# Sửa chữa
# ---------------------------------------------------------------------------
def repair(net, anchors, keys_per_doc, r, ttl, mode='lazy'):
    """Tái neo record về node còn sống.

    lazy    — chỉ GHI LẠI doc thiếu anchor, nhưng vẫn GIA HẠN TTL cho doc khoẻ.
              Gia hạn là một message keep-alive, rẻ hơn ghi lại toàn bộ record.
    blanket — ghi lại TẤT CẢ, đúng lối IPFS (republish mỗi 22 giờ bất kể).

    BUG ĐÃ SỬA: bản trước, lazy BỎ QUA hẳn doc khoẻ, nên TTL của chúng không
    được gia hạn và hết hạn dù mọi anchor còn sống. Vòng đời thành: ghi ở t=0,
    bỏ qua ở t=1 và t=2 chu kỳ, HẾT HẠN ở t=2,2 chu kỳ, ghi lại ở t=3. Tức chết
    27% thời gian bất kể chu kỳ sửa dày hay thưa — đúng con số đo được ở
    rep=30 (72,3%) và rep=60 (77,2%). Nó cũng làm availability KHÔNG ĐƠN ĐIỆU
    theo chu kỳ sửa, điều bất khả về mặt vật lý.

    Không hệ thật nào hành xử vậy: gia hạn hạn dùng của một bản sao còn sống là
    việc phải làm, và IPFS republish tất cả chính vì thế.
    """
    n_repaired = 0
    msgs = 0
    exp = net.now + ttl
    for tag, cur in anchors.items():
        alive = [ni for ni in cur if tag in net.ram[ni]]
        if mode == 'lazy' and len(alive) >= len(keys_per_doc[tag]) * r:
            # Đủ anchor: không cần ghi lại record, nhưng PHẢI gia hạn TTL,
            # nếu không nó hết hạn và bản sao khoẻ bị xoá oan.
            for ni in alive:
                net.ram[ni][tag] = exp
                msgs += 1                 # keep-alive vẫn tốn một message
            continue
        new_set = []
        for skey in keys_per_doc[tag]:
            for ni in net.knn(skey, r):
                ni = int(ni)
                # ghi mới HOẶC gia hạn bản đã có — cả hai đều tốn 1 message
                net.ram[ni][tag] = exp
                msgs += 1
                new_set.append(ni)
        anchors[tag] = new_set
        n_repaired += 1
    net.repair_msgs += msgs
    return n_repaired, msgs


class AnchorTable(dict):
    """dict tag -> [node_idx], kèm payload để tái neo được."""

    def __init__(self):
        super().__init__()
        self.payload = {}


# ---------------------------------------------------------------------------
# Đo recall
# ---------------------------------------------------------------------------
def measure_recall(net, P, Qv, gt, codes, codebook, args, n_query, rnd_route=None):
    """Chạy n_query truy vấn, đo CẢ semantic LẪN random trên CÙNG trạng thái mạng.

    Đo cùng lúc là bắt buộc: chạy hai lần riêng thì trạng thái churn đã khác,
    và tỉ lệ sem/rand — đại lượng mà cả paper xoay quanh — sẽ không so được.
    """
    reach_sum = final_sum = rand_sum = 0.0
    touched_sum = 0
    m, ksub, dsub = codebook.shape
    for qi in range(n_query):
        item = gt[qi]
        gt5 = set(r['index'] for r in item['top_5_results'][:5])
        q = Qv[qi]

        touched = set()
        for proj in P:
            for qkey in net.probe_keys63(q, proj, args.multi_probe, args.probe_bits):
                touched.update(int(x) for x in net.knn(qkey, args.k_query))
        touched_sum += len(touched)

        # tầng 1: record có nằm trên node nào được ghé không
        reachable = set()
        for ni in touched:
            reachable.update(net.ram[ni].keys())
        reach_sum += len(gt5 & reachable) / 5.0

        # tầng 3: rerank ADC trên ứng viên gom được
        cand = sorted(reachable)
        if not cand:
            continue
        qsub = q.reshape(m, dsub)
        lut = np.einsum('md,mkd->mk', qsub, codebook)       # (m, 256)
        cand_codes = codes[np.array(cand)]                  # (n_cand, m)
        # với mỗi ứng viên c và mỗi subquantizer j: cộng lut[j, code[c,j]].
        # take_along_axis(lut (m,256), idx (m,n_cand), axis=1) -> (m,n_cand)
        score = np.take_along_axis(
            lut, cand_codes.T.astype(np.intp), axis=1).sum(axis=0)
        top = np.argsort(-score)[:5]
        got = set(cand[int(t)] for t in top)
        final_sum += len(gt5 & got) / 5.0

        # --- baseline ngẫu nhiên trên CÙNG trạng thái mạng ---
        # Chạm đúng số slot danh nghĩa L*K*T, chọn node ngẫu nhiên (oracle
        # uniform-node sampling — giả định client thấy được toàn bộ thành viên).
        if rnd_route is not None:
            n_touch = min(args.k_query * len(P) * args.multi_probe, net.num_nodes)
            rtouched = rnd_route.sample(range(net.num_nodes), n_touch)
            rreach = set()
            for ni in rtouched:
                rreach.update(net.ram[ni].keys())
            rand_sum += len(gt5 & rreach) / 5.0

    return (100 * reach_sum / n_query, 100 * final_sum / n_query,
            touched_sum / n_query, 100 * rand_sum / n_query)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='code')
    ap.add_argument('--nodes', type=int, default=10000)
    ap.add_argument('--num-tables', type=int, default=5)
    ap.add_argument('--k-query', type=int, default=20)
    ap.add_argument('--multi-probe', type=int, default=8)
    ap.add_argument('--probe-bits', type=int, default=16)
    ap.add_argument('--meta-anchors', type=int, default=1, metavar='R')
    ap.add_argument('--pq-variant', default='m512')
    ap.add_argument('--seed', type=int, default=20235956)
    ap.add_argument('--nq', type=int, default=200,
                    help='số query mỗi lần đo (ít hơn v2 vì đo nhiều lần)')

    ap.add_argument('--median-session', type=float, default=60.0, metavar='PHUT',
                    help='median session (phút). Mốc: IPFS ~8h, Li 1h, Bamboo 1.4-47ph')
    ap.add_argument('--session-dist', default='weibull',
                    choices=['weibull', 'exponential'],
                    help='weibull khớp đo thật; exponential theo quy ước mô phỏng')
    ap.add_argument('--weibull-shape', type=float, default=0.5,
                    help='k<1 = đuôi nặng, đa số ổn định + thiểu số ra vào nhanh')
    ap.add_argument('--duration', type=float, default=180.0, metavar='PHUT',
                    help='tổng thời gian mô phỏng')
    ap.add_argument('--epoch', type=float, default=0, metavar='PHUT',
                    help='bước thời gian (0 = median/12)')
    ap.add_argument('--warmup', type=float, default=0, metavar='PHUT',
                    help='chạy churn trước khi đo, cho hệ về trạng thái dừng '
                         '(0 = 1 lần median session)')
    ap.add_argument('--repair-interval', type=float, default=0, metavar='PHUT',
                    help='chu kỳ sửa chữa (0 = KHÔNG sửa)')
    ap.add_argument('--ttl', type=float, default=0, metavar='PHUT',
                    help='hạn dùng của record trên node (0 = 2.2x chu kỳ sửa, '
                         'theo tỉ lệ IPFS: TTL 48h / republish 22h). '
                         'KHÔNG sửa chữa thì record không hết hạn.')
    ap.add_argument('--repair-mode', default='lazy', choices=['lazy', 'blanket'],
                    help='lazy = chỉ sửa doc thiếu anchor; blanket = republish tất cả')
    ap.add_argument('--measure-every', type=int, default=0,
                    help='đo recall mỗi bao nhiêu epoch (0 = 6 lần đo đều nhau)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.epoch <= 0:
        args.epoch = args.median_session / 12.0
    if args.warmup <= 0:
        args.warmup = args.median_session
    # Thời lượng phải đủ cho VÀI chu kỳ sửa, nếu không hệ chưa vào trạng thái
    # dừng và số đo phản ánh giai đoạn quá độ. Chu kỳ sửa DÀI HƠN thời lượng thì
    # sửa chữa không chạy lần nào — điểm đo đó vô nghĩa.
    need = max(3 * args.median_session, 3 * args.repair_interval)
    if args.duration < need:
        print(f'[!] Thời lượng {args.duration:.0f}ph quá ngắn cho chu kỳ sửa '
              f'{args.repair_interval:.0f}ph. Nâng lên {need:.0f}ph '
              f'(= 3 lần chu kỳ dài nhất).')
        args.duration = need
    # warm-up cũng cần ít nhất một chu kỳ sửa
    if args.repair_interval > 0 and args.warmup < 2 * args.repair_interval:
        args.warmup = 2 * args.repair_interval
    n_epoch = max(1, int(round(args.duration / args.epoch)))
    if args.measure_every <= 0:
        args.measure_every = max(1, n_epoch // 6)
    if args.ttl <= 0:
        # tỉ lệ IPFS: Provide Validity 48h / republish 22h = 2.18
        args.ttl = 2.2 * args.repair_interval if args.repair_interval > 0 else float('inf')

    D = f'./data/{args.dataset}'
    print(f'[*] Nạp {args.dataset} ...')
    E = np.load(f'{D}_corpus_embeddings.npy')
    Qv = np.load(f'{D}_query_embeddings.npy')
    sfx = f'_{args.pq_variant}' if args.pq_variant else ''
    codes = np.load(f'{D}_pq_codes{sfx}.npy')
    codebook = np.load(f'{D}_pq_codebook{sfx}.npy')
    gt = json.load(open(f'{D}_ground_truth.json', encoding='utf-8'))
    N_DOCS = len(E)
    n_query = min(args.nq, len(gt))
    print(f'    corpus={N_DOCS:,} query={n_query} codebook={codebook.shape}')

    L, r = args.num_tables, args.meta_anchors
    rnd = np.random.RandomState(args.seed)
    P = [rnd.choice([-1.0, 0.0, 1.0], size=(1024, 63), p=[1/6, 2/3, 1/6])
         for _ in range(L)]

    print(f'[*] Mạng {args.nodes:,} node | session {args.session_dist} '
          f'median={args.median_session:.0f}ph')
    print(f'[*] TTL record: '
          f'{"vô hạn (không sửa)" if args.ttl == float("inf") else f"{args.ttl:.0f}ph"}')
    print(f'[*] r={r} | sửa chữa: '
          f'{"KHÔNG" if args.repair_interval <= 0 else f"mỗi {args.repair_interval:.0f}ph ({args.repair_mode})"}')
    net = ChurnNetwork(args.nodes, args.seed, args.median_session,
                       args.session_dist, args.weibull_shape)

    # ---- ingest ----
    t0 = time.time()
    print(f'[*] Ingest {N_DOCS:,} doc ...')
    anchors = AnchorTable()
    keys_per_doc = {}
    for i in range(N_DOCS):
        ks = [net.key63(E[i], proj) for proj in P]
        keys_per_doc[i] = ks
        anchors.payload[i] = codes[i]
        aset = []
        for skey in ks:
            for ni in net.knn(skey, r):
                net.ram[int(ni)][i] = TTL_INF if args.repair_interval <= 0 else args.ttl
                aset.append(int(ni))
        anchors[i] = aset
        if (i + 1) % 20000 == 0:
            print(f'    {i+1:,}/{N_DOCS:,}')
    print(f'    xong {time.time()-t0:.0f}s')

    # ---- warm-up: chạy churn cho hệ về trạng thái dừng ----
    # Không đo trong giai đoạn này. Nếu không warm-up, kết quả epoch đầu phản ánh
    # trạng thái khởi tạo nhân tạo (mọi node vừa mới bắt đầu phiên) chứ không phải
    # trạng thái một mạng đang chạy.
    if args.warmup > 0:
        nw = max(1, int(round(args.warmup / args.epoch)))
        print(f'[*] Warm-up {args.warmup:.0f}ph ({nw} epoch), không đo ...')
        wsince = 0.0
        for _ in range(nw):
            net.step(args.epoch)
            wsince += args.epoch
            # PHẢI theo đúng chu kỳ. Bản trước sửa MỖI EPOCH trong warm-up, nên
            # cấu hình "sửa mỗi 1440ph" vẫn được sửa 12 lần trước khi đo — kết
            # quả trông tốt hơn thực tế.
            if args.repair_interval > 0 and wsince >= args.repair_interval:
                repair(net, anchors, keys_per_doc, r, args.ttl, args.repair_mode)
                wsince = 0.0
        print(f'    {net.total_departures:,} lượt rời mạng trong warm-up')

    # ---- vòng chính ----
    print(f'[*] Mô phỏng {args.duration:.0f}ph = {n_epoch} epoch '
          f'× {args.epoch:.1f}ph, đo mỗi {args.measure_every} epoch')
    hist = []
    rnd_route = random.Random(args.seed + 777)
    since_repair = 0.0
    dep0 = net.total_departures
    msg0 = net.repair_msgs

    def snapshot(ep, t, phase=None):
        """Chụp trạng thái. phase = vị trí trong chu kỳ sửa chữa, 0..1.

        MC3: bản trước chỉ chụp ở mốc epoch, mà epoch = median/12 còn mọi repair
        interval là bội số nguyên của epoch, nên snapshot CUỐI luôn rơi đúng lúc
        vừa repair xong. analyze_churn báo cáo hist[-1], tức con số ĐẸP NHẤT.
        Giờ chụp ở pha ngẫu nhiên trong chu kỳ, đúng như query đến bất kỳ lúc nào.

        MC12: tách ba đại lượng vốn bị gộp làm một.
          physical  — còn node nào giữ record không (repair kiểm soát cái này)
          routable  — client ĐỊNH TUYẾN tới được không (người dùng thấy cái này)
          e2e       — truy vấn có trả về đúng object không
        """
        # physical: anchor còn giữ record
        alive_meta = sum(1 for tag in anchors
                         if any(tag in net.ram[ni] for ni in anchors[tag]))
        # routable: đi từ semantic key, xem có chạm node nào đang giữ record không.
        # Lấy mẫu để khỏi tốn: 500 doc ngẫu nhiên là đủ hẹp khoảng tin cậy.
        _rs = random.Random(args.seed + 991 + ep)
        _sample = _rs.sample(range(N_DOCS), min(500, N_DOCS))
        _reach = 0
        for tag in _sample:
            found = False
            for skey in keys_per_doc[tag]:
                for ni in net.knn(skey, args.k_query):
                    if tag in net.ram[int(ni)]:
                        found = True
                        break
                if found:
                    break
            _reach += found
        routable = 100.0 * _reach / len(_sample)
        rr, fr, tc, rand = measure_recall(net, P, Qv, gt, codes, codebook,
                                          args, n_query, rnd_route)
        meta_per_node = float(np.mean([len(x) for x in net.ram]))
        row = {'epoch': ep, 't_min': round(t, 1),
               'phase': phase,
               # MC12: ba đại lượng khác nhau, KHÔNG so chéo được
               'meta_physical': 100.0 * alive_meta / N_DOCS,
               'meta_routable': routable,
               'meta_avail': 100.0 * alive_meta / N_DOCS,   # tương thích cũ
               'reach_r5': rr, 'final_r5': fr, 'random_r5': rand,
               'ratio': (fr / rand) if rand > 0 else float('nan'),
               'nodes_touched': tc,
               'footprint': meta_per_node * args.nodes / N_DOCS,
               'departures': net.total_departures - dep0,
               'repair_msgs': net.repair_msgs - msg0}
        hist.append(row)
        print(f'    t={t:>6.0f}ph{"" if phase is None else f" pha{phase:.2f}"} | '
              f'phys {row["meta_physical"]:5.1f}% route {row["meta_routable"]:5.1f}% | '
              f'sem {fr:5.1f}% rnd {rand:5.1f}% = {row["ratio"]:4.2f}x | '
              f'vết {row["footprint"]:5.2f} | msg {row["repair_msgs"]:>8,}')
        return row

    print(f'    {"":8s} {"":13s} {"":22s} {"":14s}')
    # MC3: chọn TRƯỚC các epoch sẽ đo, rải đều theo PHA trong chu kỳ sửa chữa,
    # thay vì đo ở mốc cố định. Nếu đo ở mốc thì mọi lần đo đều rơi ngay sau
    # repair và availability luôn đẹp nhất.
    _rs_meas = random.Random(args.seed + 4242)
    _per_cycle = max(1, int(round(args.repair_interval / args.epoch))) \
        if args.repair_interval > 0 else 1
    _meas_eps = set()
    for _k in range(max(6, n_epoch // args.measure_every)):
        # rải đều qua thời gian, mỗi lần lệch pha ngẫu nhiên trong chu kỳ
        _base = int((_k + 1) * n_epoch / max(6, n_epoch // args.measure_every))
        _off = _rs_meas.randrange(_per_cycle) if _per_cycle > 1 else 0
        _e = min(n_epoch, max(1, _base - _off))
        _meas_eps.add(_e)
    _meas_eps.add(n_epoch)

    snapshot(0, 0.0, phase=0.0)
    for ep in range(1, n_epoch + 1):
        net.step(args.epoch)
        since_repair += args.epoch
        _just_repaired = False
        if args.repair_interval > 0 and since_repair >= args.repair_interval:
            repair(net, anchors, keys_per_doc, r, args.ttl, args.repair_mode)
            since_repair = 0.0
            _just_repaired = True
        if ep in _meas_eps:
            # pha = đã trôi bao nhiêu phần của chu kỳ kể từ lần sửa gần nhất.
            # 0 = vừa sửa xong (trường hợp tốt nhất), gần 1 = sắp tới lần sửa
            # tiếp (trường hợp xấu nhất).
            ph = 0.0 if _just_repaired else (
                since_repair / args.repair_interval if args.repair_interval > 0 else 1.0)
            snapshot(ep, ep * args.epoch, phase=min(1.0, ph))

    # ---- tổng kết ----
    n_repair_rounds = int(args.duration / args.repair_interval) if args.repair_interval > 0 else 0
    if args.repair_interval > 0 and n_repair_rounds < 3:
        print(f'\n[!] CẢNH BÁO: sửa chữa chỉ chạy {n_repair_rounds} lần trong '
              f'thời lượng đo. Chưa đủ để vào trạng thái dừng.')

    # MC3: báo cáo TOÀN BỘ các lần đo, không chỉ lần cuối. Lần cuối là một pha
    # cụ thể trong chu kỳ, không đại diện.
    _m = [h for h in hist if h['epoch'] > 0]
    if _m:
        import statistics as _st
        print()
        print('  MC3 — availability theo PHA trong chu kỳ sửa chữa:')
        print(f"    {'pha':>5s} {'physical':>9s} {'routable':>9s} {'R@5':>7s}")
        for h in sorted(_m, key=lambda x: x.get('phase') or 0):
            ph = h.get('phase')
            print(f"    {('%.2f' % ph) if ph is not None else '  ?':>5s} "
                  f"{h['meta_physical']:>8.1f}% {h['meta_routable']:>8.1f}% "
                  f"{h['final_r5']:>6.1f}%")
        for k, lbl in [('meta_physical', 'physical'), ('meta_routable', 'routable')]:
            v = [h[k] for h in _m]
            print(f'    {lbl:9s}: mean {_st.mean(v):.1f}%  min {min(v):.1f}%  '
                  f'max {max(v):.1f}%')
        print('    (min là con số một deployment phải chịu được, không phải mean)')

    first, last = hist[0], hist[-1]
    print()
    print('=' * 78)
    print(f'KẾT QUẢ | {args.dataset} | r={r} | '
          f'{"KHÔNG sửa" if args.repair_interval <= 0 else f"sửa mỗi {args.repair_interval:.0f}ph"}')
    print('=' * 78)
    print(f'  Metadata availability : {first["meta_avail"]:.1f}% -> {last["meta_avail"]:.1f}%'
          f'  ({last["meta_avail"]-first["meta_avail"]:+.1f}đ)')
    print(f'  Recall@5 semantic     : {first["final_r5"]:.1f}% -> {last["final_r5"]:.1f}%'
          f'  ({last["final_r5"]-first["final_r5"]:+.1f}đ)')
    print(f'  Recall@5 ngẫu nhiên   : {first["random_r5"]:.1f}% -> {last["random_r5"]:.1f}%')
    print(f'  TỈ LỆ sem/rand        : {first["ratio"]:.2f}x -> {last["ratio"]:.2f}x'
          f'   <-- đại lượng quyết định')
    print(f'  Lượt rời mạng         : {last["departures"]:,} '
          f'({100.0*last["departures"]/args.nodes:.0f}% số node)')
    print(f'  Message sửa chữa      : {last["repair_msgs"]:,}')
    if last['repair_msgs'] > 0:
        per_min = last['repair_msgs'] / args.duration
        print(f'    = {per_min:,.0f} msg/phút, '
              f'{last["repair_msgs"]/N_DOCS:.2f} msg mỗi doc trong {args.duration:.0f}ph')
    meta_per_node = np.mean([len(x) for x in net.ram])
    eff_footprint = meta_per_node * args.nodes / N_DOCS
    print(f'  Metadata/node cuối    : {meta_per_node:.1f}')
    print(f'  DẤU VẾT THỰC TẾ       : {eff_footprint:.2f} bản/doc '
          f'(danh nghĩa L*r = {L*r})')
    if eff_footprint > L * r * 1.3:
        print(f'    *** CẢNH BÁO: dấu vết thực gấp {eff_footprint/(L*r):.1f} lần danh nghĩa.')
        print(f'    Sửa chữa đang tích tụ bản sao. Giảm TTL hoặc kiểm lại purge. ***')

    out = args.out or (
        f'churn_{args.dataset}_N{args.nodes}_r{r}'
        f'_ses{int(args.median_session)}_{args.session_dist}'
        f'_rep{int(args.repair_interval)}{args.repair_mode[0] if args.repair_interval>0 else ""}'
        f'_s{args.seed}_nq{n_query}.json')
    json.dump({'config': vars(args), 'corpus': N_DOCS, 'history': hist},
              open(out, 'w'), indent=2)
    print(f'\n-> Lưu: {out}')


if __name__ == '__main__':
    main()