#!/usr/bin/env python3
"""
Cài k-bucket Kademlia thay small-world ring — bản dùng cho branch thử nghiệm.

    git checkout -b kbucket
    python3 patch_kbucket.py          # áp bản vá
    git diff --stat                   # xem đổi gì

CHẠY THỬ NHANH TRƯỚC KHI CHẠY THẬT (khoảng 20 phút):

    for RT in ring kbucket; do
      echo -n "  $RT: "
      SKIP_PAYLOAD=1 ROUTING_TABLE=$RT python3 main_simulation.py \
        --dataset scifact --nodes 3000 --seed 1 --k-query 20 \
        --multi-probe 8 --meta-anchors 1 --nq 50 2>&1 \
        | grep -oE "Recall@5   : [0-9.]+%|RPC/query      : [0-9.]+|Node chạm/query: [0-9.]+" \
        | tr '\n' ' '; echo
    done

Nếu hai cấu trúc cho kết quả gần nhau thì không cần chạy lại toàn bộ; chỉ cần
nêu trong bài rằng kết quả không phụ thuộc lựa chọn này. Nếu lệch nhiều thì phải
rerun và con số hiện tại phải thay.

---------------------------------------------------------------------------
VÌ SAO CẦN: bài nói Ripple Search chạy trên "Kademlia routing state", nhưng
bootstrap_network dựng một small-world ring: mỗi peer giữ 25 láng giềng mỗi phía
trên vòng đã sắp theo node_id, cộng 20 liên kết ngẫu nhiên. Không có bucket phân
tầng theo tiền tố XOR, không có eviction, và FIND_NODE trả về TOÀN BỘ bảng.

Khác biệt không nằm ở số contact — hai bên cùng bậc, ~70 so với ~100-260 — mà ở
PHÂN BỐ. Ring dày ở gần theo thang node_id tuyến tính; k-bucket dày ở gần theo
thang XOR. Ripple Search gom peer quanh một khoá và cần các peer đó biết nhau,
điều mà ring bảo đảm còn k-bucket thì không.
"""
import re
import sys

NODE_PATCH = '''
    # ---------------- k-bucket Kademlia ----------------
    # Bảng định tuyến phân tầng theo tiền tố XOR: bucket i giữ các peer có
    # khoảng cách XOR trong [2^i, 2^(i+1)), tối đa K_BUCKET peer mỗi bucket.
    # Đây là cấu trúc mà bài mô tả; nhánh ring giữ lại để đối chiếu.
    def kb_index(self, other_id):
        d = self.node_id ^ other_id
        return d.bit_length() - 1 if d else 0

    def kb_add(self, peer, k_bucket=20):
        if peer.node_id == self.node_id:
            return
        i = self.kb_index(peer.node_id)
        b = self.kbuckets.setdefault(i, [])
        if peer in b:
            b.remove(peer); b.append(peer)      # LRU: mới dùng đẩy về cuối
        elif len(b) < k_bucket:
            b.append(peer)
        # bucket đầy: Kademlia giữ peer cũ nếu còn sống. Ở đây peer không chết
        # trong lúc bootstrap nên bỏ qua, và ta KHÔNG thay peer cũ.

    def kb_closest(self, key, count):
        """Trả count peer gần key nhất trong bảng, gom từ mọi bucket."""
        allp = [p for b in self.kbuckets.values() for p in b]
        allp.sort(key=lambda p: p.node_id ^ key)
        return allp[:count]
'''


def patch_node(path='src/node.py'):
    s = open(path, encoding='utf-8').read()
    if 'kb_closest' in s:
        print('  [đã vá] src/node.py'); return False
    # thêm kbuckets vào __init__
    m = re.search(r'(def __init__\(self[^)]*\):\n)', s)
    if not m:
        print('  ✗ không tìm thấy __init__ trong node.py'); return False
    j = s.find('\n', m.end())
    while s[j:j+1] == '\n':
        j += 1
    # chèn ngay sau dòng đầu của __init__
    ins = '        self.kbuckets = {}      # k-bucket: index -> list peer\n'
    s = s[:m.end()] + ins + s[m.end():]
    # thêm ba method sau get_neighbors
    m2 = re.search(r'(def get_neighbors\(self\):\n\s+return list\(self\.routing_table\)\n)', s)
    if not m2:
        print('  ✗ không tìm thấy get_neighbors'); return False
    s = s[:m2.end()] + NODE_PATCH + s[m2.end():]
    open(path, 'w', encoding='utf-8').write(s)
    print('  ✓ src/node.py: thêm kbuckets, kb_add, kb_closest')
    return True


def patch_network(path='src/network.py'):
    s = open(path, encoding='utf-8').read()
    if 'ROUTING_TABLE' in s:
        print('  [đã vá] src/network.py'); return False

    s = s.replace('ROUTING_MODE = _os.environ.get("ROUTING_MODE", "auto")',
'''ROUTING_MODE = _os.environ.get("ROUTING_MODE", "auto")
# Cấu trúc bảng định tuyến: "ring" = small-world ring hiện dùng, "kbucket" =
# k-bucket Kademlia phân tầng theo tiền tố XOR.
ROUTING_TABLE = _os.environ.get("ROUTING_TABLE", "ring")
K_BUCKET = int(_os.environ.get("K_BUCKET", "20"))''')

    old = '''    print("[*] Đan cấu trúc Small-World (Ring-Adjacency: 50 xa + 50 gần)...")
    for i, node in enumerate(network_nodes):'''
    new = '''    if ROUTING_TABLE == "kbucket":
        # Mỗi peer học một mẫu peer khác rồi nạp vào k-bucket. Mẫu phải đủ lớn
        # để các bucket XA được lấp; bucket GẦN thưa tự nhiên vì ít peer ở đó.
        print(f"[*] Dựng k-bucket Kademlia (k={K_BUCKET}) ...")
        sample_size = min(num_nodes, 400)
        for node in network_nodes:
            for peer in random.sample(network_nodes, sample_size):
                node.kb_add(peer, K_BUCKET)
            # Bảo đảm peer biết các peer XOR-gần nhất: đây là điều bootstrap
            # thật đạt được bằng cách tự tra chính node_id của mình.
            for peer in sorted(network_nodes,
                               key=lambda p: p.node_id ^ node.node_id)[:K_BUCKET]:
                node.kb_add(peer, K_BUCKET)
            node.routing_table = set(p for b in node.kbuckets.values() for p in b)
        _sz = [len(n.routing_table) for n in network_nodes]
        print(f"    contact/peer: TB {sum(_sz)/len(_sz):.0f} "
              f"(min {min(_sz)}, max {max(_sz)})")
        yield env.timeout(0)
        print(f"✓ Mạng lưới sẵn sàng ({time.time() - start_time:.2f}s)")
        # PHẢI trả network_nodes: hàm này là generator do env.process bọc, và
        # main_simulation nhận giá trị trả về. `return` trơn cho None, và lỗi
        # chỉ lộ ra ở data_ingestion_process nên rất khó truy.
        return network_nodes

    print("[*] Đan cấu trúc Small-World (Ring-Adjacency: 50 xa + 50 gần)...")
    for i, node in enumerate(network_nodes):'''
    if old not in s:
        print('  ✗ không tìm thấy khối bootstrap ring'); return False
    s = s.replace(old, new)
    open(path, 'w', encoding='utf-8').write(s)
    print('  ✓ src/network.py: thêm ROUTING_TABLE, nhánh dựng k-bucket')
    return True


def patch_routing(path='src/routing.py'):
    """FIND_NODE trả k contact GẦN NHẤT thay vì toàn bộ bảng."""
    s = open(path, encoding='utf-8').read()
    if 'kb_closest' in s:
        print('  [đã vá] src/routing.py'); return False
    old = '            candidates.update(node.get_neighbors())'
    new = '''            # Kademlia trả về k contact GẦN NHẤT với khoá, không phải toàn
            # bộ bảng. Nhánh ring giữ hành vi cũ để đối chiếu.
            if hasattr(node, "kb_closest") and node.kbuckets:
                candidates.update(node.kb_closest(key, k))
            else:
                candidates.update(node.get_neighbors())'''
    if old not in s:
        print('  ✗ không tìm thấy chỗ gom neighbor'); return False
    s = s.replace(old, new)
    open(path, 'w', encoding='utf-8').write(s)
    print('  ✓ src/routing.py: FIND_NODE trả k contact gần nhất khi dùng k-bucket')
    return True


if __name__ == '__main__':
    print('Áp bản vá k-bucket:')
    ok = [patch_node(), patch_network(), patch_routing()]
    print()
    if any(ok):
        print('Xong. Kiểm cú pháp:')
        import ast
        for f in ('src/node.py', 'src/network.py', 'src/routing.py'):
            try:
                ast.parse(open(f).read()); print(f'  ✓ {f}')
            except SyntaxError as e:
                print(f'  ✗ {f} dòng {e.lineno}: {e.msg}'); sys.exit(1)
        print()
        print('CHẠY THỬ (xem docstring đầu file):')
        print('  for RT in ring kbucket; do ... done')
    else:
        print('Không có gì thay đổi.')