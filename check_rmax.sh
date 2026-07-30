#!/usr/bin/env bash
# Kiểm tra điểm nghi ngờ số 4: Bảng 13 ghi R_max=15 -> 77.0 và R_max=20 -> 76.0,
# nhưng iterative_find_k_closest_nodes() KHÔNG dùng RNG nào và max_rounds chỉ là
# trần vòng lặp. Cả hai cấu hình đều có 0.0% lookup chạm trần, nên chúng PHẢI
# cho kết quả trùng khít từng bit. Nếu không trùng -> có nguồn phi tất định
# chưa được ghi nhận; nếu trùng -> Bảng 13 sai khi chép số từ log.
#
# Chạy trong tmux, cần venv đã activate. Mỗi run 30-90 phút.
#   tmux new -s rmax
#   bash check_rmax.sh 2>&1 | tee check_rmax.log
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}

echo "=== 1. R_max=15 vs R_max=20, --out TÁCH BIỆT (đây là chỗ bug) ==="
# QUAN TRỌNG: template tên file mặc định trong main_simulation.py là
#   result_full_{ds}_r{r}_K{k}_T{T}[_RANDOM]_s{seed}.json
# KHÔNG có R_MAX. Vòng sweep rmax in {5,10,15,20} dùng y hệt các tham số còn lại
# nên cả 4 lần ghi đè lên CÙNG một file; chỉ run cuối (rmax=20) còn sót.
# => luôn truyền --out khi sweep R_MAX.
for rmax in 15 20; do
  out="result_rmax${rmax}.json"
  if [ -s "$out" ]; then echo "  [skip] $out"; continue; fi
  echo "  --> R_MAX=$rmax"
  R_MAX=$rmax timeout 7200 $PY main_simulation.py \
      --dataset code --nodes 10000 --seed 20235956 \
      --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 20 \
      --out "$out" > "log_rmax${rmax}.txt" 2>&1 \
    || echo "  [TIMEOUT] R_MAX=$rmax — log_rmax${rmax}.txt bị CẮT, đừng đọc số từ nó"
done

echo
echo "=== 2. So sánh hai file (bỏ qua field thời gian nếu có) ==="
if [ -s result_rmax15.json ] && [ -s result_rmax20.json ]; then
  $PY - <<'PYEOF'
import json
a = json.load(open('result_rmax15.json')); b = json.load(open('result_rmax20.json'))
skip = {'elapsed','wall_time','runtime','timestamp','duration'}
ka, kb = set(a) - skip, set(b) - skip
diff = [(k, a.get(k), b.get(k)) for k in sorted(ka | kb) if a.get(k) != b.get(k)]
if not diff:
    print("  TRÙNG KHÍT. => Bảng 13 sai ở chỗ chép số: R_max=15 và 20 phải cùng recall.")
    print("     Sửa: cho hai hàng cùng một giá trị, hoặc bỏ cột Recall khỏi Bảng 13")
    print("     và giữ đúng cột 'lookups at cap' như phần text vốn đã khuyến nghị.")
else:
    print("  KHÁC NHAU ở %d field => CÓ nguồn phi tất định:" % len(diff))
    for k, x, y in diff:
        print(f"     {k}: rmax15={x!r}  rmax20={y!r}")
    print("     Cần seed lại nguồn đó, hoặc ghi rõ vào Threats to Validity.")
PYEOF
else
  echo "  thiếu file, xem log_rmax*.txt"
fi

echo
echo "=== 3. Kiểm tra log của vòng sweep cũ có bị cắt do timeout không ==="
for f in muc5_21_rmax5.txt muc5_21_rmax10.txt muc5_21_rmax15.txt muc5_21_rmax20.txt; do
  [ -f "$f" ] || { echo "  $f: KHÔNG CÓ"; continue; }
  if grep -q "Lưu:" "$f"; then echo "  $f: hoàn tất"; else echo "  $f: *** BỊ CẮT *** (không có dòng 'Lưu:') — số trong Bảng 13 lấy từ đây KHÔNG dùng được"; fi
done

echo
echo "=== 4. File JSON chi phí đang dùng cho Bảng 12 thuộc R_MAX nào? ==="
f=result_full_code_r1_K20_T8_s20235956.json
if [ -f "$f" ]; then
  echo "  $f — mtime: $(stat -c %y "$f")"
  echo "  So mtime này với muc5_21_rmax20.txt: nếu gần nhau thì file đã bị run"
  echo "  R_MAX=20 ghi đè, trong khi Bảng 2 khai R_max=15. Không đổi số (0% chạm"
  echo "  trần ở cả hai) nhưng cần chạy lại có --out để provenance đúng."
else
  echo "  không thấy $f"
fi
