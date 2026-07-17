#!/bin/bash
# ============================================================================
# CHẠY TOÀN BỘ SIMULATION CHO PAPER
#   nohup bash run_paper.sh > run_paper.log 2>&1 &
#
# Sinh số cho: Bảng chính, Bảng r*, probe-vs-widen, Metadata hotspot,
#              Payload/node-failure, Baseline tập trung, Latency.
# Bỏ qua file đã có -> ngắt giữa chừng chạy lại vẫn tiếp tục.
# Ước tính: ~7 giờ (thêm nhóm H). A-D: 256 run × 65s = 4.6h. E: 6 × ~5ph = 30ph.
# F: 8 × ~2ph = 16ph. G: 2 × ~30ph = 1h (SimPy rất chậm).
# ============================================================================
set -u
SEEDS="20235956 1 2 3 4 5 6 7 8 9"
NQ=500
PY=python3

# Bỏ qua nếu file kết quả đã có -> ngắt giữa chừng, chạy lại vẫn tiếp tục chỗ dở.
# Tên file do main_simulation_v2.py sinh; ta dựng lại pattern để kiểm tra.
run() {
    local ds="code" L=5 K=20 R=1 T=1 N=10000 seed="" rand="" pq="m512" nq=$NQ
    local args=("$@") i=0
    while [ $i -lt ${#args[@]} ]; do
        case "${args[$i]}" in
            --dataset) ds="${args[$((i+1))]}" ;;
            --num-tables) L="${args[$((i+1))]}" ;;
            --k-query) K="${args[$((i+1))]}" ;;
            --meta-anchors) R="${args[$((i+1))]}" ;;
            --multi-probe) T="${args[$((i+1))]}" ;;
            --nodes) N="${args[$((i+1))]}" ;;
            --seed) seed="${args[$((i+1))]}" ;;
            --random-routing) rand="_RANDOM" ;;
            --no-pq) pq="nopq" ;;
        esac
        i=$((i+1))
    done
    local f="result_${ds}_N${N}_L${L}_K${K}_MA${R}_T${T}_${pq}${rand}_s${seed}_nq${nq}.json"
    if [ -f "$f" ]; then echo "  [skip] $f"; return; fi
    $PY main_simulation_v2.py --nodes $N --nq $nq --pq-variant m512 "$@" >/dev/null 2>&1 \
        || echo "  [LỖI] $*"
}

echo "##### A. BẢNG CHÍNH — cấu hình chốt (L=5 K=20 T=8 r=1) #####"
for s in $SEEDS; do for ds in code scifact; do
  echo "[A] $ds s=$s"
  run --dataset $ds --seed $s --num-tables 5 --multi-probe 8 --k-query 20 --meta-anchors 1 --use-pq
  run --dataset $ds --seed $s --num-tables 5 --multi-probe 8 --k-query 20 --meta-anchors 1 --no-pq
  run --dataset $ds --seed $s --num-tables 5 --multi-probe 8 --k-query 20 --meta-anchors 1 --use-pq --random-routing
done; done

echo "##### B. NGƯỠNG r* — contribution chính (sweep L·r) #####"
# Tỉ lệ sem/rand giảm đơn điệu theo TÍCH L*r, không theo r đơn lẻ.
for s in $SEEDS; do
  echo "[B] s=$s"
  for cfg in "5 1" "5 2" "5 3" "8 1" "12 1" "8 5" "8 10"; do
    set -- $cfg; L=$1; R=$2
    run --dataset code --seed $s --num-tables $L --meta-anchors $R --multi-probe 8 --k-query 20 --use-pq
    run --dataset code --seed $s --num-tables $L --meta-anchors $R --multi-probe 8 --k-query 20 --use-pq --random-routing
  done
done

echo "##### C. PROBE vs WIDEN — ngân sách node khớp #####"
for s in $SEEDS; do
  echo "[C] s=$s"
  run --dataset code --seed $s --num-tables 5 --multi-probe 3 --k-query 20 --meta-anchors 1 --use-pq
  run --dataset code --seed $s --num-tables 5 --multi-probe 1 --k-query 40 --meta-anchors 1 --use-pq
  run --dataset code --seed $s --num-tables 5 --multi-probe 5 --k-query 20 --meta-anchors 1 --use-pq
  run --dataset code --seed $s --num-tables 5 --multi-probe 1 --k-query 65 --meta-anchors 1 --use-pq
  run --dataset code --seed $s --num-tables 5 --multi-probe 1 --k-query 100 --meta-anchors 1 --use-pq
done

echo "##### D. METADATA HOTSPOT — Zipf + prefix occupancy #####"
# Chỉ 3 seed: phân bố load quyết định bởi corpus + node_id, không phải bởi query.
for s in 20235956 1 2; do for ds in code scifact; do
  echo "[D] $ds s=$s"
  $PY main_simulation_v2.py --dataset $ds --nodes 10000 --nq $NQ --pq-variant m512 \
      --seed $s --num-tables 5 --multi-probe 8 --k-query 20 --meta-anchors 1 --use-pq \
      --zipf 1.0 --prefix-occupancy > "hotspot_${ds}_s${s}.txt" 2>&1
done; done

echo "##### H. QUY MÔ MẠNG N — kiểm chứng Phương trình r* #####"
# P_rand = 1-(1-Lr/N)^(KLT) GIẢM khi N tăng ở L*r cố định, nên tỉ lệ sem/rand
# phải TĂNG theo N. Đây là phép kiểm chứng trực tiếp lý thuyết, không phải
# weak scaling (corpus cố định nên object/node giảm — đã ghi trong Threats).
for s in 20235956 1 2; do for N in 5000 10000 20000 40000; do
  echo "[H] N=$N s=$s"
  run --dataset code --seed $s --nodes $N --num-tables 5 --multi-probe 8 --k-query 20 --meta-anchors 1 --use-pq
  run --dataset code --seed $s --nodes $N --num-tables 5 --multi-probe 8 --k-query 20 --meta-anchors 1 --use-pq --random-routing
done; done

echo "##### E. PAYLOAD / NODE FAILURE — 3 seed, 2 corpus #####"
# main_churn_test.py nay đo TÁCH BẠCH 3 tầng: metadata availability / payload
# decode / end-to-end. Bản cũ tra GLOBAL_METADATA_DHT nên metadata không bao giờ
# chết -> chỉ đo ngưỡng erasure code (P(Bin(30,0.7)>=20) ~ 0.76 = đúng 76.4% cũ).
for s in 2026 1 2; do
  echo "[E] code s=$s"
  [ -s "failure_code_s${s}.txt" ] || DATASET=code NUM_FILES=20000 SAMPLE_SEED=$s \
      $PY main_churn_test.py > "failure_code_s${s}.txt" 2>&1 || true
  echo "[E] scifact s=$s"
  [ -s "failure_scifact_s${s}.txt" ] || DATASET=scifact NUM_FILES=5183 SAMPLE_SEED=$s \
      $PY main_churn_test.py > "failure_scifact_s${s}.txt" 2>&1 || true
done

echo "##### F. BASELINE TẬP TRUNG — HNSW / bucket-LSH / crypto-DHT / random #####"
# Ngân sách phải khớp UNIQUE candidates của V-Engram (từ Bảng chính, ~22% corpus
# trên code = ~4500, /L=5 -> ~900/bảng). Quét vài mức để vẽ đường recall-vs-pool.
for pool in 30 100 300 900; do for c in code scifact; do
  echo "[F] $c pool=$pool"
  [ -s "baseline_${c}_pool${pool}.txt" ] || CORPUS=$c POOL_PER_TABLE=$pool \
      $PY baselines.py > "baseline_${c}_pool${pool}.txt" 2>&1 || true
done; done

echo "##### G. LATENCY / RPC — SimPy, chỉ 20 query (rất chậm) #####"
# main_simulation.py có latency + payload tier mà v2 bỏ. Chỉ cần vài query để
# báo cáo rounds/RPC/latency trung bình cho Table 1.
for c in code scifact; do
  echo "[G] $c (có thể mất 30+ phút)"
  [ -s "latency_${c}.txt" ] || timeout 5400 $PY main_simulation.py --dataset $c \
      --nodes 10000 --seed 20235956 --k-query 20 --multi-probe 8 --meta-anchors 1 --nq 20 \
      > "latency_${c}.txt" 2>&1 || echo "  [BỎ QUA] latency $c quá 90 phút"
done

echo ""; echo "##### TỔNG HỢP #####"
MIN_NQ=$NQ $PY summarize.py > paper_tables.txt 2>&1
echo "-> paper_tables.txt      (Bảng chính, r*, probe-vs-widen)"
echo "-> hotspot_*.txt         (Zipf RPC load + prefix occupancy)"
echo "-> failure_*.txt         (metadata/payload/end-to-end theo mức node loss)"
echo "-> baseline_*.txt        (HNSW, bucket-LSH, crypto-DHT, random)"
echo "-> latency_*.txt         (rounds/RPC/latency cho Table 1)"
echo ""
echo "--- Hotspot (code) ---"; grep -h "RPC load\|c= " hotspot_code_s20235956.txt 2>/dev/null | head -6
echo "--- Node failure (code) ---"; grep -A 4 "Node loss" failure_code_s2026.txt 2>/dev/null | head -5
echo "--- Latency (code) ---"; grep -E "Rounds/query|RPC/query|Recall@5" latency_code.txt 2>/dev/null | head -3