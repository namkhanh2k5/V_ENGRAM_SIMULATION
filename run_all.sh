#!/usr/bin/env bash
# MỘT LỆNH chạy toàn bộ simulation, mọi kết quả vào MỘT file log.
# Foreground:  bash run_all.sh
# Chạy nền:    bash run_all.sh --bg     (rồi đóng terminal vẫn chạy)
set -euo pipefail

mkdir -p logs
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -x "venv/bin/python" ]; then
  PYTHON_BIN="venv/bin/python"
fi

if [ "${1:-}" = "--bg" ]; then
  nohup "$PYTHON_BIN" -u run_all_experiments.py >> logs/console_$(date +%Y%m%d_%H%M%S).txt 2>&1 &
  echo "Đang chạy nền (PID $!). Theo dõi:  tail -f logs/v_engram_all_*.log"
else
  "$PYTHON_BIN" -u run_all_experiments.py
fi
