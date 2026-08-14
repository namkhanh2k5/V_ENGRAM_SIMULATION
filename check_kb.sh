#!/bin/bash
# Kiểm tiến độ rerun k-bucket
echo "  A. headline   : $(grep -l 'Recall@5' kbf_A_*.txt 2>/dev/null | wc -l)/40"
echo "  B. termination: $(grep -l 'Recall@5' kbf_B_*.txt 2>/dev/null | wc -l)/20"
echo "  C. margin     : $(grep -l 'Recall@5' kbf_C_*.txt 2>/dev/null | wc -l)/10"
echo "  D. chi phí    : $(grep -l 'BẢNG CHI PHÍ' kbf_D_*.txt 2>/dev/null | wc -l)/3"
echo ""
echo "  đang chạy: $(ps aux | grep main_simulation.py | grep -v grep | wc -l) tiến trình"
