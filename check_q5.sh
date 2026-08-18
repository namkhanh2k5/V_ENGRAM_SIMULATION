#!/bin/bash
echo "  A. headline   : $(grep -l 'Recall@5' q5_A_*.txt 2>/dev/null | wc -l)/40"
echo "  B. termination: $(grep -l 'Recall@5' q5_B_*.txt 2>/dev/null | wc -l)/40"
echo "  C. margin     : $(grep -l 'Recall@5' q5_C_*.txt 2>/dev/null | wc -l)/20"
echo "  D. chi phí    : $(grep -l 'BẢNG CHI PHÍ' q5_D_*.txt 2>/dev/null | wc -l)/3"
echo "  đang chạy: $(ps aux | grep main_simulation.py | grep -v grep | wc -l)"
