import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CẤU HÌNH FONT CHUẨN HỌC THUẬT (LATEX-STYLE)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
})

# ==========================================
# BIỂU ĐỒ 2: MINH HỌA LSH BOUNDARY EFFECT & RIPPLE SEARCH
# ==========================================
def plot_boundary_effect():
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Vẽ mặt phẳng và siêu mặt phẳng (Hyperplane)
    x = np.linspace(0, 10, 100)
    ax.plot(x, x, 'k--', linewidth=2, label="LSH Hyperplane")
    
    # Tô màu 2 vùng không gian hash khác nhau
    ax.fill_between(x, x, 10, color='blue', alpha=0.05, label="Hash Bit = 1")
    ax.fill_between(x, 0, x, color='red', alpha=0.05, label="Hash Bit = 0")
    
    # Tọa độ các vector
    Q = (3, 7)    # Query Vector
    A = (6, 6.2)  # Target A (Cùng phía với Query)
    B = (6.2, 5.8)  # Target B (Rất gần A nhưng bị rớt sang phía kia)
    
    # Vẽ các điểm
    ax.plot(*Q, 'go', markersize=10, label="Query Vector (Q)")
    ax.plot(*A, 'b^', markersize=10, label="Target A (Found, Same Bucket)")
    ax.plot(*B, 'rv', markersize=10, label="Target B (Lost, Boundary Effect)")
    
    # Vẽ các vòng tròn đồng tâm (Ripple Search Multi-probe) vớt điểm B
    circles = [1.5, 3.5, 5.0]
    for r in circles:
        circle = plt.Circle(Q, r, color='green', fill=False, linestyle=':', linewidth=1.5, alpha=0.6)
        ax.add_patch(circle)
    
    # Thêm mũi tên Ripple Search
    ax.annotate('Multi-Probe\nRipple Search', xy=(7, 4), xytext=(8, 2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=11, ha='center')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Semantic Dimension X")
    ax.set_ylabel("Semantic Dimension Y")
    ax.set_title("LSH Boundary Effect & Multi-Probe Mitigation")
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('fig_boundary_effect.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Đã lưu biểu đồ: fig_boundary_effect.pdf")


# ==========================================
# BIỂU ĐỒ 3: DUAL-AXIS LINE CHART (IMPACT OF L)
# ==========================================
def plot_impact_of_L():
    # Dữ liệu từ thực nghiệm
    L = [1, 2, 3, 4, 5]
    success_rate = [50.4, 77.4, 89.6, 94.6, 97.44]
    avg_hops = [10.0, 20.3, 30.3, 40.2, 50.4]
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    # Trục Y bên trái: Success Rate
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Projections ($L$)')
    ax1.set_ylabel('Success@5 (%)', color=color1, fontweight='bold')
    line1 = ax1.plot(L, success_rate, marker='o', markersize=8, color=color1, linewidth=2, label='Success@5')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(40, 105)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Đánh dấu điểm tối ưu L=5
    ax1.annotate('Optimal\n(97.4%)', xy=(5, 97.44), xytext=(4.2, 85),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=6),
                 fontsize=11, color='blue', ha='center')

    # Trục Y bên phải: Avg Hops
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('Average Routing Hops', color=color2, fontweight='bold')  
    line2 = ax2.plot(L, avg_hops, marker='s', markersize=8, color=color2, linewidth=2, linestyle='--', label='Avg Hops')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 60)

    # Gộp Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right', frameon=True, shadow=True)
    
    plt.title("Ablation Study: Impact of Projections on Recall vs. Routing Cost")
    plt.tight_layout()
    plt.savefig('fig_impact_L.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Đã lưu biểu đồ: fig_impact_L.pdf")


# ==========================================
# BIỂU ĐỒ 4: O(log N) SCALABILITY TEST
# ==========================================
def plot_scalability():
    N_nodes = [10000, 15000, 20000, 25000]
    avg_hops = [51.2, 50.8, 51.9, 51.8]
    
    # Tính đường lý thuyết O(log N) làm baseline để so sánh
    # Đặt hệ số c sao cho đường log(N) khớp với điểm N=10000 (Hops = 51.2)
    c = 51.2 / np.log2(10000)
    N_theory = np.linspace(10000, 25000, 100)
    hops_theory = c * np.log2(N_theory)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Vẽ đường thực nghiệm
    ax.plot(N_nodes, avg_hops, marker='^', markersize=10, color='black', linewidth=2.5, label='Observed V-Engram Hops (L=5)')
    
    # Vẽ đường lý thuyết O(log N)
    ax.plot(N_theory, hops_theory, color='gray', linestyle='--', linewidth=2, label=r'Theoretical Baseline $\mathcal{O}(\log_2 N)$')
    
    ax.set_xlabel('Network Size ($N$ Nodes)')
    ax.set_ylabel('Average Routing Hops')
    ax.set_title("Scalability Analysis: Network Size vs. Routing Overhead")
    
    # Set ylim để thấy rõ đường line cực kỳ phẳng so với scale đồ thị
    ax.set_ylim(40, 65)
    
    # Đổi tick trục X cho đẹp
    ax.set_xticks(N_nodes)
    ax.set_xticklabels(['10K', '15K', '20K', '25K'])
    
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('fig_scalability.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Đã lưu biểu đồ: fig_scalability.pdf")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    print("Đang tạo biểu đồ chất lượng cao (PDF) cho bài báo Springer LNCS...")
    plot_boundary_effect()
    plot_impact_of_L()
    plot_scalability()
    print("Hoàn thành! Bạn có thể sử dụng các file PDF để chèn vào LaTeX.")