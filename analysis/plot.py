import matplotlib.pyplot as plt

# Data
node_counts = [50, 200, 500, 700, 1500, 5000]
q_learning_implementations = {
    'Sequential Q-learning': [0.0615737, 2.11942, 12.3951, 40.7896, 410.857, 5044.71],
    'OpenMP Q-learning':     [0.711252, 3.25352, 14.5135, 36.8152, 314.82, 3500.12],
    'CUDA Q-learning':       [0.254046, 0.297333, 0.911919, 1.79595, 20.1306, 173.145],
}

# Style settings
colors = ['#FF5733', '#2980B9', '#27AE60']
markers = ['o', 's', 'D']

# Plotting
plt.figure(figsize=(10, 6))
for (impl, times), color, marker in zip(q_learning_implementations.items(), colors, markers):
    plt.plot(node_counts, times, label=impl, marker=marker, linewidth=2.5, color=color)

plt.xlabel("Number of Nodes", fontsize=12)
plt.ylabel("Execution Time (s)", fontsize=12)
plt.yscale('log')  # Use log scale for better visibility
plt.title("Comparative Analysis of Q-learning Implementations", fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xticks(node_counts)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the figure
plt.savefig("q_learning_comparative_analysis_corrected.png", dpi=300)
plt.show()

# Calculate speedups
def calculate_speedup(sequential_times, parallel_times):
    return [seq/par for seq, par in zip(sequential_times, parallel_times)]

# OpenMP Speedup
plt.figure(figsize=(10, 6))
openmp_speedup = calculate_speedup(q_learning_implementations['Sequential Q-learning'],
                                 q_learning_implementations['OpenMP Q-learning'])
plt.plot(node_counts, openmp_speedup, marker='o', linewidth=2.5, color='#2980B9')
plt.xlabel("Number of Nodes", fontsize=12)
plt.ylabel("Speedup (Sequential/OpenMP)", fontsize=12)
plt.title("OpenMP Q-learning Speedup Analysis", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(node_counts)
plt.tight_layout()
plt.savefig("openmp_speedup_analysis.png", dpi=300)
plt.close()

# CUDA Speedup
plt.figure(figsize=(10, 6))
cuda_speedup = calculate_speedup(q_learning_implementations['Sequential Q-learning'],
                               q_learning_implementations['CUDA Q-learning'])
plt.plot(node_counts, cuda_speedup, marker='D', linewidth=2.5, color='#27AE60')
plt.xlabel("Number of Nodes", fontsize=12)
plt.ylabel("Speedup (Sequential/CUDA)", fontsize=12)
plt.title("CUDA Q-learning Speedup Analysis", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(node_counts)
plt.tight_layout()
plt.savefig("cuda_speedup_analysis.png", dpi=300)
plt.close()
