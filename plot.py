import matplotlib.pyplot as plt

# Data
node_counts = [50, 200, 500, 700, 1000]
q_learning_implementations = {
    'Sequential Q-learning': [0.0615737, 2.11942, 12.3951, 40.7896, 410.857],
    'OpenMP Q-learning':     [0.711252, 3.25352, 14.5135, 36.8152, 314.82],
    'CUDA Q-learning':       [0.254046, 0.297333, 0.911919, 1.79595, 20.1306],
}

# Color and marker styles
colors = ['#FF5733', '#2980B9', '#27AE60']
markers = ['o', 's', 'D']

# Plot
plt.figure(figsize=(10, 6))

for (impl, times), color, marker in zip(q_learning_implementations.items(), colors, markers):
    plt.plot(node_counts, times, label=impl, marker=marker, linewidth=2.5, color=color)

plt.xlabel("Number of Nodes", fontsize=12)
plt.ylabel("Execution Time (s)", fontsize=12)
plt.yscale('log')  # Log scale for clarity
plt.title("Comparative Analysis of Q-learning Implementations", fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xticks(node_counts)
plt.legend(fontsize=10)
plt.tight_layout()

# Save figure
plt.savefig("q_learning_comparative_analysis.png", dpi=300)
plt.show()
