import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'results', 'experiment1_hidden_size_results.csv')
fig_path = os.path.join(script_dir, 'results', 'experiment1_hidden_size_plot.png')

# Check if the CSV exists before plotting
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find results CSV at {csv_path}. Run experiment1_hidden_size.py first!")

df = pd.read_csv(csv_path)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Top subplot: MSE (zoomed to data range)
ax1.plot(df['hidden_size'], df['final_train_mse'], color='tab:blue', marker='o', linewidth=2)
ax1.set_ylabel('Final Training MSE', color='tab:blue', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
# Automatically scale y-limits tightly around the data instead of forcing 0
ax1.set_ylim(df['final_train_mse'].min() * 0.95, df['final_train_mse'].max() * 1.05)

# Bottom subplot: Misclassification Rate
ax2.plot(df['hidden_size'], df['final_train_misclass'], color='tab:red', marker='s', linestyle='--', linewidth=2)
ax2.set_ylabel('Misclassification Rate', color='tab:red', fontsize=11)
ax2.set_xlabel('Number of Hidden Nodes', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.set_ylim(-0.02, 0.05)  # Zoomed near 0 since error is ~1%

plt.suptitle('Experiment 1: Network Capacity vs. Training Performance', fontsize=14)
fig.tight_layout()

# Save high-resolution image for your report
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Plot successfully generated and saved to {fig_path}")

plt.show()