import os
import numpy as np
import matplotlib.pyplot as plt
from part1_classification.generate_data import generate_non_linearly_separable_data, subsample_scenario
from shared.mlp import MLP

SEED = 42
EPOCHS = 1000
ETA = 0.1
ALPHA = 0.9
BEST_HIDDEN_SIZE = 4  # Sufficient capacity based on our experiments

X, T, classA, classB = generate_non_linearly_separable_data(n=100, seed=SEED)

# Create a dense 2D grid over the feature space
x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.vstack([xx.ravel(), yy.ravel()])

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
scenarios = [1, 2, 3]

print(f"Generating decision boundaries for h={BEST_HIDDEN_SIZE} across all scenarios...")

for scenario, ax in zip(scenarios, axes):
    # Get subsampled data for this specific scenario
    X_train, T_train, X_val, T_val = subsample_scenario(classA, classB, scenario, seed=SEED)
    
    # Train MLP on the training subset only
    mlp = MLP(layers=[2, BEST_HIDDEN_SIZE, 1], seed=SEED)
    mlp.train(X_train, T_train, eta=ETA, alpha=ALPHA, epochs=EPOCHS, mode="batch")
    
    # Evaluate the trained model across the entire grid
    grid_out = mlp.forward(grid_points)
    zz = grid_out.reshape(xx.shape)
    
    # Plot the color gradient and the exact boundary (0.0)
    contour = ax.contourf(xx, yy, zz, levels=50, cmap='RdBu', alpha=0.4)
    ax.contour(xx, yy, zz, levels=[0.0], colors='black', linewidths=2, linestyles='--')
    
    # Filter training and validation points by class
    train_a = X_train[:, T_train == -1]
    train_b = X_train[:, T_train == 1]
    val_a = X_val[:, T_val == -1]
    val_b = X_val[:, T_val == 1]

    # Plot Training points (Solid Circles)
    ax.scatter(train_a[0], train_a[1], color='blue', edgecolor='k', label='Train Class A', alpha=0.9)
    ax.scatter(train_b[0], train_b[1], color='red', edgecolor='k', label='Train Class B', alpha=0.9)
    
    # Plot Validation points (X markers)
    ax.scatter(val_a[0], val_a[1], color='blue', marker='x', s=60, label='Val Class A (Unseen)', alpha=0.8)
    ax.scatter(val_b[0], val_b[1], color='red', marker='x', s=60, label='Val Class B (Unseen)', alpha=0.8)

    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('Feature 1 ($x_1$)', fontsize=12)
    if scenario == 1:
        ax.set_ylabel('Feature 2 ($x_2$)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)

# Add colorbar and legend
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(contour, cax=cbar_ax, label='Network Output Value')
axes[2].legend(loc='upper right', bbox_to_anchor=(1.65, 1.0))

plt.suptitle(f'Experiment 2: Decision Boundaries vs. Sampling Bias (Batch, $h={BEST_HIDDEN_SIZE}$)', y=1.05, fontsize=16)

# Save and Show
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
fig_path = os.path.join(results_dir, f'experiment2_decision_boundaries_h{BEST_HIDDEN_SIZE}.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Decision boundaries plot saved to {fig_path}")

plt.show()