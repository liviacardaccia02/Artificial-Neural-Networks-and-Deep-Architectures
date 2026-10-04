import os
import numpy as np
import matplotlib.pyplot as plt
from shared.mlp import MLP

SEED = 42
np.random.seed(SEED)
EPOCHS = 1000
ETA = 0.1
ALPHA = 0.9
BEST_HIDDEN_SIZE = 4 

ndata = 100
mA = [1.0, 0.3]
sigmaA = 0.2
mB = [0.0, -0.1]
sigmaB = 0.3

# Class A: Bimodal (Left cloud at -1.0, Right cloud at 1.0) to wrap around Class B
classA_x = np.concatenate([
    np.random.randn(ndata // 2) * sigmaA - mA[0],
    np.random.randn(ndata // 2) * sigmaA + mA[0]
])
classA_y = np.random.randn(ndata) * sigmaA + mA[1]
classA = np.vstack((classA_x, classA_y))

# Class B: Unimodal (Center cloud at 0.0)
classB_x = np.random.randn(ndata) * sigmaB + mB[0]
classB_y = np.random.randn(ndata) * sigmaB + mB[1]
classB = np.vstack((classB_x, classB_y))

# Create grid for decision boundary visualization
X_all = np.hstack((classA, classB))
x_min, x_max = X_all[0, :].min() - 0.5, X_all[0, :].max() + 0.5
y_min, y_max = X_all[1, :].min() - 0.5, X_all[1, :].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.vstack([xx.ravel(), yy.ravel()])

def get_scenario_data(scenario):
    idxA = np.random.permutation(ndata)
    idxB = np.random.permutation(ndata)
    
    if scenario == 1:
        # Random 25% from each class
        train_idxA, train_idxB = idxA[:25], idxB[:25]
        
    elif scenario == 2:
        # Random 50% from each class
        train_idxA, train_idxB = idxA[:50], idxB[:50]
        
    elif scenario == 3:
        # 20% from Class A where x < 0 (left cloud) and 80% where x > 0 (right cloud)
        idxA_neg = np.where(classA[0, :] < 0)[0]
        idxA_pos = np.where(classA[0, :] >= 0)[0]
        np.random.shuffle(idxA_neg)
        np.random.shuffle(idxA_pos)
        
        # Out of 50 training samples for Class A: 10 from left, 40 from right
        train_idxA = np.concatenate([idxA_neg[:10], idxA_pos[:40]])
        train_idxB = idxB[:50] # Keep Class B at 50% for parity

    val_idxA = np.setdiff1d(np.arange(ndata), train_idxA)
    val_idxB = np.setdiff1d(np.arange(ndata), train_idxB)
    
    X_train = np.hstack((classA[:, train_idxA], classB[:, train_idxB]))
    T_train = np.array([-1]*len(train_idxA) + [1]*len(train_idxB)).reshape(1, -1)
    
    X_val = np.hstack((classA[:, val_idxA], classB[:, val_idxB]))
    T_val = np.array([-1]*len(val_idxA) + [1]*len(val_idxB)).reshape(1, -1)
    
    return X_train, T_train, X_val, T_val

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

print(f"Generating decision boundaries for h={BEST_HIDDEN_SIZE} across all scenarios...")

for i, ax in enumerate(axes):
    scenario = i + 1
    X_train, T_train, X_val, T_val = get_scenario_data(scenario)
    
    mlp = MLP(layers=[2, BEST_HIDDEN_SIZE, 1], seed=SEED)
    mlp.train(X_train, T_train, eta=ETA, alpha=ALPHA, epochs=EPOCHS, mode="batch")
    
    grid_out = mlp.forward(grid_points)
    zz = grid_out.reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, zz, levels=50, cmap='RdBu', alpha=0.4)
    ax.contour(xx, yy, zz, levels=[0.0], colors='black', linewidths=2, linestyles='--')
    
    train_a = X_train[:, T_train.flatten() == -1]
    train_b = X_train[:, T_train.flatten() == 1]
    val_a = X_val[:, T_val.flatten() == -1]
    val_b = X_val[:, T_val.flatten() == 1]

    # Two colors, but three clear point clouds will be visible
    ax.scatter(train_a[0], train_a[1], color='blue', edgecolor='k', label='Train Class A (Outer)', alpha=0.9)
    ax.scatter(train_b[0], train_b[1], color='red', edgecolor='k', label='Train Class B (Inner)', alpha=0.9)
    
    ax.scatter(val_a[0], val_a[1], color='blue', marker='x', s=60, label='Val Class A (Unseen)', alpha=0.8)
    ax.scatter(val_b[0], val_b[1], color='red', marker='x', s=60, label='Val Class B (Unseen)', alpha=0.8)

    ax.set_title(f'Scenario {scenario}', fontsize=14)
    ax.set_xlabel('Feature 1 ($x_1$)', fontsize=12)
    if scenario == 1:
        ax.set_ylabel('Feature 2 ($x_2$)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
fig.colorbar(contour, cax=cbar_ax, label='Network Output Value')
axes[2].legend(loc='upper right', bbox_to_anchor=(1.65, 1.0))

plt.suptitle(f'Experiment 2: Decision Boundaries vs. Sampling Bias (Batch, $h={BEST_HIDDEN_SIZE}$)', y=1.05, fontsize=16)

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
fig_path = os.path.join(results_dir, f'experiment2_decision_boundaries_h{BEST_HIDDEN_SIZE}.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Decision boundaries plot saved to {fig_path}")

plt.show()