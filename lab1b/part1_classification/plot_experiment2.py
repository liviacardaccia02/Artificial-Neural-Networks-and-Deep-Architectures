import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')

scenarios = [1, 2, 3]
modes = ['batch', 'sequential']
markers = ['o', 's', '^']
colors = ['tab:blue', 'tab:green', 'tab:orange']

# 1. Plot Final Performance vs. Hidden Size (Capacity Comparison)
for mode in modes:
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    has_data = False
    for scenario, marker, color in zip(scenarios, markers, colors):
        csv_path = os.path.join(results_dir, f'experiment2_subsampling_scenario{scenario}_results_{mode}.csv')
        
        if os.path.exists(csv_path):
            has_data = True
            df = pd.read_csv(csv_path)
            ax_train.plot(df['hidden_size'], df['final_train_mse'], marker=marker, color=color, linewidth=2, label=f'Scenario {scenario}')
            ax_val.plot(df['hidden_size'], df['val_mse'], marker=marker, color=color, linewidth=2, label=f'Scenario {scenario}')
        else:
            print(f"Notice: Could not find {csv_path}.")
            
    if not has_data:
        plt.close(fig)
        continue

    ax_train.set_title(f'Training MSE ({mode.capitalize()} Mode)', fontsize=13)
    ax_train.set_xlabel('Number of Hidden Nodes', fontsize=11)
    ax_train.set_ylabel('Mean Squared Error (MSE)', fontsize=11)
    ax_train.grid(True, linestyle=':', alpha=0.6)
    ax_train.legend(fontsize=10)

    ax_val.set_title(f'Validation MSE ({mode.capitalize()} Mode)', fontsize=13)
    ax_val.set_xlabel('Number of Hidden Nodes', fontsize=11)
    ax_val.grid(True, linestyle=':', alpha=0.6)
    ax_val.legend(fontsize=10)

    plt.suptitle(f'Experiment 2: Generalization Performance Across Subsampling Scenarios ({mode.capitalize()})', fontsize=15, y=1.03)
    fig.tight_layout()

    fig_path = os.path.join(results_dir, f'experiment2_comparison_{mode}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved: {fig_path}")
    plt.show()

# Plot Training vs. Validation Learning Curves across Epochs 
window_size = 15  # Smoothing window for moving average

for mode in modes:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    has_curve_data = False
    
    for scenario, color, ax in zip(scenarios, colors, axes):
        curve_path = os.path.join(results_dir, f'learning_curve_scenario{scenario}_{mode}.csv')
        if os.path.exists(curve_path):
            has_curve_data = True
            df = pd.read_csv(curve_path)
            
            # Compute smoothed rolling means to eliminate jagged noise
            train_smooth = df['train_mse'].rolling(window=window_size, min_periods=1).mean()
            val_smooth = df['val_mse'].rolling(window=window_size, min_periods=1).mean()
            
            # Plot raw data faintly, smoothed trend boldly
            ax.plot(df['epoch'], df['train_mse'], color=color, alpha=0.25, linestyle='-')
            ax.plot(df['epoch'], train_smooth, color=color, linestyle='-', linewidth=2, label=f'Train (Scenario {scenario})')
            
            ax.plot(df['epoch'], df['val_mse'], color='black', alpha=0.25, linestyle='--')
            ax.plot(df['epoch'], val_smooth, color='black', linestyle='--', linewidth=2, label=f'Validation (Scenario {scenario})')
            
            ax.set_ylabel('MSE', fontsize=11)
            ax.set_title(f'Scenario {scenario}', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=9, loc='upper right')
        else:
            print(f"Notice: Could not find learning curve data at {curve_path}.")
            
    if not has_curve_data:
        plt.close(fig)
        continue

    axes[-1].set_xlabel('Epochs', fontsize=12)
    plt.suptitle(f'Experiment 2: Training vs. Validation Learning Curves ({mode.capitalize()} Mode, $h=4$)', fontsize=15, y=0.92)
    fig.tight_layout()
    
    curve_fig_path = os.path.join(results_dir, f'experiment2_learning_curves_smoothed_{mode}.png')
    plt.savefig(curve_fig_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved: {curve_fig_path}")
    plt.show()