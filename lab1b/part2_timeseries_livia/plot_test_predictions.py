import numpy as np
import matplotlib.pyplot as plt
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras

from part2_timeseries_livia.build_dataset import make_embedding, split_data
from part2_timeseries_livia.generate_mackey_glass import mackey_glass
from part2_timeseries_livia.grid_search import build_and_train_model


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    X_raw = mackey_glass(n_points=1600)
    X_embedded, Y_embedded = make_embedding(X_raw, t_start=301, t_end=1500)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X_embedded, Y_embedded, n_train=700, n_val=300, n_test=200)

    # Note: This was obtained from the grid search results, from the grid_search.py script. 
    # The best and worst architectures were determined based on validation MSE.
    best_arch = (5, 4)  
    worst_arch = (3, 2) 

    print(f"Training Most Suitable ({best_arch[0]}x{best_arch[1]})...")
    result_best = build_and_train_model(X_train, Y_train, X_val, Y_val, nh1=best_arch[0], nh2=best_arch[1])
    model_best = result_best['model']
    
    print(f"Training Least Suitable ({worst_arch[0]}x{worst_arch[1]})...")
    result_worst = build_and_train_model(X_train, Y_train, X_val, Y_val, nh1=worst_arch[0], nh2=worst_arch[1])
    model_worst = result_worst['model']

    pred_best = model_best.predict(X_test, verbose=0).flatten()
    pred_worst = model_worst.predict(X_test, verbose=0).flatten()

    # Calculate final Test MSE
    mse_best = np.mean((pred_best - Y_test) ** 2)
    mse_worst = np.mean((pred_worst - Y_test) ** 2)

    print(f"\nFinal Test MSE (Most Suitable {best_arch[0]}x{best_arch[1]}):  {mse_best:.5f}")
    print(f"Final Test MSE (Least Suitable {worst_arch[0]}x{worst_arch[1]}): {mse_worst:.5f}")

    time_axis = np.arange(len(Y_test))

    plt.figure(figsize=(14, 6))

    # Plot True Targets
    plt.plot(time_axis, Y_test, color='black', linewidth=2, label='True Target Series')
    
    # Plot Best Architecture Predictions
    plt.plot(time_axis, pred_best, color='tab:blue', linestyle='--', linewidth=1.5, 
             label=f'Best Arch {best_arch[0]}x{best_arch[1]} (MSE: {mse_best:.4f})')
    
    # Plot Worst Architecture Predictions
    plt.plot(time_axis, pred_worst, color='tab:red', linestyle='-.', linewidth=1.5, alpha=0.8,
             label=f'Worst Arch {worst_arch[0]}x{worst_arch[1]} (MSE: {mse_worst:.4f})')

    plt.title('Mackey-Glass Test Set Predictions: Most vs. Least Suitable Architectures', fontsize=14)
    plt.xlabel('Test Set Time Step (Relative)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    fig_path = os.path.join(results_dir, 'test_set_predictions.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {fig_path}")
    plt.show()

if __name__ == "__main__":
    main()