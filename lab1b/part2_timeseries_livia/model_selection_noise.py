import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from part2_timeseries_livia.build_dataset import make_embedding, split_data, add_noise
from part2_timeseries_livia.generate_mackey_glass import mackey_glass

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras.regularizers import l2

# From grid search results
CLEAN_BEST_ARCH = (5, 4)
CLEAN_WORST_ARCH = (3, 2)

def train_for_selection(X_train_noisy, Y_train, X_val, Y_val, nh1, nh2, l2_reg, epochs=2000):
    """Phase 1: Fixed epochs to purely evaluate the effect of architecture and lambda."""
    keras.utils.set_random_seed(42) # Fixed seed for fair selection comparison
                                
    model = keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(nh1, activation='tanh', kernel_regularizer=l2(l2_reg)),    
        keras.layers.Dense(nh2, activation='tanh', kernel_regularizer=l2(l2_reg)),
        keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mse'])
    history = model.fit(X_train_noisy, Y_train, validation_data=(X_val, Y_val), 
                        epochs=epochs, batch_size=len(X_train_noisy), verbose=0)
    
    return min(history.history['val_mse']) # Return best validation error achieved

def train_for_final_comparison(X_train_noisy, Y_train, X_val, Y_val, X_test, Y_test, nh1, nh2, l2_reg, seed):
    """Phase 2: Full training with Early Stopping on the validation set."""
    keras.utils.set_random_seed(seed)
                                
    model = keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(nh1, activation='tanh', kernel_regularizer=l2(l2_reg)),    
        keras.layers.Dense(nh2, activation='tanh', kernel_regularizer=l2(l2_reg)),
        keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['mse'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=3000, restore_best_weights=True)

    model.fit(X_train_noisy, Y_train, validation_data=(X_val, Y_val), 
              epochs=10000, batch_size=len(X_train_noisy), callbacks=[early_stop], verbose=0)
    
    test_mse = model.evaluate(X_test, Y_test, verbose=0)[1]
    return test_mse, model

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    X_clean = mackey_glass(n_points=1600)
    X_embedded, Y_embedded = make_embedding(X_clean, t_start=301, t_end=1500)
    X_train_clean, Y_train, X_val, Y_val, X_test, Y_test = split_data(X_embedded, Y_embedded)
    
    SIGMA = 0.09
    X_train_noisy = add_noise(X_train_clean, sigma=SIGMA, seed=42)

    # ==========================================
    # PHASE 1: MODEL SELECTION (Validation Set)
    # ==========================================
    print(f"--- Phase 1: Model Selection (Sigma = {SIGMA}) ---")
    nh1_grid = [3, 4, 5]
    nh2_grid = [2, 4, 6]
    lambda_grid = [0.0, 0.001, 0.01, 0.1]
    
    best_val_mse = float('inf')
    best_config = None

    for n1 in nh1_grid:
        for n2 in nh2_grid:
            for l2_val in lambda_grid:
                val_mse = train_for_selection(X_train_noisy, Y_train, X_val, Y_val, n1, n2, l2_val)
                print(f"Arch {n1}x{n2}, Lambda {l2_val}: Val MSE = {val_mse:.5f}")
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_config = (n1, n2, l2_val)

    print(f"\n=> WINNER: Architecture {best_config[0]}x{best_config[1]} with Lambda {best_config[2]} (Val MSE: {best_val_mse:.5f})")

    # ==========================================
    # PHASE 2: FINAL COMPARISON (Test Set)
    # ==========================================
    print(f"\n--- Phase 2: Final Test Set Evaluation ---")
    
    models_to_test = {
        "New Best Model (Noisy)": best_config,
        f"Clean Best ({CLEAN_BEST_ARCH[0]}x{CLEAN_BEST_ARCH[1]}, L2=0)": (CLEAN_BEST_ARCH[0], CLEAN_BEST_ARCH[1], 0.0),
        f"Clean Worst ({CLEAN_WORST_ARCH[0]}x{CLEAN_WORST_ARCH[1]}, L2=0)": (CLEAN_WORST_ARCH[0], CLEAN_WORST_ARCH[1], 0.0)
    }

    N_SEEDS = 5
    results_summary = []
    trained_models = {} # Store the last seeded model for plotting

    for name, config in models_to_test.items():
        print(f"Retraining {name} across {N_SEEDS} seeds...")
        test_mses = []
        for seed in range(N_SEEDS):
            test_mse, model = train_for_final_comparison(X_train_noisy, Y_train, X_val, Y_val, X_test, Y_test, 
                                                         config[0], config[1], config[2], seed)
            test_mses.append(test_mse)
            if seed == 0: trained_models[name] = model # Save one for plotting
            
        mean_test_mse = np.mean(test_mses)
        std_test_mse = np.std(test_mses)
        results_summary.append({'Model': name, 'Mean_Test_MSE': mean_test_mse, 'Std_Test_MSE': std_test_mse})
        print(f"  -> Mean Test MSE: {mean_test_mse:.5f} +/- {std_test_mse:.5f}")

    # Save Results
    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(os.path.join(results_dir, 'model_selection_noise_comparison.csv'), index=False)

    # ==========================================
    # PHASE 3: PLOTTING THE COMPARISON
    # ==========================================
    time_axis = np.arange(len(Y_test))
    plt.figure(figsize=(14, 6))
    plt.plot(time_axis, Y_test, color='black', linewidth=2, label='True Target Series (Clean)')

    colors = ['tab:blue', 'tab:orange', 'tab:red']
    styles = ['--', '-.', ':']
    
    for i, (name, model) in enumerate(trained_models.items()):
        preds = model.predict(X_test, verbose=0).flatten()
        plt.plot(time_axis, preds, color=colors[i], linestyle=styles[i], linewidth=1.5, alpha=0.9, label=name)

    plt.title(f'Mackey-Glass Test Set Predictions (Trained on Noisy Data, sigma=0.09)', fontsize=14)
    plt.xlabel('Test Set Time Step (Relative)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    fig_path = os.path.join(results_dir, 'noisy_test_set_predictions.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {fig_path}")
    plt.show()

if __name__ == "__main__":
    main()