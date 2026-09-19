import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from part2_timeseries_livia.build_dataset import make_embedding, split_data, add_noise
from part2_timeseries_livia.generate_mackey_glass import mackey_glass

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import torch
from keras.regularizers import l2

def build_and_train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, nh1, nh2, learning_rate=0.01, patience=3000, epochs=10000, seed=None):
    if seed is not None:
        keras.utils.set_random_seed(seed)
                                
    model = keras.Sequential([keras.layers.Input(shape=(5,)),
                              keras.layers.Dense(nh1, activation='tanh'),    
                              keras.layers.Dense(nh2, activation='tanh'),
                              keras.layers.Dense(1, activation='linear')])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=[keras.metrics.MeanSquaredError(name="mse")] 
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=patience,
        restore_best_weights=True,
        mode='min'
    )

    history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=epochs,          
    batch_size=len(X_train), # Full batch descent
    callbacks=[early_stop],
    verbose=0              
    )

    # Calculate Test MSE using the best restored weights
    test_mse = model.evaluate(X_test, Y_test, verbose=0)[1] # index 1 is the 'mse' metric

    return {
        'nh1': nh1,
        'nh2': nh2,
        'seed': seed,
        'val_mse': min(history.history['val_mse']), # Best validation MSE achieved
        'test_mse': test_mse,
        'epochs_run': len(history.history['loss']), # How many epochs it took before stopping
        'model': model  # Return the trained model for further analysis if needed
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    X = mackey_glass(n_points=1600)
    X_embedded, Y_embedded = make_embedding(X, t_start=301, t_end=1500)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_data(X_embedded, Y_embedded, n_train=700, n_val=300, n_test=200)

    # Grid Search Parameters
    nh1_grid = [3, 4, 5]
    nh2_grid = [2, 4, 6]
    N_SEEDS = 10 
    
    all_runs = []

    print("Starting Grid Search...")
    for n1 in nh1_grid:
        for n2 in nh2_grid:
            print(f"Evaluating architecture {n1}x{n2} over {N_SEEDS} seeds...")
            for seed in range(N_SEEDS):
                res = build_and_train_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, n1, n2, seed=seed)
                all_runs.append(res)

    # Save raw runs to CSV
    df_runs = pd.DataFrame(all_runs)
    runs_path = os.path.join(results_dir, 'grid_search_raw_runs.csv')
    df_runs.to_csv(runs_path, index=False)
    
    # Calculate Mean & Variance per architecture, for robustness assessment
    summary = df_runs.groupby(['nh1', 'nh2']).agg(
        val_mse_mean=('val_mse', 'mean'),
        val_mse_std=('val_mse', 'std'),
        test_mse_mean=('test_mse', 'mean'),
        test_mse_std=('test_mse', 'std'),
        avg_epochs=('epochs_run', 'mean')
    ).reset_index()
    
    # Rank by Validation MSE to find Most/Least Suitable
    summary = summary.sort_values(by='val_mse_mean')
    
    summary_path = os.path.join(results_dir, 'grid_search_summary.csv')
    summary.to_csv(summary_path, index=False)
    
    print("\nGrid Search Complete! Summary sorted by Validation MSE:")
    print(summary.to_string(index=False))
    
    most_suitable = summary.iloc[0]
    least_suitable = summary.iloc[-1]
    
    print(f"\nMOST SUITABLE Architecture: {int(most_suitable['nh1'])}x{int(most_suitable['nh2'])} (Val MSE Mean: {most_suitable['val_mse_mean']:.5f})")
    print(f"LEAST SUITABLE Architecture: {int(least_suitable['nh1'])}x{int(least_suitable['nh2'])} (Val MSE Mean: {least_suitable['val_mse_mean']:.5f})")

if __name__ == "__main__":
    main()