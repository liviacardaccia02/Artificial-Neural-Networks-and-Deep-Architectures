import numpy as np
import os
import pandas as pd
from part2_timeseries.build_dataset import make_embedding, split_data, add_noise
from part2_timeseries.generate_mackey_glass import mackey_glass

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras.regularizers import l2

def train_noise_model(X_train_noisy, Y_train, X_val, Y_val, nh1, nh2, l2_reg, epochs=3000, seed=42):
    """Trains a model WITH weight decay and WITHOUT early stopping."""
    keras.utils.set_random_seed(seed)
                                
    model = keras.Sequential([
        keras.layers.Input(shape=(5,)),
        keras.layers.Dense(nh1, activation='tanh', kernel_regularizer=l2(l2_reg)),    
        keras.layers.Dense(nh2, activation='tanh', kernel_regularizer=l2(l2_reg)),
        keras.layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='mse',
        metrics=[keras.metrics.MeanSquaredError(name="mse")] 
    )

    history = model.fit(
        X_train_noisy, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,          
        batch_size=len(X_train_noisy), 
        verbose=0              
    )

    # Calculate Weight Norm (sum of absolute weights) to see lambda's effect
    weight_norm = 0.0
    for layer in model.layers:
        weights = layer.get_weights()
        if weights: # If layer has weights
            weight_norm += np.sum(np.abs(weights[0])) # Add L1 norm of weights

    return {
        'nh1': nh1,
        'nh2': nh2,
        'l2_lambda': l2_reg,
        'val_mse_final': history.history['val_mse'][-1], # Error at the final epoch
        'train_mse_final': history.history['mse'][-1],
        'weight_norm': weight_norm
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    X_clean = mackey_glass(n_points=1600)
    X_embedded, Y_embedded = make_embedding(X_clean, t_start=301, t_end=1500)
    X_train_clean, Y_train, X_val, Y_val, X_test, Y_test = split_data(X_embedded, Y_embedded)

    nh1 = 5 # Fixed from previous best architecture
    nh2_grid = [3, 6, 9]
    sigma_values = [0.05, 0.15] 
    lambda_values = [0.0, 0.001, 0.01, 0.1] # Sweep to see regularisation effects
    N_SEEDS = 5 # Multiple runs for robustness
    
    all_runs = []
    print("Starting Noise & Regularization Experiments...")

    for sigma in sigma_values:
        print(f"\n--- Testing Noise Level Sigma = {sigma} ---")
        
        # Add noise ONLY to the training features for this iteration
        # Note: We re-add noise per seed inside the loop if we want stochastic noise, 
        # but for simplicity, applying it once per sigma level is acceptable here.
        
        for n2 in nh2_grid:
            for l2_val in lambda_values:
                for seed in range(N_SEEDS):
                    print(f"Training model with nh2={n2}, lambda={l2_val}, seed={seed}...")
                    # Generate fresh noise for every seed to ensure robust averaging
                    X_train_noisy = add_noise(X_train_clean, sigma=sigma, seed=seed)
                    
                    res = train_noise_model(X_train_noisy, Y_train, X_val, Y_val, nh1, n2, l2_val, epochs=3000, seed=seed)
                    res['sigma'] = sigma
                    all_runs.append(res)
                print(f"Finished nh2={n2}, lambda={l2_val}")

    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv(os.path.join(results_dir, 'noise_experiment_raw.csv'), index=False)
    
    # Aggregate to get the mean validation error and mean weight norm across the seeds
    summary = df_runs.groupby(['sigma', 'nh2', 'l2_lambda']).agg(
        val_mse_mean=('val_mse_final', 'mean'),
        train_mse_mean=('train_mse_final', 'mean'),
        weight_norm_mean=('weight_norm', 'mean')
    ).reset_index()
    
    summary_path = os.path.join(results_dir, 'noise_experiment_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nExperiment complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()