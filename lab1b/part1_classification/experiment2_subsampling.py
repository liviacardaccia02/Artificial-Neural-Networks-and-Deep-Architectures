import numpy as np
import pandas as pd
import os
from part1_classification.generate_data import generate_non_linearly_separable_data, subsample_scenario
from shared.mlp import MLP

SEED = 42
EPOCH = 1000
ETA = 0.1
ALPHA = 0.9
HIDDEN_SIZE = [2, 4, 6, 8, 12, 16, 20]
CURVE_HIDDEN_SIZE = 4

X, T, classA, classB = generate_non_linearly_separable_data(n=100, seed=SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for mode in ['batch', 'sequential']:
    for scenario in [1, 2, 3]:
        X_train, T_train, X_val, T_val = subsample_scenario(classA, classB, scenario, seed=SEED)
        
        results = []
        for h in HIDDEN_SIZE:
            mlp = MLP(layers=[2, h, 1], seed=SEED)
            history = mlp.train(X_train, T_train, eta=ETA, alpha=ALPHA, epochs=EPOCH, mode=mode, X_val=X_val, T_val=T_val)
            
            final_train_mse = history['train_mse'][-1]
            final_train_misclass = history['train_misclass'][-1]
            
            results.append({
                'hidden_size': h,
                'final_train_mse': final_train_mse,
                'final_train_misclass': final_train_misclass,
                'val_mse': history['val_mse'][-1],
                'val_misclass': history['val_misclass'][-1]
            })

            print(f"Scenario {scenario} | Mode: {mode} | Hidden Size: {h} | Train MSE: {history['train_mse'][-1]:.4f} | Val MSE: {history['val_mse'][-1]:.4f}")

            if h == CURVE_HIDDEN_SIZE:
                df_curve = pd.DataFrame({
                    'epoch': np.arange(EPOCH),
                    'train_mse': history['train_mse'],
                    'val_mse': history['val_mse'],
                    'train_misclass': history['train_misclass'],
                    'val_misclass': history['val_misclass']
                })
                curve_path = os.path.join(results_dir, f'learning_curve_scenario{scenario}_{mode}.csv')
                df_curve.to_csv(curve_path, index=False)

        file_path = os.path.join(results_dir, f'experiment2_subsampling_scenario{scenario}_results_{mode}.csv')
        df = pd.DataFrame(results)
        df.to_csv(file_path, index=False)