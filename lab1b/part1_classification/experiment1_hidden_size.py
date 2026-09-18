import numpy as np
import pandas as pd
import os
from part1_classification.generate_data import generate_non_linearly_separable_data
from shared.mlp import MLP


SEED = 42
EPOCH = 1000
ETA = 0.1
ALPHA = 0.9
HIDDEN_SIZE = [1, 2, 3, 4, 6, 8, 12, 16, 20]

X, T, _, _ = generate_non_linearly_separable_data(n=100, seed=SEED)

results = []
for h in HIDDEN_SIZE:
    mlp = MLP(layers=[2, h, 1], seed=SEED)
    history = mlp.train(X, T, eta=ETA, alpha=ALPHA, epochs=EPOCH, mode="batch")
    
    final_train_mse = history['train_mse'][-1]
    final_train_misclass = history['train_misclass'][-1]
    
    results.append({
        'hidden_size': h,
        'final_train_mse': final_train_mse,
        'final_train_misclass': final_train_misclass
    })

    print(f"Hidden Size: {h}, Train MSE: {final_train_mse:.4f}, Train Misclass: {final_train_misclass:.4f}")


# Save raw data as a csv file
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, 'results')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_path = os.path.join(results_dir, 'experiment1_hidden_size_results.csv')
df = pd.DataFrame(results)
df.to_csv(file_path, index=False)