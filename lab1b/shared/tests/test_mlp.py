# sanity check for the MLP implementation

from shared.mlp import train
import numpy as np

X = np.array([[-1, -1, 1, 1], 
              [-1, 1, -1, 1]])  # Symmetric Input for XOR
T = np.array([[-1, 1, 1, -1]])  # Symmetric Target

W, V, history = train(X, T, n_hidden=4, eta=0.2, alpha=0.9, epochs=1000, mode="sequential")

assert history['train_misclass'][-1] == 0.0, "Network failed to solve XOR"