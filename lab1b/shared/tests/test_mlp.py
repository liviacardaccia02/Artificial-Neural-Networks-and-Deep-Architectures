# sanity check for the MLP implementation

from shared.mlp import train, MLP
import numpy as np

X = np.array([[-1, -1, 1, 1], 
              [-1, 1, -1, 1]])  # Symmetric Input for XOR
T = np.array([[-1, 1, 1, -1]])  # Symmetric Target

W, V, history = train(X, T, n_hidden=4, eta=0.2, alpha=0.9, epochs=1000, mode="sequential")

assert history['train_misclass'][-1] == 0.0, "Network failed to solve XOR"

mlp = MLP([2, 4, 3, 1])

print(mlp.forward(np.array([[-1], [-1]])))  # Should output values close to T

mlp.train(X, T, eta=0.2, alpha=0.9, epochs=5000, mode="batch")

print(mlp.forward(X))  # Should output values close to T