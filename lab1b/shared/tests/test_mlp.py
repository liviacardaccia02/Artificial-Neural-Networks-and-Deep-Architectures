# sanity check for the MLP implementation

from shared.mlp import MLP
import numpy as np

X = np.array([[-1, -1, 1, 1], 
              [-1, 1, -1, 1]])  # Symmetric Input for XOR
T = np.array([[-1, 1, 1, -1]])  # Symmetric Target

mlp = MLP([2, 4, 3, 1])

print(mlp.forward(np.array([[-1], [-1]])))  # Should output values close to T

mlp.train(X, T, eta=0.2, alpha=0.9, epochs=5000, mode="batch")

print(mlp.forward(X))  # Should output values close to T