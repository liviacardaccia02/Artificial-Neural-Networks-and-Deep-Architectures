import numpy as np
from shared.activations import phi, phi_prime
from typing import List

class MLP:
    """
    Multi-Layer Perceptron with backpropagation and momentum.

    Attributes:
        layers (List[int]): List of integers representing the number of neurons in each layer.
        weights (List[np.ndarray]): List of weight matrices for each layer.
        prev_dW (List[np.ndarray]): List of previous weight updates for momentum.
        hins (List[np.ndarray]): List of hidden layer inputs.
        houts (List[np.ndarray]): List of hidden layer outputs.
        input (np.ndarray): The input to the network.
        oin (np.ndarray): The output of the last hidden layer before the final output.
        out (np.ndarray): The final output of the network.
        deltas (List[np.ndarray]): List of delta values for each layer during backpropagation.

    Methods:
        forward(X): Performs a forward pass through the network.
        backward(T): Performs backpropagation to compute the deltas.
        update_weights(eta, alpha): Updates the weights using the computed deltas and momentum.
        train(X, T, eta, alpha, epochs, mode="batch", X_val=None, T_val=None): Trains the network using the specified mode (batch or sequential).
    
    """
    def __init__(self, layers: List[int], seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            
        self.layers = layers
        self.weights = []
        
        for i in range(len(layers) - 1):
            # Initialize weights with small random values
            W = np.random.normal(0, 1/np.sqrt(layers[i]), (layers[i + 1], layers[i] + 1))  # +1 for bias
            self.weights.append(W)
        
        self.prev_dW = [np.zeros_like(W) for W in self.weights]
        
        for i in self.weights:
            print(f"Weight shape: {i.shape}")
        
    def forward(self, X):
        self.hins = []
        self.houts = []
        
        self.input = X
        current = X
        
        self.oin = None
        self.out = None
        
        X_biased = np.vstack((X, np.ones((1, X.shape[1])))) 
        self.hins.append(X_biased)
        
        for i in range(len(self.weights)-1):
            W = self.weights[i]
            
            current_biased = np.vstack(
                (current, np.ones((1, current.shape[1])))
            )
            
            hin = np.dot(W, current_biased)
            hout = phi(hin)
            
            self.hins.append(hin)
            self.houts.append(hout)
            
            current = hout   
        
        current_biased = np.vstack(
            (current, np.ones((1, current.shape[1])))
        )

        self.oin = np.dot(self.weights[-1], current_biased)
        self.out = phi(self.oin)

        self.hins.append(self.oin)
        self.houts.append(self.out)

        return self.out
    
    def backward(self, T):
        self.deltas = [None] * len(self.weights)
        
        self.deltas[-1] = (
            (self.out - T) * phi_prime(self.out)
        )
        
        for i in reversed(range(len(self.weights) - 1)):
            W_next = self.weights[i + 1]
            delta_next = self.deltas[i + 1]

            delta_h = (
                np.dot(W_next[:, :-1].T, delta_next)
                * phi_prime(self.houts[i])
            )

            self.deltas[i] = delta_h

        return self.deltas
    
    def update_weights(self, eta, alpha):
        for i in range(len(self.weights)):
            if i == 0:
                previous_output = self.input
            else:
                previous_output = self.houts[i - 1]
                
            previous_output_biased = np.vstack((previous_output, np.ones((1, previous_output.shape[1]))))
            
            delta = self.deltas[i]
            dW = np.dot(delta, previous_output_biased.T) / previous_output_biased.shape[1]
            
            update = eta * dW + alpha * self.prev_dW[i]
            self.weights[i] -= update
            self.prev_dW[i] = update
            
            
    def train(self, X, T, eta, alpha, epochs, mode="batch", X_val=None, T_val=None):
        history = {'train_mse': [], 'train_misclass': [], 'val_mse': [], 'val_misclass': []}
        
        for epoch in range(epochs):
            if mode == "batch":
                out = self.forward(X)
                train_loss = np.mean((out - T) ** 2)
                train_misclass = np.mean(np.sign(out) != np.sign(T))
                
                self.backward(T)
                self.update_weights(eta, alpha)
                
            elif mode == "sequential":
                epoch_loss_sum = 0
                epoch_misclass_sum = 0
                random_indices = np.random.permutation(X.shape[1])

                if T.ndim == 1:
                    T = T.reshape(1, -1)

                for i in random_indices:
                    x_i = X[:, i:i+1]
                    t_i = T[:, i:i+1]
                    out = self.forward(x_i)
                    
                    epoch_loss_sum += np.mean((out - t_i) ** 2)
                    epoch_misclass_sum += np.mean(np.sign(out) != np.sign(t_i))
                    
                    self.backward(t_i)
                    self.update_weights(eta, alpha)
                
                train_loss = epoch_loss_sum / X.shape[1]
                train_misclass = epoch_misclass_sum / X.shape[1]
            
            history['train_mse'].append(train_loss)
            history['train_misclass'].append(train_misclass)
            
            if X_val is not None and T_val is not None:
                val_out = self.forward(X_val)
                val_loss = np.mean((val_out - T_val) ** 2)
                val_misclass = np.mean(np.sign(val_out) != np.sign(T_val))
                
                history['val_mse'].append(val_loss)
                history['val_misclass'].append(val_misclass)
        
        return history