import numpy as np
from shared.activations import phi, phi_prime

def forward(X, W, V): 
    # Add bias term to input
    X_biased = np.vstack((X, np.ones((1, X.shape[1])))) 
    # Hidden layer input 
    hin = np.dot(W, X_biased) 
    # Hidden layer output 
    hout = phi(hin)  

    # Add bias term to hidden layer output
    hout = np.vstack((hout, np.ones((1, hout.shape[1]))))  
    # Output layer input
    oin = np.dot(V, hout)  
    # Output layer output
    out = phi(oin) 

    return hin, hout, oin, out

def backward(out, T, hout, W, V):
    # Calculate output layer error
    delta_o = (out - T) * phi_prime(out)

    # Calculate hidden layer error
    delta_h = np.dot(V[:, :-1].T, delta_o) * phi_prime(hout[:-1, :])  # Exclude bias term from hout

    return delta_o, delta_h

def update_weights(W, V, dW, dV, X, hout, delta_h, delta_o, eta, alpha):
    # Update weights with momentum
    dW = (dW * alpha) - (np.dot(delta_h, np.vstack((X, np.ones((1, X.shape[1])))).T) / X.shape[1]) * (1 - alpha)
    dV = (dV * alpha) - (np.dot(delta_o, hout.T) / hout.shape[1]) * (1 - alpha)

    W += eta * dW
    V += eta * dV

    return W, V, dW, dV

def train(X, T, n_hidden, eta, alpha, epochs, mode="batch", X_val=None, T_val=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_input = X.shape[0]
    n_output = T.shape[0]
    history = {'train_mse': [], 'train_misclass': [], 'val_mse': [], 'val_misclass': []}

    # Initialize weights
    W = np.random.normal(0, 1/np.sqrt(n_input), (n_hidden, n_input + 1))  # +1 for bias
    V = np.random.normal(0, 1/np.sqrt(n_hidden), (n_output, n_hidden + 1))  # +1 for bias

    dW = np.zeros_like(W)
    dV = np.zeros_like(V)

    for epoch in range(epochs):
        if mode == "batch":
            hin, hout, oin, out = forward(X, W, V)

            # Calculate epoch error using pre-update predictions
            train_loss = np.mean((out - T) ** 2)
            train_misclass = np.mean(np.sign(out) != np.sign(T))

            delta_o, delta_h = backward(out, T, hout, W, V)
            W, V, dW, dV = update_weights(W, V, dW, dV, X, hout, delta_h, delta_o, eta, alpha)
            
        elif mode == "sequential":
            epoch_loss_sum = 0
            epoch_misclass_sum = 0
            random_indices = np.random.permutation(X.shape[1])
            for i in random_indices:
                x_i = X[:, i:i+1]
                t_i = T[:, i:i+1]
                hin, hout, oin, out = forward(x_i, W, V)

                # Accumulate running error for this sample
                epoch_loss_sum += np.mean((out - t_i) ** 2)
                epoch_misclass_sum += np.mean(np.sign(out) != np.sign(t_i))

                delta_o, delta_h = backward(out, t_i, hout, W, V)
                W, V, dW, dV = update_weights(W, V, dW, dV, x_i, hout, delta_h, delta_o, eta, alpha)

            # Average the accumulated error over the number of samples
            train_loss = epoch_loss_sum / X.shape[1]
            train_misclass = epoch_misclass_sum / X.shape[1]

        # Store history
        history['train_mse'].append(train_loss)
        history['train_misclass'].append(train_misclass)

        # Optionally evaluate on validation set
        if X_val is not None and T_val is not None:
            _, _, _, val_out = forward(X_val, W, V)
            val_loss = np.mean((val_out - T_val) ** 2)
            val_misclass = np.mean(np.sign(val_out) != np.sign(T_val))
            history['val_misclass'].append(val_misclass)
            history['val_mse'].append(val_loss)

    return W, V, history