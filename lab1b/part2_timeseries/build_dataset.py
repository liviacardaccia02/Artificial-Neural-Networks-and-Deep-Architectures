import numpy as np

T_START = 301
T_END = 1500
LAGS = [-20, -15, -10, -5, 0]
HORIZON = 5 # to predict x(t+5)

def make_embedding(X, t_start=T_START, t_end=T_END, lags=LAGS, horizon=HORIZON):
    """
    Builds the time-delayed embedding matrix X and target vector Y.
    Returns NumPy arrays formatted for Keras (N_samples, N_features).
    """

    X_embedded = []
    Y_embedded = []

    for t in range(t_start, t_end + 1):
        input_vector = [X[t + lag] for lag in lags]
        target_value = X[t + horizon]

        X_embedded.append(input_vector)
        Y_embedded.append(target_value)

    return np.array(X_embedded), np.array(Y_embedded)

def split_data(X_embedded, Y_embedded, n_train=700, n_val=300, n_test=200):
    """
    Splits the dataset into training, validation, and test sets.
    Returns the data for each set.
    """

    n_total = len(X_embedded)
    if n_train + n_val + n_test != n_total:
        raise ValueError(f"Splits sum to {n_train + n_val + n_test}, dataset has {n_total}")

    X_train = (X_embedded[:n_train])
    Y_train = (Y_embedded[:n_train])
    X_val = (X_embedded[n_train:n_train + n_val])
    Y_val = (Y_embedded[n_train:n_train + n_val])
    X_test = (X_embedded[n_train + n_val:])
    Y_test = (Y_embedded[n_train + n_val:])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def add_noise(X, sigma, seed=None):
    """
    Adds zero-mean Gaussian noise to the input features.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise