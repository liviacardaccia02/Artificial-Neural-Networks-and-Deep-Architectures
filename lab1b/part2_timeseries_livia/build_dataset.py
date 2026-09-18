import numpy as np

T_START = 301
T_END = 1500
LAGS = [-20, -15, -10, -5, 0]

def make_embedding(x, t_start=T_START, t_end=T_END):
    """
    Builds the time-delayed embedding matrix X and target vector Y.
    Returns NumPy arrays formatted for Keras (N_samples, N_features).
    """

    X_embedded = []
    Y_embedded = []

    for t in range(t_start, t_end + 1):
        input_vector = [x[t + lag] for lag in LAGS]
        target_value = x[t + 5]

        X_embedded.append(input_vector)
        Y_embedded.append(target_value)

    return np.array(X_embedded), np.array(Y_embedded)

def split_data(X_embedded, Y_embedded, n_train=700, n_val=300, n_test=200):
    """
    Splits the dataset into training, validation, and test sets.
    Returns tuples of (X, Y) for each set.
    """

    n_total = len(X_embedded)
    if n_train + n_val + n_test != n_total:
        raise ValueError(f"Splits sum to {n_train + n_val + n_test}, dataset has {n_total}")

    train_data = (X_embedded[:n_train], Y_embedded[:n_train])
    val_data = (X_embedded[n_train:n_train + n_val], Y_embedded[n_train:n_train + n_val])
    test_data = (X_embedded[n_train + n_val:], Y_embedded[n_train + n_val:])

    return train_data, val_data, test_data

def add_noise(X, sigma, seed=None):
    """
    Adds zero-mean Gaussian noise to the input features.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise