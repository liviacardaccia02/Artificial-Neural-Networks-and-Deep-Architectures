"""Time-delay embedding and the train/validation/test split (lab section 4.1).

A feed-forward MLP has no memory, so the *time* structure has to be put into
the input vector by hand. That is the time-delay (or "time-lagged") embedding:
each sample is a short window of the past, and the target is a value in the
future.

    input(t)  = [x(t-20), x(t-15), x(t-10), x(t-5), x(t)]   -> 5 inputs
    target(t) =  x(t+5)                                      -> 1 output

Takens' embedding theorem is the justification: for a deterministic dynamical
system, a delay vector of sufficient dimension reconstructs a state equivalent
to the true (hidden) state. Once the state is reconstructed, the next value is
a *deterministic function* of that vector -- exactly the kind of thing an MLP
can approximate.

The split into train/validation/test must be into CONSECUTIVE blocks, not a
random shuffle. Neighbouring windows overlap heavily (input(t) and input(t+1)
share four of five components), so a random split would leak almost-identical
samples into validation and test and give an optimistic, meaningless error.
"""

import numpy as np

LAGS = (-20, -15, -10, -5, 0)  # five inputs
HORIZON = 5                    # predict x(t+5)
T_START, T_END = 301, 1500     # 1200 samples, as fixed by the lab


def make_embedding(x, t_start=T_START, t_end=T_END, lags=LAGS, horizon=HORIZON):
    """Build (X, y) with X of shape (N, 5) and y of shape (N,).

    Keras expects samples along rows -- the opposite of the column convention
    used in part I and in the MATLAB snippets in the lab PDF. Transposing here
    once is cheaper than fighting the library everywhere else.
    """
    t = np.arange(t_start, t_end + 1)
    if t.max() + horizon >= len(x):
        raise ValueError(
            f"series too short: need at least {t.max() + horizon + 1} points, got {len(x)}"
        )
    X = np.stack([x[t + lag] for lag in lags], axis=1)
    y = x[t + horizon]
    return X.astype(np.float32), y.astype(np.float32)


def split_blocks(X, y, n_train=800, n_val=200, n_test=200):
    """Consecutive, non-overlapping blocks. Last n_test samples are the test set.

    TEAM DECISION: 800 / 200 / 200. The lab fixes only the 200-sample test
    block; the 800/200 split of the remaining 1000 is ours. 200 validation
    samples cover roughly two full oscillations of the series, which is enough
    for the validation error to be a stable model-selection signal, while 800
    training samples keep the training set large relative to the <=100 weights
    of the biggest network in the grid.
    """
    n = len(X)
    if n_train + n_val + n_test != n:
        raise ValueError(f"splits sum to {n_train + n_val + n_test}, dataset has {n}")
    i, j = n_train, n_train + n_val
    return (X[:i], y[:i]), (X[i:j], y[i:j]), (X[j:], y[j:])


def add_noise(y, sigma, rng):
    """Zero-mean Gaussian noise, used in section 4.3 task 4."""
    if sigma <= 0:
        return y.copy()
    return (y + rng.normal(0.0, sigma, size=y.shape)).astype(np.float32)


def add_noise_xy(X, y, sigma, rng):
    """Corrupt inputs and targets with the same noise level.

    The measurement interpretation: we never observe the clean series, so every
    lagged value in the input window is corrupted too, not just the target.
    """
    if sigma <= 0:
        return X.copy(), y.copy()
    Xn = (X + rng.normal(0.0, sigma, size=X.shape)).astype(np.float32)
    yn = (y + rng.normal(0.0, sigma, size=y.shape)).astype(np.float32)
    return Xn, yn


def load_data(n_points=1600, **split_kwargs):
    """Convenience: series -> embedding -> blocks, in one call."""
    from generate_mackey_glass import mackey_glass

    x = mackey_glass(n_points=n_points)
    X, y = make_embedding(x)
    return x, split_blocks(X, y, **split_kwargs)


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    x, (train, val, test) = load_data()
    for name, (Xs, ys) in zip(("train", "val", "test"), (train, val, test)):
        print(f"{name:5s} X={Xs.shape} y={ys.shape}  y range [{ys.min():.3f}, {ys.max():.3f}]")
