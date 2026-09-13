"""Three-layer perceptron for Mackey-Glass prediction (lab section 4.2).

"Three-layer perceptron" in this course counts WEIGHT layers, so the network is

    5 inputs -> nh1 hidden -> nh2 hidden -> 1 linear output

Design choices, all defensible at the presentation:

* tanh hidden units. The lab asks for "sigmoidal" units. tanh is sigmoidal, and
  it is exactly the transfer function used in part I: part I's
  phi(x) = 2/(1+e^-x) - 1 equals tanh(x/2). Being zero-centred it is better
  conditioned than the logistic sigmoid, so gradients do not all share a sign.
* LINEAR output. The target x(t+5) lies in about [0.24, 1.38]. A logistic
  sigmoid is bounded in (0,1) and could not even represent the top of that
  range; a saturated sigmoid also has near-zero derivative, so backprop
  multiplies the error by ~0 and upstream layers stop learning. A linear output
  has derivative 1, unbounded range, and makes delta_output = (o - t) exactly.
* Full-batch gradient descent -- the lab asks for batch backprop, so one weight
  update per epoch using all training patterns.
* Adam. The lab warns that a bad learning rate gives premature convergence or
  absurdly long runs; Adam adapts a per-parameter step from the running first
  and second moments of the gradient and removes most of that hand-tuning.
* No input standardisation. Measured, not assumed: inputs already sit in
  [0.2, 1.4], a well-conditioned range for tanh, and standardising changed the
  validation error by less than the seed-to-seed spread. Leaving the data in
  its natural units also keeps the noise levels sigma = 0.05 / 0.15 directly
  interpretable against the signal scale.

TWO IMPLEMENTATIONS LIVE HERE.

`train_torch` (default) is a plain PyTorch loop. `train_keras` is the Keras 3
equivalent, using keras.regularizers.l2 and keras.callbacks.EarlyStopping --
the "suitable library functions" the lab asks us to identify. They are verified
to agree in `compare_backends.py`.

The default is PyTorch for a measured reason: with a full batch, one epoch is a
single gradient step, and Keras's per-epoch bookkeeping dwarfs the step itself.
compare_backends.py measures 20.3 ms/epoch for Keras against 0.58 ms for the
raw torch step -- a 35x difference on a loaded machine, and closer to 70x when
idle. Over the ~1500 training runs in this part that is the difference between
a one-hour sweep and a multi-day one.
"""

import copy
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
import torch.nn as nn

N_INPUTS = 5


# --------------------------------------------------------------------------
# PyTorch implementation (default)
# --------------------------------------------------------------------------

def build_net(nh1, nh2, seed=None):
    """5 -> nh1 (tanh) -> nh2 (tanh) -> 1 (linear)."""
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(N_INPUTS, nh1),
        nn.Tanh(),
        nn.Linear(nh1, nh2),
        nn.Tanh(),
        nn.Linear(nh2, 1),
    )


def l2_penalty(net):
    """Sum of squared WEIGHTS, biases excluded.

    Biases are left out on purpose: penalising a bias only shifts the function
    vertically, it does not make it smoother, and it drags predictions towards
    zero. This matches what keras.regularizers.l2 does via `kernel_regularizer`.
    """
    return sum((m.weight ** 2).sum() for m in net if isinstance(m, nn.Linear))


def train_torch(
    net,
    Xtr,
    ytr,
    Xval,
    yval,
    lam=0.0,
    lr=0.01,
    epochs=10000,
    early_stopping=True,
    patience=1500,
    eval_every=10,
    record_history=False,
):
    """One full-batch training run.

    The loss actually minimised is

        MSE(w) + lam * sum(w^2)

    so the gradient carries an extra 2*lam*w and the update becomes
    w <- (1 - 2*eta*lam) * w - eta * dMSE/dw: every weight shrinks
    multiplicatively on every step and only survives if the data keeps pushing
    it back. That is weight decay, and it is why the weight histogram contracts
    towards zero as lam grows.

    The penalty is added to the OPTIMISED loss only. Every number returned or
    monitored is plain MSE, so models trained with different lam stay
    comparable.

    Returns (history, best_val_mse). `history` is empty unless record_history.
    """
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32).unsqueeze(1)
    Xval_t = torch.as_tensor(Xval, dtype=torch.float32)
    yval_t = torch.as_tensor(yval, dtype=torch.float32).unsqueeze(1)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mse_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    history = {"epoch": [], "train_mse": [], "val_mse": []}

    for epoch in range(epochs):
        opt.zero_grad()
        loss = mse_fn(net(Xtr_t), ytr_t)
        if lam > 0:
            loss = loss + lam * l2_penalty(net)
        loss.backward()
        opt.step()

        if epoch % eval_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                tr = mse_fn(net(Xtr_t), ytr_t).item()
                va = mse_fn(net(Xval_t), yval_t).item()
            if record_history:
                history["epoch"].append(epoch)
                history["train_mse"].append(tr)
                history["val_mse"].append(va)
            if va < best_val:
                best_val, best_epoch = va, epoch
                if early_stopping:
                    best_state = copy.deepcopy(net.state_dict())
            elif early_stopping and epoch - best_epoch >= patience:
                break

    if early_stopping:
        # "restore best weights": the point of early stopping is to KEEP the
        # weights from the best epoch, not the ones you happened to stop on.
        net.load_state_dict(best_state)

    return {"history": history, "best_val": best_val, "best_epoch": best_epoch,
            "epochs_run": epoch + 1}


def predict(net, X):
    with torch.no_grad():
        return net(torch.as_tensor(X, dtype=torch.float32)).numpy().ravel()


def mse(net, X, y):
    """Plain mean squared error from predictions -- never includes the penalty."""
    return float(np.mean((predict(net, X) - y) ** 2))


def get_weights(net):
    """All weight-matrix entries flattened (biases excluded).

    Used for the weight-distribution question in section 4.3 task 4.
    """
    return np.concatenate(
        [m.weight.detach().numpy().ravel() for m in net if isinstance(m, nn.Linear)]
    )


def n_params(nh1, nh2):
    """Trainable parameter count, biases included."""
    return (N_INPUTS * nh1 + nh1) + (nh1 * nh2 + nh2) + (nh2 + 1)


# --------------------------------------------------------------------------
# Keras 3 implementation (cross-check; identifies the library functions the
# lab asks for)
# --------------------------------------------------------------------------

def build_keras(nh1, nh2, lam=0.0, lr=0.01, seed=None):
    import keras

    if seed is not None:
        keras.utils.set_random_seed(seed)
    reg = keras.regularizers.l2(lam) if lam and lam > 0 else None
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(N_INPUTS,)),
            keras.layers.Dense(nh1, activation="tanh", kernel_regularizer=reg),
            keras.layers.Dense(nh2, activation="tanh", kernel_regularizer=reg),
            keras.layers.Dense(1, activation="linear", kernel_regularizer=reg),
        ],
        name=f"mlp_{nh1}x{nh2}",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        # separate metric because `loss` includes the l2 penalty once lam > 0
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


def train_keras(model, Xtr, ytr, Xval, yval, epochs=10000, early_stopping=True,
                patience=1500, verbose=0):
    import keras

    callbacks = []
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_mse", patience=patience,
                restore_best_weights=True, mode="min",
            )
        )
    return model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        batch_size=len(Xtr),   # full batch == batch backprop
        epochs=epochs, shuffle=False, verbose=verbose, callbacks=callbacks,
    )


def keras_mse(model, X, y):
    return float(np.mean((model.predict(X, verbose=0).ravel() - y) ** 2))
