"""Cross-check: the PyTorch loop and the Keras 3 model agree.

Why this file exists. All the sweeps run on the plain PyTorch loop in
`model.train_torch`, because with a full batch one epoch is a single gradient
step and Keras's per-epoch bookkeeping costs roughly 70x more wall time than
the step itself -- the difference between a 20-minute sweep and a 20-hour one.

But the lab asks us to "identify suitable library functions" for regularisation
and early stopping. Those are `keras.regularizers.l2` and
`keras.callbacks.EarlyStopping`, used in `model.build_keras` /
`model.train_keras`. This script trains the same architecture both ways and
checks the errors land in the same place, so the fast path is justified rather
than just asserted.

The two will not agree to machine precision -- different RNG streams give
different initial weights, and the loss surface has many minima -- so the test
is whether the Keras result falls inside the seed-to-seed spread of the torch
result.

Run:  python compare_backends.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from build_dataset import load_data
from model import (build_keras, build_net, keras_mse, mse, train_keras,
                   train_torch)

NH1, NH2 = 5, 6
LAM = 1e-4
LR = 0.01
EPOCHS = 3000     # modest, so the Keras side finishes in reasonable time
PATIENCE = 3000   # no early stop within the budget: compare like with like
N_SEEDS = 5


def main():
    _, (train, val, test) = load_data()
    (Xtr, ytr), (Xval, yval), (Xte, yte) = train, val, test

    print(f"architecture {NH1}x{NH2}, lambda={LAM}, lr={LR}, {EPOCHS} epochs\n")

    # ---- PyTorch ---------------------------------------------------------
    t0 = time.time()
    tv, tt = [], []
    for s in range(N_SEEDS):
        net = build_net(NH1, NH2, seed=s)
        train_torch(net, Xtr, ytr, Xval, yval, lam=LAM, lr=LR,
                    epochs=EPOCHS, patience=PATIENCE)
        tv.append(mse(net, Xval, yval))
        tt.append(mse(net, Xte, yte))
    t_torch = time.time() - t0
    tv, tt = np.array(tv), np.array(tt)
    print(f"PyTorch  ({N_SEEDS} seeds, {t_torch:.1f}s total, "
          f"{t_torch / N_SEEDS / EPOCHS * 1000:.3f} ms/epoch)")
    print(f"  val  MSE {tv.mean():.3e} +/- {tv.std(ddof=1):.1e}")
    print(f"  test MSE {tt.mean():.3e} +/- {tt.std(ddof=1):.1e}")

    # ---- Keras 3 ---------------------------------------------------------
    t0 = time.time()
    kv, kt = [], []
    for s in range(N_SEEDS):
        m = build_keras(NH1, NH2, lam=LAM, lr=LR, seed=s)
        train_keras(m, Xtr, ytr, Xval, yval, epochs=EPOCHS, patience=PATIENCE)
        kv.append(keras_mse(m, Xval, yval))
        kt.append(keras_mse(m, Xte, yte))
    t_keras = time.time() - t0
    kv, kt = np.array(kv), np.array(kt)
    print(f"\nKeras 3  ({N_SEEDS} seeds, {t_keras:.1f}s total, "
          f"{t_keras / N_SEEDS / EPOCHS * 1000:.3f} ms/epoch)")
    print(f"  val  MSE {kv.mean():.3e} +/- {kv.std(ddof=1):.1e}")
    print(f"  test MSE {kt.mean():.3e} +/- {kt.std(ddof=1):.1e}")

    print(f"\nKeras / PyTorch wall-time ratio: {t_keras / t_torch:.1f}x")

    # ---- verdict ---------------------------------------------------------
    pooled = np.sqrt((tv.var(ddof=1) + kv.var(ddof=1)) / 2)
    gap = abs(kv.mean() - tv.mean())
    print(f"\nvalidation-mean gap {gap:.2e}, pooled seed std {pooled:.2e} "
          f"({gap / pooled:.2f} sigma)")
    if gap <= 2 * pooled:
        print("AGREE: the two implementations are within the seed-to-seed spread.")
    else:
        print("DISAGREE: investigate before trusting the fast path.")


if __name__ == "__main__":
    main()
