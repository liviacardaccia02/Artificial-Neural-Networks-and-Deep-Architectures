"""Section 4.3, tasks 1-3: architecture grid search and test evaluation.

Task 2 -- evaluate the Cartesian product nh1 in {3,4,5} x nh2 in {2,4,6} with
early stopping on the validation set, pick the most and least suitable
architecture by VALIDATION error, and report how robust that choice is to the
random weight initialisation.

Task 3 -- report the TEST error of those two architectures.

The discipline that matters: the test set is read only at the end, after every
decision is made. The moment you select on test error, test error stops being
an unbiased estimate of generalisation.

Each configuration is trained N_SEEDS times, because the run-to-run spread from
random initialisation is itself one of the lab's required results. A single run
of 4x6 beating a single run of 5x4 tells you nothing.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from build_dataset import load_data
from model import build_net, mse, n_params, train_torch

NH1_GRID = (3, 4, 5)
NH2_GRID = (2, 4, 6)
N_SEEDS = 10       # TEAM DECISION: stable mean, meaningful std
LR = 0.01
# Generous ceiling, and early stopping decides the actual duration per model.
# Measured reason: at a fixed 10k budget the small architectures had simply not
# converged yet, and the grid then ranked CONVERGENCE SPEED rather than
# capacity -- 3x2 kept improving until ~48k epochs and nearly caught 5x6. With
# this ceiling every architecture reaches its own plateau before it is judged.
EPOCHS = 60000
PATIENCE = 3000


def run_config(nh1, nh2, train, val, test, seed, lam=0.0, lr=LR,
               epochs=EPOCHS, patience=PATIENCE, return_net=False):
    (Xtr, ytr), (Xval, yval), (Xte, yte) = train, val, test
    net = build_net(nh1, nh2, seed=seed)
    info = train_torch(net, Xtr, ytr, Xval, yval, lam=lam, lr=lr,
                       epochs=epochs, patience=patience)
    out = {
        "nh1": nh1, "nh2": nh2, "seed": seed, "lam": lam,
        "train_mse": mse(net, Xtr, ytr),
        "val_mse": mse(net, Xval, yval),
        "test_mse": mse(net, Xte, yte),
        "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"],
        "n_params": n_params(nh1, nh2),
    }
    if return_net:
        out["net"] = net
    return out


def grid_search(train, val, test, nh1_grid=NH1_GRID, nh2_grid=NH2_GRID,
                n_seeds=N_SEEDS, verbose=True, **kw):
    runs = []
    for nh1 in nh1_grid:
        for nh2 in nh2_grid:
            batch = [run_config(nh1, nh2, train, val, test, seed=s, **kw)
                     for s in range(n_seeds)]
            runs.extend(batch)
            if verbose:
                v = np.array([b["val_mse"] for b in batch])
                print(f"  {nh1}x{nh2}  val MSE {v.mean():.3e} +/- {v.std(ddof=1):.3e}")
    return runs


def summarise(runs):
    """Mean and std over seeds, per architecture.

    The lab asks for the first AND second statistical moment. The std across
    seeds IS the robustness-to-initialisation answer: small std means the
    architecture finds an equally good minimum whatever the starting weights;
    large std means the number you report depends on luck.
    """
    keys = sorted({(r["nh1"], r["nh2"]) for r in runs})
    rows = []
    for nh1, nh2 in keys:
        sel = [r for r in runs if (r["nh1"], r["nh2"]) == (nh1, nh2)]
        row = {"nh1": nh1, "nh2": nh2, "n_params": sel[0]["n_params"], "n_seeds": len(sel)}
        for field in ("train_mse", "val_mse", "test_mse", "best_epoch", "epochs_run"):
            vals = np.array([r[field] for r in sel], dtype=float)
            row[f"{field}_mean"] = vals.mean()
            row[f"{field}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def pick_best_worst(rows):
    """Most and least suitable architecture, ranked by MEAN VALIDATION error."""
    ranked = sorted(rows, key=lambda r: r["val_mse_mean"])
    return ranked[0], ranked[-1]


def format_table(rows, highlight=()):
    head = (f"{'arch':>6} {'par':>4} {'train MSE':>22} {'val MSE':>22} "
            f"{'test MSE':>22} {'best ep':>9}")
    lines = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda q: q["val_mse_mean"]):
        tag = "*" if (r["nh1"], r["nh2"]) in highlight else " "
        lines.append(
            f"{r['nh1']}x{r['nh2']}{tag:>3} {r['n_params']:>4} "
            f"{r['train_mse_mean']:>9.2e}+/-{r['train_mse_std']:<10.1e} "
            f"{r['val_mse_mean']:>9.2e}+/-{r['val_mse_std']:<10.1e} "
            f"{r['test_mse_mean']:>9.2e}+/-{r['test_mse_std']:<10.1e} "
            f"{r['best_epoch_mean']:>9.0f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    _, (train, val, test) = load_data()
    print(f"train {train[0].shape}  val {val[0].shape}  test {test[0].shape}")
    print(f"grid: {len(NH1_GRID) * len(NH2_GRID)} architectures x {N_SEEDS} seeds\n")

    runs = grid_search(train, val, test)
    rows = summarise(runs)
    best, worst = pick_best_worst(rows)

    print()
    print(format_table(rows, highlight=((best["nh1"], best["nh2"]),
                                        (worst["nh1"], worst["nh2"]))))
    print()
    print(f"most suitable : {best['nh1']}x{best['nh2']}  "
          f"val {best['val_mse_mean']:.3e} +/- {best['val_mse_std']:.1e}  "
          f"test {best['test_mse_mean']:.3e} +/- {best['test_mse_std']:.1e}")
    print(f"least suitable: {worst['nh1']}x{worst['nh2']}  "
          f"val {worst['val_mse_mean']:.3e} +/- {worst['val_mse_std']:.1e}  "
          f"test {worst['test_mse_mean']:.3e} +/- {worst['test_mse_std']:.1e}")
