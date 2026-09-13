"""Section 4.3, task 5 (non-mandatory): model selection at sigma = 0.09.

Three networks are trained FROM SCRATCH on data corrupted with sigma = 0.09 and
compared on the test set:

  A. the best (nh1 x nh2, lambda) chosen by validation error under this noise
  B. the "most suitable" architecture from task 2, with lambda = 0
  C. the "least suitable" architecture from task 2, with lambda = 0

The point: the architecture that wins on clean data need not win on noisy data.
On clean data capacity is nearly free -- extra units can only help fit a
deterministic function. Under noise the surplus flexibility gets spent fitting
the particular noise realisation, so capacity becomes a liability. Weight decay
is what buys that capacity back.

Which subset selects, and why: the VALIDATION set. It is the only block that is
neither used for weight updates nor reserved as the final test. Selecting on
training error would pick whichever model overfits hardest; selecting on test
error would destroy the test set's role as an unbiased generalisation estimate.
For models B and C nothing is selected -- architecture and lambda are already
fixed -- so there the validation set does only one job, driving early stopping.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from build_dataset import add_noise_xy, load_data
from model import build_net, mse, train_torch

SIGMA = 0.09
NH1_GRID = (3, 4, 5)
NH2_GRID = (3, 6, 9)
LAMBDAS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
N_SEEDS = 5
EPOCHS = 60000
PATIENCE = 3000
LR = 0.01


def train_noisy(nh1, nh2, lam, train, val, test, seed, sigma=SIGMA,
                early_stopping=True, epochs=EPOCHS, patience=PATIENCE, lr=LR,
                return_net=False):
    (Xtr, ytr), (Xval, yval), (Xte, yte) = train, val, test
    rng = np.random.default_rng(2000 + seed)
    Xtr_n, ytr_n = add_noise_xy(Xtr, ytr, sigma, rng)

    net = build_net(nh1, nh2, seed=seed)
    info = train_torch(net, Xtr_n, ytr_n, Xval, yval, lam=lam, lr=lr,
                       epochs=epochs, early_stopping=early_stopping, patience=patience)
    out = {
        "nh1": nh1, "nh2": nh2, "lam": lam, "seed": seed, "sigma": sigma,
        "val_mse": mse(net, Xval, yval),
        "test_mse": mse(net, Xte, yte),
        "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"],
    }
    if return_net:
        out["net"] = net
    return out


def select_best(train, val, test, nh1_grid=NH1_GRID, nh2_grid=NH2_GRID,
                lambdas=LAMBDAS, n_seeds=N_SEEDS, sigma=SIGMA, verbose=True):
    """Search (nh1, nh2, lambda) on VALIDATION error only. Test stays untouched."""
    results = []
    for nh1 in nh1_grid:
        for nh2 in nh2_grid:
            for lam in lambdas:
                v = [train_noisy(nh1, nh2, lam, train, val, test, seed=s, sigma=sigma)
                     for s in range(n_seeds)]
                vm = np.array([r["val_mse"] for r in v])
                results.append({"nh1": nh1, "nh2": nh2, "lam": lam,
                                "val_mse_mean": float(vm.mean()),
                                "val_mse_std": float(vm.std(ddof=1))})
                if verbose:
                    print(f"  {nh1}x{nh2} lam={lam:<7g} val {vm.mean():.3e} "
                          f"+/- {vm.std(ddof=1):.1e}")
    return min(results, key=lambda r: r["val_mse_mean"]), results


def compare(candidates, train, val, test, n_seeds=N_SEEDS, sigma=SIGMA):
    """Retrain each (label, nh1, nh2, lam) from scratch on noisy data; report test MSE."""
    out = []
    for label, nh1, nh2, lam in candidates:
        runs = [train_noisy(nh1, nh2, lam, train, val, test, seed=s, sigma=sigma)
                for s in range(n_seeds)]
        te = np.array([r["test_mse"] for r in runs])
        va = np.array([r["val_mse"] for r in runs])
        out.append({"label": label, "nh1": nh1, "nh2": nh2, "lam": lam,
                    "val_mse_mean": float(va.mean()), "val_mse_std": float(va.std(ddof=1)),
                    "test_mse_mean": float(te.mean()), "test_mse_std": float(te.std(ddof=1))})
    return out


def format_table(rows):
    head = f"{'model':>33} {'arch':>6} {'lambda':>8} {'val MSE':>23} {'test MSE':>23}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['label']:>33} {r['nh1']}x{r['nh2']:<4} {r['lam']:>8g} "
            f"{r['val_mse_mean']:>10.3e}+/-{r['val_mse_std']:<10.1e} "
            f"{r['test_mse_mean']:>10.3e}+/-{r['test_mse_std']:<10.1e}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    BEST = tuple(int(v) for v in os.environ.get("BEST_ARCH", "5,6").split(","))
    WORST = tuple(int(v) for v in os.environ.get("WORST_ARCH", "3,2").split(","))

    _, (train, val, test) = load_data()
    print(f"selecting the best model at sigma={SIGMA} on the validation set")
    best, all_results = select_best(train, val, test)
    print(f"\nselected: {best['nh1']}x{best['nh2']} lambda={best['lam']}\n")

    rows = compare(
        [
            (f"A: selected at sigma={SIGMA}", best["nh1"], best["nh2"], best["lam"]),
            ("B: task-2 most suitable, lam=0", BEST[0], BEST[1], 0.0),
            ("C: task-2 least suitable, lam=0", WORST[0], WORST[1], 0.0),
        ],
        train, val, test,
    )
    print(format_table(rows))

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    json.dump({"selected": best, "search": all_results, "comparison": rows},
              open(os.path.join(here, "results", "model_selection_noise.json"), "w"), indent=1)
    print("\nsaved results/model_selection_noise.json")
