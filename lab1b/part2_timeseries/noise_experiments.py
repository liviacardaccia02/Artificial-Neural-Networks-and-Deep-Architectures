"""Section 4.3, task 4: noise, second-hidden-layer size, and weight decay.

Setup fixed by the lab:
  * zero-mean Gaussian noise added to the TRAINING data, sigma = 0.05 and 0.15
  * nh1 fixed to the first-hidden-layer size of the most suitable architecture
  * nh2 varied over {3, 6, 9}
  * L2 weight decay with constant lambda
  * NO early stopping

(The PDF prints "nh2 = 3, 6, 9 6" -- the trailing 6 is a footnote marker, not a
number. The grid is 3, 6, 9, consistent with "max 9 nodes per hidden layer" in
section 4.2.)

Two choices that must be stated in the report:

1. Noise is added to the training block only; validation and test stay clean.
   That is the literal reading of "add noise to training data", and it is the
   more informative one: a noisy validation set would carry an irreducible
   error floor of sigma^2 (0.0225 at sigma = 0.15) that is fifty times larger
   than the model differences we are trying to measure, and it would swamp
   them. With a clean validation set, validation MSE measures directly how much
   of the TRUE dynamics each model recovered from corrupted observations --
   exactly what the lambda question is about.

2. Noise corrupts inputs and targets together. In the measurement
   interpretation the whole observed series is noisy, and the lagged values in
   the input window are simply earlier observations of that same series.

Why early stopping is dropped: early stopping and weight decay are both
regularisers competing for the same job. Keeping both would confound them --
you could not tell whether a good result came from lambda or from stopping
early. With a fixed epoch budget for every run, lambda is the only thing that
varies.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from build_dataset import add_noise_xy, load_data
from model import build_net, get_weights, mse, train_torch

SIGMAS = (0.0, 0.05, 0.15)
NH2_GRID = (3, 6, 9)
# TEAM DECISION: logarithmic sweep. Weight decay acts on the SCALE of lambda,
# so a linear grid would spend most of its points in a region where nothing
# changes.
LAMBDAS = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
N_SEEDS = 5
EPOCHS = 20000  # fixed budget, identical for every run, no early stopping
LR = 0.01


def run_noise_config(nh1, nh2, sigma, lam, train, val, test, seed,
                     epochs=EPOCHS, lr=LR, keep_weights=True):
    """Corrupt the training block, train for a fixed budget, evaluate on clean data."""
    (Xtr, ytr), (Xval, yval), (Xte, yte) = train, val, test
    rng = np.random.default_rng(1000 + seed)
    Xtr_n, ytr_n = add_noise_xy(Xtr, ytr, sigma, rng)

    net = build_net(nh1, nh2, seed=seed)
    train_torch(net, Xtr_n, ytr_n, Xval, yval, lam=lam, lr=lr,
                epochs=epochs, early_stopping=False)

    w = get_weights(net)
    rec = {
        "nh1": nh1, "nh2": nh2, "sigma": sigma, "lam": lam, "seed": seed,
        # error on the NOISY data the network actually saw
        "train_mse_noisy": mse(net, Xtr_n, ytr_n),
        # error on the SAME inputs against CLEAN targets: how much of the true
        # function was recovered rather than how much noise was memorised
        "train_mse_clean": mse(net, Xtr, ytr),
        "val_mse": mse(net, Xval, yval),
        "test_mse": mse(net, Xte, yte),
        "w_std": float(w.std()),
        "w_absmean": float(np.abs(w).mean()),
        "w_absmax": float(np.abs(w).max()),
    }
    if keep_weights:
        rec["weights"] = w
    return rec


def sweep(nh1, train, val, test, sigmas=SIGMAS, nh2_grid=NH2_GRID,
          lambdas=LAMBDAS, n_seeds=N_SEEDS, verbose=True, **kw):
    runs = []
    for sigma in sigmas:
        for nh2 in nh2_grid:
            for lam in lambdas:
                batch = [run_noise_config(nh1, nh2, sigma, lam, train, val, test,
                                          seed=s, **kw) for s in range(n_seeds)]
                runs.extend(batch)
                if verbose:
                    v = np.array([b["val_mse"] for b in batch])
                    print(f"  sigma={sigma:<5} {nh1}x{nh2}  lam={lam:<7g} "
                          f"val {v.mean():.3e} +/- {v.std(ddof=1):.1e}")
    return runs


FIELDS = ("train_mse_noisy", "train_mse_clean", "val_mse", "test_mse",
          "w_std", "w_absmean", "w_absmax")


def summarise(runs, fields=FIELDS):
    keys = sorted({(r["sigma"], r["nh2"], r["lam"]) for r in runs})
    rows = []
    for sigma, nh2, lam in keys:
        sel = [r for r in runs if (r["sigma"], r["nh2"], r["lam"]) == (sigma, nh2, lam)]
        row = {"sigma": sigma, "nh1": sel[0]["nh1"], "nh2": nh2, "lam": lam}
        for f in fields:
            vals = np.array([r[f] for r in sel], dtype=float)
            row[f"{f}_mean"] = vals.mean()
            row[f"{f}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def best_lambda_per_config(rows):
    """For each (sigma, nh2), the lambda with the lowest mean validation MSE.

    Reading this down the sigma axis answers the lab's last question: does the
    optimal amount of regularisation grow with the amount of noise?
    """
    out = {}
    for sigma in sorted({r["sigma"] for r in rows}):
        for nh2 in sorted({r["nh2"] for r in rows}):
            sel = [r for r in rows if r["sigma"] == sigma and r["nh2"] == nh2]
            out[(sigma, nh2)] = min(sel, key=lambda r: r["val_mse_mean"])
    return out


def format_table(rows):
    head = (f"{'sigma':>6} {'arch':>6} {'lambda':>8} {'train(noisy)':>13} "
            f"{'train(clean)':>13} {'val MSE':>13} {'test MSE':>13} "
            f"{'std(w)':>8} {'max|w|':>8}")
    lines = [head, "-" * len(head)]
    prev = None
    for r in sorted(rows, key=lambda q: (q["sigma"], q["nh2"], q["lam"])):
        if prev is not None and (r["sigma"], r["nh2"]) != prev:
            lines.append("")
        prev = (r["sigma"], r["nh2"])
        lines.append(
            f"{r['sigma']:>6.2f} {r['nh1']}x{r['nh2']:<4} {r['lam']:>8g} "
            f"{r['train_mse_noisy_mean']:>13.3e} {r['train_mse_clean_mean']:>13.3e} "
            f"{r['val_mse_mean']:>13.3e} {r['test_mse_mean']:>13.3e} "
            f"{r['w_std_mean']:>8.3f} {r['w_absmax_mean']:>8.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    NH1 = int(os.environ.get("NH1", 5))  # grid-search winner
    _, (train, val, test) = load_data()
    print(f"noise sweep, nh1={NH1}, {EPOCHS} epochs, no early stopping\n")
    runs = sweep(NH1, train, val, test)
    rows = summarise(runs)
    print()
    print(format_table(rows))
    print("\nbest lambda per (sigma, nh2), by validation MSE:")
    for (sigma, nh2), r in sorted(best_lambda_per_config(rows).items()):
        print(f"  sigma={sigma:<5} {NH1}x{nh2}  lambda*={r['lam']:<8g} "
              f"val={r['val_mse_mean']:.3e}")

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "results"), exist_ok=True)
    # Plain JSON rather than a pickled .npy: these caches get committed and read
    # back by the notebook, and JSON is inspectable and carries no code-execution
    # risk. The only non-JSON field is the weight vector, so list() it.
    dump = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            for r in runs]
    json.dump(dump, open(os.path.join(here, "results", "noise_runs.json"), "w"))
    json.dump(rows, open(os.path.join(here, "results", "noise_rows.json"), "w"), indent=1)
    print("\nsaved results/noise_runs.json and results/noise_rows.json")
