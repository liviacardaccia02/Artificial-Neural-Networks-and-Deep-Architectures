"""Figures for part II. Every function returns the matplotlib Figure."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def save(fig, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_series(x, t_start=301, t_end=1500, n_train=800, n_val=200):
    """The raw Mackey-Glass series with the three data blocks marked."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(np.arange(len(x)), x, lw=0.8, color="#333")
    b1, b2 = t_start + n_train, t_start + n_train + n_val
    for lo, hi, c, label in [
        (t_start, b1, "#4c9f70", "train"),
        (b1, b2, "#e0a458", "validation"),
        (b2, t_end + 1, "#c05746", "test"),
    ]:
        ax.axvspan(lo, hi, alpha=0.22, color=c, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Mackey-Glass series (beta=0.2, gamma=0.1, n=10, tau=25)")
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    return fig


def plot_learning_curves(history, title="learning curves", best_epoch=None):
    """Training vs validation MSE per epoch, log scale.

    `history` is the dict returned by model.train_torch(record_history=True).

    Log scale is not cosmetic: the error spans several orders of magnitude, and
    on a linear axis everything after the first ~100 epochs collapses onto a
    flat line at zero, hiding exactly the part where over-fitting would show.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ep = history["epoch"]
    ax.semilogy(ep, history["train_mse"], label="train")
    ax.semilogy(ep, history["val_mse"], label="validation")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="k", ls="--", lw=0.9,
                   label=f"best epoch ({best_epoch})")
    ax.set_xlabel("epoch (= one full-batch weight update)")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return fig


def plot_grid_heatmap(rows, field="val_mse_mean", title=None):
    """Mean error over the nh1 x nh2 grid."""
    nh1s = sorted({r["nh1"] for r in rows})
    nh2s = sorted({r["nh2"] for r in rows})
    M = np.full((len(nh1s), len(nh2s)), np.nan)
    for r in rows:
        M[nh1s.index(r["nh1"]), nh2s.index(r["nh2"])] = r[field]

    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(nh2s)), nh2s)
    ax.set_yticks(range(len(nh1s)), nh1s)
    ax.set_xlabel("nh2 (second hidden layer)")
    ax.set_ylabel("nh1 (first hidden layer)")
    ax.set_title(title or field)
    for i in range(len(nh1s)):
        for j in range(len(nh2s)):
            ax.text(j, i, f"{M[i, j]:.1e}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="MSE")
    return fig


def plot_arch_spread(rows):
    """Mean +/- std of validation error per architecture: the robustness figure."""
    rows = sorted(rows, key=lambda r: r["val_mse_mean"])
    labels = [f"{r['nh1']}x{r['nh2']}" for r in rows]
    mean = np.array([r["val_mse_mean"] for r in rows])
    std = np.array([r["val_mse_std"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(len(rows)), mean, yerr=std, fmt="o", capsize=4, color="#2b5d8a")
    ax.set_yscale("log")
    ax.set_xticks(range(len(rows)), labels, rotation=0)
    ax.set_xlabel("architecture (nh1 x nh2), sorted by mean validation MSE")
    ax.set_ylabel("validation MSE")
    ax.set_title("architecture ranking and sensitivity to weight initialisation")
    ax.grid(alpha=0.3, axis="y")
    return fig


def plot_test_predictions(nets, Xte, yte, t_test, labels):
    """Test predictions against the known targets, plus the residuals."""
    from model import predict

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(t_test, yte, color="k", lw=1.6, label="target x(t+5)")
    colors = ["#2b8a5d", "#c05746", "#8a2be2"]
    for (m, lab, c) in zip(nets, labels, colors):
        pred = predict(m, Xte)
        axes[0].plot(t_test, pred, lw=1.1, alpha=0.9, color=c, label=lab)
        axes[1].plot(t_test, pred - yte, lw=0.9, color=c, label=lab)
    axes[0].set_ylabel("x(t+5)")
    axes[0].legend(fontsize=9)
    axes[0].set_title("test-set predictions vs targets")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("t")
    axes[1].grid(alpha=0.3)
    return fig


def plot_lambda_curves(rows, sigma, field="val_mse_mean", std_field="val_mse_std"):
    """Error vs lambda, one curve per nh2, at a given noise level.

    The expected shape is a U: too little regularisation and the network fits
    noise, too much and it cannot fit the signal. The bottom of the U is the
    bias-variance trade-off point, and it should move RIGHT as sigma grows.
    """
    sel = [r for r in rows if r["sigma"] == sigma]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for nh2 in sorted({r["nh2"] for r in sel}):
        s = sorted([r for r in sel if r["nh2"] == nh2], key=lambda r: r["lam"])
        lam = np.array([max(r["lam"], 1e-6) for r in s])  # 0 plotted at the left edge
        m = np.array([r[field] for r in s])
        e = np.array([r[std_field] for r in s])
        ax.errorbar(lam, m, yerr=e, marker="o", capsize=3, label=f"nh2={nh2}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lambda  (leftmost point = 0, plotted at 1e-6)")
    ax.set_ylabel("validation MSE")
    ax.set_title(f"effect of weight decay, sigma = {sigma}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return fig


def plot_weight_distribution(runs, sigma, nh2, lambdas):
    """Histogram of all kernel weights, one panel per lambda.

    This is the direct visual answer to "what is the effect of regularisation
    on the distribution of weights": the histogram should contract towards zero
    as lambda grows, and the tails should disappear first.
    """
    fig, axes = plt.subplots(1, len(lambdas), figsize=(3.0 * len(lambdas), 3.0),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lam in zip(axes, lambdas):
        w = np.concatenate([r["weights"] for r in runs
                            if r["sigma"] == sigma and r["nh2"] == nh2 and r["lam"] == lam])
        ax.hist(w, bins=40, color="#2b5d8a", alpha=0.85)
        ax.set_title(f"lambda={lam:g}\nstd={w.std():.2f}  max|w|={np.abs(w).max():.2f}",
                     fontsize=9)
        ax.set_xlabel("weight value")
        ax.axvline(0, color="k", lw=0.7)
    axes[0].set_ylabel("count")
    fig.suptitle(f"weight distribution, sigma={sigma}, nh2={nh2}"
                 "   (NOTE: x-scale differs per panel -- read std and max|w|)",
                 y=1.08)
    fig.tight_layout()
    return fig
