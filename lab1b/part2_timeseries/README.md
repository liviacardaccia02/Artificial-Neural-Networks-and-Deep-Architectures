# Part II — Mackey-Glass chaotic time-series prediction (lab section 4)

Owner: Tytus. Independent of Part I — this part uses Keras 3 (PyTorch backend),
not the hand-written backprop in `lab1b/shared/`.

## Files

| file | what it does | lab section |
|---|---|---|
| `generate_mackey_glass.py` | Euler-discretised Mackey-Glass series | 4.1 |
| `build_dataset.py` | time-delay embedding, consecutive train/val/test split, noise | 4.1 |
| `model.py` | 5→nh1→nh2→1 MLP, training loop, MSE and weight helpers | 4.2 |
| `grid_search.py` | nh1×nh2 grid search, robustness to init, test evaluation | 4.3 tasks 1–3 |
| `noise_experiments.py` | σ ∈ {0.05, 0.15}, nh2 ∈ {3,6,9}, λ sweep, no early stopping | 4.3 task 4 |
| `model_selection_noise.py` | σ = 0.09 model selection and three-way comparison | 4.3 task 5 (non-mandatory) |
| `compare_backends.py` | checks the PyTorch loop and the Keras 3 model agree | — |
| `plots.py` | every figure used in the report | — |
| `part2_timeseries.ipynb` | runs everything end to end, produces the tables and figures | — |

## Training engine: PyTorch, with Keras as cross-check

The sweeps run on a plain PyTorch loop (`model.train_torch`). With a full batch
one epoch is a single gradient step, and Keras's per-epoch bookkeeping dwarfs
the step itself: `compare_backends.py` measures **20.3 ms/epoch for Keras vs
0.58 ms for raw torch — 35×** (nearer 70× on an idle machine). Across the ~1500
training runs in this part that is a one-hour sweep versus a multi-day one.

The Keras 3 path (`model.build_keras` / `model.train_keras`) is kept because the
lab asks us to identify suitable library functions for regularisation and early
stopping: `keras.regularizers.l2` and `keras.callbacks.EarlyStopping`.
`compare_backends.py` trains both and checks they land within the seed-to-seed
spread. Measured verdict: validation means differ by **0.55 σ** — the two
implementations agree, so the fast path is justified rather than asserted.

## Environment

```bash
conda env create -f environment.yml
conda activate annda
```

Keras 3 runs on the PyTorch backend (`KERAS_BACKEND=torch`, set inside
`model.py`). Do **not** mix conda-forge `numpy` with pip `torch` in the same
env — both ship `libomp` and the process aborts with `OMP: Error #15`. The
`environment.yml` installs everything through pip for that reason.

## Team decisions (things the lab leaves open)

| decision | value | why |
|---|---|---|
| train / val / test split of the 1200 samples | 800 / 200 / 200 | test size is fixed by the lab; 200 validation samples ≈ two oscillations, enough for a stable selection signal, while 800 training samples stay large relative to the ≤ 100 weights of the biggest grid model |
| seeds per configuration (grid search) | 10 | mean and std over seeds; the std *is* the robustness-to-initialisation answer |
| seeds per configuration (noise sweep) | 5 | 180 configurations × 5 keeps runtime sane |
| λ grid | {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} | weight decay acts on the *scale* of λ, so a log grid is the right sweep |
| optimiser / learning rate | Adam, 1e-2, full batch | full batch = batch backprop as required; Adam's per-parameter step sizes remove the hand-tuning the lab warns about |
| epoch budget | 60 000 ceiling with early stopping (patience 3000); 20 000 fixed when early stopping is off | **measured**: at a 10 000 budget the grid ranked convergence *speed*, not capacity — 3×2 kept improving to ~42 000 epochs. A generous ceiling lets every architecture reach its own plateau before it is judged. See NOTES.md §8 |
| noise applied to | training block only, inputs and targets | literal reading of "add noise to training data"; a noisy validation set has an error floor of σ², which would swamp the model differences being measured |

## Resolved ambiguity

The PDF prints `nh2 = 3, 6, 9 6` in section 4.3 task 4. The trailing `6` is a
**footnote marker**, not part of the list. The grid is `3, 6, 9`, which is also
what section 4.2's "max 9 nodes per hidden layer" implies. No TA query needed.

## Run

```bash
conda activate annda
cd lab1b/part2_timeseries
python generate_mackey_glass.py     # sanity check on the series
python build_dataset.py             # sanity check on the shapes
python grid_search.py               # tasks 1-3   (~27 min)
NH1=5 python noise_experiments.py   # task 4, NH1 = winner from the grid search
BEST_ARCH=5,6 WORST_ARCH=3,4 python model_selection_noise.py   # task 5
python compare_backends.py          # PyTorch vs Keras equivalence check
```

Or just open `part2_timeseries.ipynb`, which chains all of it and writes the
figures to `results/`.
