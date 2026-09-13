# Part II — the theory you need to defend at the presentation and the exam

Reference notes for `lab1b/part2_timeseries`. Everything here maps onto a
question the lab (or an exam) actually asks.

---

## 1. Why Mackey-Glass, and what "chaotic" buys the examiner

The system is the delay differential equation

```
dx/dt = β·x(t−τ) / (1 + x(t−τ)ⁿ) − γ·x(t)
```

Three properties matter:

**It is deterministic.** There is no randomness in the generator. So a perfect
predictor exists in principle, and any error your network makes is *your*
error, not irreducible noise. (In practice our networks plateau around
val MSE 4e−4 — the limit is the small architectures the lab permits, not the
data.) It also makes the noise experiments in task 4 a clean controlled
comparison: you know exactly how much of the error you injected.

**It is infinite-dimensional but low-dimensional in practice.** A delay
equation has an infinite-dimensional state space (you need the whole history
over `[t−τ, t]`), but the *attractor* the trajectory settles onto has a low
fractal dimension (≈ 2.1 for τ = 17, rising with τ). That is what makes a
5-dimensional delay vector sufficient.

**It is chaotic for τ > 16.8.** Nearby trajectories separate exponentially
(positive largest Lyapunov exponent). Consequences you should be able to state:

- Long-horizon prediction is impossible in principle; short-horizon (here +5
  steps) is fine. The prediction horizon that stays accurate is set by the
  Lyapunov time, not by network size.
- The series never repeats. A model cannot win by memorising a period — it has
  to learn the local map. This is exactly why it is a benchmark.
- The series is bounded and aperiodic, so train/validation/test blocks taken
  from different stretches are statistically comparable but never identical.

`τ = 25` is chosen well inside the chaotic regime, so the benchmark is hard.

**Euler discretisation.** Step size 1 gives
`x(t+1) = x(t) + 0.2·x(t−25)/(1+x(t−25)¹⁰) − 0.1·x(t)`. This is a first-order
approximation with O(h) local error; the lab accepts it because we only need
*a* chaotic series to learn, not a numerically faithful solution of the ODE.

**Gotcha in the code:** `x[t-25]` for `t < 25` must be forced to 0. In numpy a
negative index wraps to the end of the array and would silently generate a
different series.

---

## 2. Time-delay embedding — why 5 inputs, why those lags

An MLP is a static function `R⁵ → R`. It has no memory. To predict a time
series with it you must put time into the input vector by hand:

```
input(t)  = [x(t−20), x(t−15), x(t−10), x(t−5), x(t)]
target(t) =  x(t+5)
```

**Takens' embedding theorem** is the justification. For a deterministic
dynamical system with attractor of box-counting dimension `d`, a delay vector
of dimension `m > 2d` with almost any delay `Δ` is a *diffeomorphic embedding*
of the attractor. Practically: the delay vector uniquely identifies the system's
state, so

```
x(t+5) = F( x(t−20), …, x(t) )
```

for some deterministic function `F`. Once that holds, the problem is no longer
time-series prediction — it is plain regression, and the MLP's universal
approximation property applies to `F`.

Two parameters to be able to defend:

- **Embedding dimension m = 5.** Must exceed twice the attractor dimension.
  For τ = 25 the attractor dimension is around 3, so m = 5 is at the practical
  edge — adequate, not generous. If asked "would more lags help?", the honest
  answer is: possibly slightly, at the cost of more input weights and more
  variance.
- **Delay Δ = 5.** Too small and consecutive components are nearly identical
  (redundant, badly conditioned inputs); too large and they become effectively
  independent (the geometry of the attractor is lost). Δ = 5 with a window of
  20 samples is comfortably inside one oscillation of the series.

The prediction horizon is also 5, so the task is genuinely extrapolative — the
answer is not in the input window.

---

## 3. Why the split must be consecutive blocks

The lab insists on **consecutive, non-overlapping** blocks, last 200 = test.
The reason is data leakage:

`input(t)` and `input(t+1)` share four of their five components. If you shuffled
randomly, an almost-identical sample would sit in both the training and the
validation set. Your validation error would then measure *interpolation between
near-duplicates*, not generalisation, and it would be optimistically biased —
often by an order of magnitude. Every model-selection decision built on it would
be worthless.

Blocks also match the deployment story: you fit on the past and predict the
future. Random shuffling lets the model "see the future", which is unavailable
at prediction time.

**Our split: 800 train / 200 validation / 200 test.** Only the test size is
fixed by the lab. The justification for 200 validation samples: it is roughly
two full oscillations of the series, enough for the validation MSE to be a
stable ranking signal, while 800 training samples keep the sample-to-parameter
ratio healthy (the biggest grid model, 5×6, has 5·5+5 + 5·6+6 + 6+1 = 73
parameters, so ≈ 11 samples per parameter).

---

## 4. Network configuration, and why every choice is what it is

```
5 inputs → nh1 tanh → nh2 tanh → 1 linear
```

"Three-layer perceptron" counts **weight layers**, so two hidden layers. Watch
this in the exam — the naming convention differs between textbooks.

**Sigmoidal hidden units — we use `tanh`.** The lab asks for "sigmoidal"
transfer functions, which names a *shape* (a smooth monotone S), not one
specific formula. `tanh` qualifies, and it is exactly part I's transfer
function: `φ(x) = 2/(1+e^-x) − 1 = tanh(x/2)`. Being zero-centred it is better
conditioned than the logistic sigmoid, whose output is always positive, which
forces all weights feeding a given unit to receive same-signed gradients.

Each unit is a smooth step. A weighted sum of smooth steps can approximate any
continuous function on a compact set (universal approximation). One hidden
layer already suffices in principle; the second layer buys *composition* — it
combines features the first layer built, usually reaching the same accuracy
with fewer total units.

**Linear output.** The single most important architectural choice here, and a
classic exam question.

- The target `x(t+5)` lives roughly in `[0.2, 1.4]`. A logistic sigmoid is
  bounded in `(0,1)` and the symmetric version `2/(1+e⁻ˣ)−1` in `(−1,1)`.
  A sigmoid output physically *cannot* produce 1.4.
- Worse, `φ'(x) = φ(1−φ)` (or `(1+φ)(1−φ)/2` for the symmetric form) → 0 when
  saturated. A saturated output unit has near-zero gradient, so backprop
  multiplies the error by ≈ 0 and nothing upstream learns. This is the
  vanishing-gradient failure mode in its simplest form.
- A linear output has derivative 1 and unbounded range. For regression with
  squared error it is the default, and it makes `δ⁽ᵒ⁾ = (o − t)` exactly — the
  raw error, undistorted.

(If you *insisted* on a sigmoid output you would have to rescale the targets
into its range first. Mention this if asked — it shows you understand it is a
range problem, not a magic rule.)

**Batch backprop.** `batch_size = len(X_train)`, so one epoch = one gradient
step computed over all 800 patterns. The gradient is the exact gradient of the
epoch loss, so the descent path is smooth and deterministic given the
initialisation. Contrast with sequential/online: noisy steps, can escape
shallow local minima, but never converges exactly.

**No input standardisation.** Measured, not assumed: the inputs already sit in
[0.2, 1.4], a well-conditioned range for tanh, and standardising moved the
validation error by less than the seed-to-seed spread. Leaving the data in its
natural units also keeps σ = 0.05 / 0.15 directly interpretable against the
signal scale.

**Adam at lr = 0.01.** The lab warns that a bad learning rate gives premature
convergence or absurdly long runs. Adam adapts a per-parameter step size from
running estimates of the gradient's first and second moment, which removes most
of that hand-tuning. Say so explicitly if asked why you did not use plain SGD
with momentum — plain full-batch SGD on this problem needs a much larger η and
is far more sensitive to it.

---

## 5. Early stopping

Train, watch validation MSE each epoch, keep the weights from the epoch with
the lowest validation MSE, stop when it has not improved for `patience` epochs.

**Why it regularises.** Weights start small (near-linear regime of the sigmoid,
so the network starts as an almost-linear model) and grow as training proceeds.
Growing weights = growing effective capacity. Stopping early therefore caps the
effective capacity *without* changing the architecture. For a linear model this
is provably close to L2 regularisation with `λ ∝ 1/(η · epochs)` — that is a
neat result worth quoting.

**The premature-stopping trap the lab warns about.** Full-batch descent on this
series has long plateaus where validation error is flat before dropping again.
A small patience (Keras defaults to 0) stops in the first plateau and you report
a badly underfitted model. Our fix: `patience = 300` with
`restore_best_weights=True`, and a 3000-epoch ceiling. Always report the epoch
at which stopping actually fired — if it is consistently the ceiling, early
stopping never triggered and you should say so.

**Do not monitor `loss` when λ > 0.** In Keras the reported `loss` includes the
weight-decay penalty. Monitor a separate pure-MSE metric (`val_mse` in
`model.py`), otherwise the number you stop on and the number you report are
different things, and models with different λ are not comparable.

---

## 6. Weight decay (L2) — the mechanics

Objective becomes

```
E(w) = MSE(w) + λ·Σ wᵢ²
```

so the gradient gains a term `2λwᵢ`, and the plain-gradient update is

```
wᵢ ← wᵢ − η(∂MSE/∂wᵢ) − 2ηλwᵢ  =  (1 − 2ηλ)·wᵢ − η·∂MSE/∂wᵢ
```

The factor `(1 − 2ηλ)` is where the name comes from: every weight shrinks
multiplicatively on every step. A weight survives only if the data gradient
keeps pushing it back. Weights that the data does not need decay to ≈ 0.

**Why that helps generalisation.** Small weights keep the sigmoid units in
their near-linear region, so the function the network implements is *smoother*
— lower curvature, fewer wiggles between training points. Noise is
high-frequency and uncorrelated; a smooth function cannot chase it. You trade
bias (a smoother model cannot fit the true function exactly either) for variance
(the fit depends much less on which noise realisation you got).

**Effect on the weight distribution — expect this in your histograms.** As λ
grows, the histogram of all kernel weights contracts towards zero:
`std(w)` falls, `max|w|` falls, the heavy tails vanish first, and at large λ
the distribution collapses to a narrow spike at 0 (the network is then
effectively a constant plus a tiny linear term — underfitting). Plot
`plot_weight_distribution` for the report; this is the direct answer.

**Biases are not penalised.** Penalising a bias only shifts the function
vertically, it does not make it smoother, and it biases the predictions towards
zero. Keras's `kernel_regularizer` correctly leaves biases alone.

**The λ-vs-noise interaction — the lab's final question.** Bias-variance
decomposition of expected squared error:

```
E[(ŷ − y)²] = bias² + variance + σ²
```

The `σ²` term is irreducible. But the *variance* term scales with σ²: more
observation noise means the fitted function depends more strongly on which
noise realisation you drew. Regularisation reduces variance at the cost of
bias, so the optimum sits where the marginal variance reduction equals the
marginal bias increase. **As σ increases, the variance term grows, so the
optimal λ increases.** Concretely: expect λ* at σ = 0.15 to be roughly an order
of magnitude larger than λ* at σ = 0.05, and λ* ≈ 0 on clean data. Your
`plot_lambda_curves` U-curves should have their minimum move visibly right as σ
grows. That, plus "capacity you cannot afford under noise you can afford when
regularised", is the whole story of task 4.

---

## 7. Model selection discipline

Three sets, three distinct jobs:

| set | used for | touched how often |
|---|---|---|
| training | weight updates | every epoch |
| validation | early stopping, choosing nh1/nh2/λ | every epoch, and once per candidate |
| test | the final generalisation estimate | exactly once, at the very end |

The moment you select a model on test error, the test error stops being an
unbiased estimate of generalisation — you have optimised against it, so it is
just another training signal, and it is optimistically biased by roughly the
spread of the candidates you searched over. Selecting over 9 architectures and
reporting the best *validation* score has the same bias problem in miniature,
which is exactly why you then re-measure on test.

**Why repeats and both moments.** Random initialisation makes every run a draw
from a distribution over solutions. A single run of 4×6 beating a single run of
5×4 tells you nothing. Report `mean ± std` over ≥ 5 seeds. The std is not
bookkeeping — it *is* the lab's "how robust is this architecture to random
weight initialisation?" answer:

- small mean, small std → genuinely good and reliable
- small mean, large std → it *can* be good; you got lucky; needs restarts
- large std generally → the loss surface has many dissimilar local minima that
  the architecture can fall into

Expect the **smallest** architectures (3×2) to show both the worst mean and
often a large std: too few units means the reachable solutions differ a lot
depending on where you start.

---

## 8. What we actually measured

These are our numbers, not predictions. Use them as sanity checks.

### Clean-data grid (10 seeds per architecture, early stopping, 60k ceiling)

| arch | params | val MSE (mean ± std) | test MSE | std/mean | mean best epoch |
|---|---|---|---|---|---|
| **5×6** | 73 | **3.605e−4 ± 3.3e−5** | 4.417e−4 | 9.2 % | 13 590 |
| 3×6 | 49 | 3.737e−4 ± 2.9e−5 | 4.36e−4 | 7.9 % | 20 572 |
| 5×4 | 59 | 3.747e−4 ± 4.7e−5 | 4.20e−4 | 12.6 % | 19 219 |
| 4×6 | 61 | 3.881e−4 ± 3.1e−5 | 4.41e−4 | 7.9 % | 13 986 |
| 4×4 | 49 | 4.015e−4 ± 4.7e−5 | 4.28e−4 | 11.6 % | 19 387 |
| 5×2 | 45 | 4.076e−4 ± 5.2e−5 | 4.46e−4 | 12.7 % | 25 477 |
| 4×2 | 37 | 4.089e−4 ± 5.5e−5 | 4.51e−4 | 13.4 % | 33 848 |
| 3×2 | 29 | 4.302e−4 ± 3.9e−5 | 4.78e−4 | 9.0 % | 41 962 |
| **3×4** | 39 | **4.378e−4 ± 5.3e−5** | 4.573e−4 | 12.2 % | 14 791 |

Reference scale: `var(y_train) ≈ 0.086`, and a least-squares **linear** model on
the same five inputs gets validation MSE **8.5e−3**. So every network is about
20× better than linear, and the whole grid sits in a narrow band.

### The four findings worth defending

**1. Architecture barely matters here — but the extremes are still separable.**
The grid spans only 3.6e−4 to 4.4e−4, a 22 % range, and neighbouring
architectures have overlapping error bars. A Welch t-test over the 10 seeds:

- 5×6 (best) vs 3×4 (worst): t = 3.89, **p = 0.0014** — a real difference.
- 3×4 vs 3×2, the two worst: t = 0.36, **p = 0.72** — *not* significant. The
  "least suitable" label is a coin flip between them, and the report should say
  so rather than pretending the ranking is crisp.
- 3×6 (49 params) vs 5×6 (73): p = 0.36 — not significant. Roughly two thirds
  of the parameters buys nothing measurable.

**2. The convergence trap — the most important methodological result.**
Our first grid used a fixed 10 000-epoch budget and produced a completely
different answer: 3×2 came last at 6.94e−4 ± 2.0e−4. With a 60 000 ceiling and
`patience = 3000` the same architecture reaches **4.30e−4 ± 3.9e−5** — better
*and* five times more reproducible. It had simply not finished converging.

Across the grid, `corr(n_params, mean best epoch) = −0.77`, and convergence
time spans **13 590 epochs (5×6) to 41 962 (3×2), a 3.1× ratio**. Smaller
networks converge far more slowly: fewer parameters means a narrower, more
ill-conditioned loss surface with less gradient redundancy to exploit.

So **a fixed epoch budget ranks convergence speed, not capacity.** This is
exactly the failure the lab hints at in section 4.2 ("inappropriate choice of
the learning rate can lead to premature convergence"), and catching it is worth
more marks than the ranking itself.

**3. Nothing in the clean grid over-fits.** Train ≈ val ≈ test everywhere (all
within about 20 % of each other), because the largest model has 73 parameters
against 800 training samples. Over-fitting only appears well outside the lab's
range — a 64×64 network reaches train 8.1e−6 but validation 5.7e−4, *worse*
than every model in the grid. This is why task 4 has to **add noise** before
regularisation can do anything useful.

**4. Test error is systematically above validation error** (≈ 4.2–4.8e−4 vs
3.6–4.4e−4) for *every* architecture. Since it is uniform, it reflects the test
block being a slightly harder stretch of the series, not over-fitting. Say this
explicitly instead of hiding it.

### Noise experiments (task 4), nh1 = 5, 20 000 epochs, no early stopping

**Optimal λ rises with the noise level — the headline result.**

| σ | λ\* (nh2=3) | λ\* (nh2=6) | λ\* (nh2=9) |
|---|---|---|---|
| 0.00 | 0 | 0 | 0 |
| 0.05 | 1e−5 | 1e−5 | 1e−4 |
| 0.15 | 1e−4 | 1e−4 | 1e−4 |

Exactly what the bias–variance decomposition predicts. On clean data there is no
variance to suppress, so any λ > 0 only adds bias: validation error rises
monotonically, 3.8e−4 (λ=0) → 2.8e−3 (λ=1e−3) → 5.2e−2 (λ=0.1). That clean run
is the control condition that makes the noisy results meaningful.

**Capacity flips from harmless to harmful — and λ undoes it.**
At σ = 0.15 with **λ = 0**, more second-layer units make things strictly worse:

| nh2 | val MSE, λ=0 | val MSE, λ=1e−4 |
|---|---|---|
| 3 | 1.054e−2 | 8.480e−3 |
| 6 | 1.352e−2 | 8.598e−3 |
| 9 | 1.654e−2 | 8.875e−3 |

Unregularised, going 3 → 9 units costs 57 % more error. With λ = 1e−4 the three
architectures are within 5 % of each other. **Regularisation neutralises the
capacity penalty**, which is the whole point of task 4. On clean data the same
sweep is flat or mildly favours capacity — so the liability is created by the
noise, not by the architecture.

**The clean-vs-noisy training columns show over-fitting directly.**
At σ = 0.15, λ = 0, as nh2 goes 3 → 6 → 9:

- training MSE against the *noisy* targets **falls**: 3.53e−2 → 3.32e−2 → 3.05e−2
- training MSE against the *clean* targets **rises**: 8.31e−3 → 9.48e−3 → 1.09e−2
- validation MSE **rises**: 1.05e−2 → 1.35e−2 → 1.65e−2

The extra capacity is being spent fitting the noise realisation, and the price
is paid in recovery of the truth. This pair of columns is the single clearest
piece of evidence in the whole lab — quote it.

**Effect of λ on the weight distribution** (σ = 0.15, nh2 = 6, pooled over seeds):

| λ | std(w) | max\|w\| |
|---|---|---|
| 0 | 2.89 | 22.15 |
| 1e−5 | 1.10 | 3.19 |
| 1e−4 | 0.65 | 2.09 |
| 1e−3 | 0.29 | 1.07 |
| 1e−2 | 0.19 | 0.62 |
| 1e−1 | 0.00 | 0.00 |

Monotone contraction towards zero, tails disappearing first, and total collapse
at λ = 0.1 (the network becomes a constant — maximal bias, zero variance).

**Noise inflates weights when unregularised.** At λ = 0, std(w) for nh2 = 9 goes
1.54 (σ=0) → 1.68 (σ=0.05) → 3.76 (σ=0.15), with max\|w\| reaching 22.5. Fitting
noise requires sharp, high-curvature wiggles, and sharp wiggles require large
weights. That is the mechanism by which penalising large weights suppresses
noise-fitting — not an analogy, the actual causal chain.

### Task 5 (σ = 0.09, non-mandatory)

| model | arch | λ | test MSE |
|---|---|---|---|
| A: selected under noise | 5×6 | 1e−5 | 3.723e−3 ± 4.9e−4 |
| B: task-2 most suitable | 5×6 | 0 | 3.739e−3 ± 5.6e−4 |
| C: task-2 least suitable | 3×4 | 0 | 3.984e−3 ± 5.5e−4 |

Two things to explain, both of which sound like negative results but are not:

1. **A barely beats B, and selected the same architecture.** Because task 5 keeps
   **early stopping on** (the lab says the validation set is used for it there),
   and early stopping is already a regulariser. It caps effective capacity before
   weight decay gets a chance to matter, so the selected λ is tiny (1e−5). This
   is precisely why task 4 *removes* early stopping — otherwise the two
   regularisers are confounded and λ looks useless.
2. **Test MSE (≈ 3.7e−3) is below the noise variance σ² = 8.1e−3.** Not a
   contradiction: the test targets are *clean*, so the network is being scored on
   how well it recovered the true dynamics, and averaging over 800 noisy training
   samples removes much of the noise. Compare with clean data (task 3, ≈ 4.4e−4):
   training on σ = 0.09 data costs roughly 8× in generalisation error.

## 9. Likely questions, short answers

**Why not a recurrent network?** It would be the natural choice, but the lab is
about MLPs. The delay embedding is precisely the trick that lets a static
network do a dynamic task.

**Why is the training error lower than the validation error?** The weights were
fitted to minimise the training error, so it is optimistically biased by
construction. The gap is a rough measure of overfitting.

**Why does early stopping count as regularisation if it changes no objective?**
Because it limits the *effective* capacity: weights grow during training, and
capacity grows with them. Stopping early is a budget on how far into the
high-capacity regime you travel.

**Could you use both early stopping and weight decay?** Yes, in practice. The
lab drops early stopping in task 4 deliberately, so that λ is the only varying
regulariser and its effect is not confounded.

**Why is the output layer regularised too?** It need not be; penalising only
the hidden layers is also defensible. We penalise all kernels for consistency
so the weight-distribution histogram has one uniform interpretation.

**What is the irreducible error here?** On clean data, zero (the generator is
deterministic). On data with additive noise of variance σ², it is σ² — no model
can beat that on noisy targets, which is exactly why we also report the error
against the clean targets.
