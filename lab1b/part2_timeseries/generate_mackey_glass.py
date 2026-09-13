"""Mackey-Glass chaotic time series generation (lab section 4.1).

The continuous system is the delay differential equation

    dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n) - gamma * x(t)

With beta=0.2, gamma=0.1, n=10 and tau=25 the system is chaotic: bounded,
aperiodic, and exponentially sensitive to the initial condition. That is what
makes it a benchmark -- a model cannot succeed by memorising a period, it has
to learn the local dynamics.

Euler discretisation with step 1 gives the recurrence used in the lab:

    x(t+1) = x(t) + 0.2 * x(t-25) / (1 + x(t-25)^10) - 0.1 * x(t)
"""

import numpy as np

BETA = 0.2
GAMMA = 0.1
N_EXP = 10
TAU = 25
X0 = 1.5


def mackey_glass(n_points=1600, beta=BETA, gamma=GAMMA, n=N_EXP, tau=TAU, x0=X0):
    """Return the series as an array x where x[t] is the value at time t.

    x(0) = x0 and x(t) = 0 for t < 0, exactly as the lab specifies. The guard
    on negative indices matters: numpy would silently wrap x[-1] round to the
    end of the array and produce a different (wrong) series.
    """
    x = np.zeros(n_points, dtype=np.float64)
    x[0] = x0
    for t in range(n_points - 1):
        x_delayed = x[t - tau] if t - tau >= 0 else 0.0
        x[t + 1] = x[t] + beta * x_delayed / (1.0 + x_delayed ** n) - gamma * x[t]
    return x


if __name__ == "__main__":
    x = mackey_glass()
    print(f"length      : {len(x)}")
    print(f"min / max   : {x.min():.4f} / {x.max():.4f}")
    print(f"mean / std  : {x.mean():.4f} / {x.std():.4f}")
    print(f"x[300:305]  : {np.round(x[300:305], 4)}")
