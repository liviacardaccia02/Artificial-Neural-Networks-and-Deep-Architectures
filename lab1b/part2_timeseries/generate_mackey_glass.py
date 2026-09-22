import numpy as np
import matplotlib.pyplot as plt

BETA = 0.2
GAMMA = 0.1
N_EXP = 10
TAU = 25

def mackey_glass(n_points): 
  X = np.array([0.0] * n_points)
  X[0] = 1.5 # Initial condition

  for t in range(n_points - 2):
    delay_index = t - TAU
    if delay_index < 0:
      x_delayed = 0.0
    else:
      x_delayed = X[delay_index]

    # Calculate the next step using the Euler formula
    X[t + 1] = X[t] + (BETA * x_delayed) / (1.0 + x_delayed ** N_EXP) - (GAMMA * X[t])

  return X

def plot_mackey_glass(n_points):
  X = mackey_glass(n_points)
  plt.figure(figsize=(10, 5))
  plt.plot(X, label='Mackey-Glass Time Series')
  plt.title('Mackey-Glass Chaotic Time Series')
  plt.xlabel('Time Steps')
  plt.ylabel('Value')
  plt.legend()
  plt.grid()
  plt.show()


plot_mackey_glass(1600)