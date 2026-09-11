import numpy as np

def phi(x):
    return (2 / 1 + np.exp(-x)) - 1

def phi_prime(x):
    return (1 + phi(x)) * (1 - phi(x)) / 2