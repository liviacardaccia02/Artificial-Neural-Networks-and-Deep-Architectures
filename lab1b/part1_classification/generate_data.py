import numpy as np
from matplotlib import pyplot as plt

def generate_non_linearly_separable_data(n=100, mA=np.array([1.0, 0.3]), mB=np.array([0.0, -0.1]), sigmaA=0.2, sigmaB=0.3, has_bias=True, seed=None):
    """
    Generates a non-linearly separable dataset with two classes.

    Parameters:
    - n: Number of samples per class.
    - mA: Mean of class A.
    - mB: Mean of class B.
    - sigmaA: Standard deviation of class A.
    - sigmaB: Standard deviation of class B.

    Returns:
    - X: Feature matrix of shape (2, 2*n).
    - T: Target labels of shape (2*n,).
    """

    if seed is not None:
        np.random.seed(seed) # For reproducibility
    
    classA = np.random.randn(n, 2) * sigmaA + mA
    classA = classA.T
    
    classB = np.random.randn(n, 2) * sigmaB + mB
    classB = classB.T
    
    X = np.hstack((classA, classB))
    if has_bias:
        X = np.vstack((X, np.ones((1, 2*n))))
    
    # Create target labels
    T = np.hstack((np.ones(n), np.zeros(n)))
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(X.shape[1])
    X = X[:, shuffle_idx]
    T = T[shuffle_idx]

    return X, T

def plot_data(X, T, title=None):
    """
    Plots the generated dataset.

    Parameters:
    - X: Feature matrix of shape (2, 2*n).
    - T: Target labels of shape (2*n,).
    """
    
    plt.scatter(X[0, T == 1], X[1, T == 1], label="Class A", color='blue')
    plt.scatter(X[0, T != 1], X[1, T != 1], label="Class B", color='red')
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    if title:
        plt.title(title)
    else:
        plt.title("Current Dataset")
    plt.legend()
    plt.grid()
    plt.show()