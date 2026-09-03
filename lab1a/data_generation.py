import numpy as np
import matplotlib.pyplot as plt


def generate_linearly_separable_data(n=100, mA=np.array([1.0, 0.5]), mB=np.array([-1.0, 0.0]), sigmaA=0.3, sigmaB=0.3, has_bias=True):
    """
    Generates a linearly separable dataset with two classes.

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

def generate_non_linearly_separable_data(has_bias=True):
    X,T = generate_linearly_separable_data(has_bias=has_bias)
    
    noise_indices = np.random.choice(X.shape[1], size=int(0.1 *X.shape[1]), replace=False)
    T[noise_indices] = 1 - T[noise_indices]
    return X, T

def generate_non_linearly_separable_data_second_version(n=100, mA=np.array([1.0, 0.3]), mB=np.array([0.0, -0.1]), sigmaA=0.2, sigmaB=0.3, has_bias=True):
    mA = np.array([1.0, 0.3])
    sigmaA = 0.2

    mB = np.array([0.0, -0.1])
    sigmaB = 0.3

    classA = np.empty((2, n))
    classB = np.empty((2, n))

    classA[0, :] = np.concatenate([
        np.random.randn(round(0.5 * n)) * sigmaA - mA[0],
        np.random.randn(round(0.5 * n)) * sigmaA + mA[0]
    ])

    classA[1, :] = np.random.randn(n) * sigmaA + mA[1]

    classB[0, :] = np.random.randn(n) * sigmaB + mB[0]

    classB[1, :] = np.random.randn(n) * sigmaB + mB[1]
    X = np.hstack((classA, classB))
    if has_bias:
        X = np.vstack((X, np.ones((1, 2*n))))
    
    T = np.hstack((np.ones(n), np.zeros(n)))
    
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
            plt.title("Current Data")
    plt.legend()
    plt.grid()
    plt.show()