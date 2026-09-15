import numpy as np
from matplotlib import pyplot as plt

def generate_non_linearly_separable_data(n=100, mA=np.array([1.0, 0.3]), mB=np.array([0.0, -0.1]), sigmaA=0.2, sigmaB=0.3, seed=None):
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
    - classA: Feature matrix of class A.
    - classB: Feature matrix of class B.
    """

    if seed is not None:
        np.random.seed(seed) # For reproducibility
    
    classA = np.random.randn(n, 2) * sigmaA + mA
    classA = classA.T
    
    classB = np.random.randn(n, 2) * sigmaB + mB
    classB = classB.T
    
    X = np.hstack((classA, classB))
    
    # Create target labels
    T = np.hstack((np.ones(n) * (-1), np.ones(n)))
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(X.shape[1])
    X = X[:, shuffle_idx]
    T = T[shuffle_idx]

    return X, T, classA, classB

def subsample_scenario(classA, classB, scenario, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X_train, T_train, X_val, T_val = None, None, None, None
    n_A = classA.shape[1]
    n_B = classB.shape[1]

    if scenario == 1:
        # Scenario 1: Random 25% from each class held out for training (75% validation).
        n_train_A = int(0.25 * n_A)
        n_train_B = int(0.25 * n_B)

        train_idx_A = np.random.choice(n_A, n_train_A, replace=False)
        train_idx_B = np.random.choice(n_B, n_train_B, replace=False)
        val_idx_A = np.setdiff1d(np.arange(n_A), train_idx_A)
        val_idx_B = np.setdiff1d(np.arange(n_B), train_idx_B)

        X_train = np.hstack((classA[:, train_idx_A], classB[:, train_idx_B]))
        T_train = np.hstack((np.ones(n_train_A) * (-1), np.ones(n_train_B)))

        X_val = np.hstack((classA[:, val_idx_A], classB[:, val_idx_B]))
        T_val = np.hstack((np.ones(val_idx_A.size) * (-1), np.ones(n_B - n_train_B)))

    elif scenario == 2:
        # Random 50% from Class A only held out for training
        n_train_A = int(0.5 * n_A)
        train_idx_A = np.random.choice(n_A, n_train_A, replace=False)
        val_idx_A = np.setdiff1d(np.arange(n_A), train_idx_A)

        X_train = np.hstack((classA[:, train_idx_A], classB))
        T_train = np.hstack((np.ones(n_train_A) * (-1), np.ones(n_B)))

        X_val = classA[:, val_idx_A]
        T_val = np.ones(n_A - n_train_A) * (-1)

    elif scenario == 3:
        # Scenario 3: 20% from Class A left (x_1 < 0) and 80% from Class A right (x_1 >= 0) (100% Class B included)
        classA_left_idx = np.where(classA[0, :] < 0)[0]
        classA_right_idx = np.where(classA[0, :] >= 0)[0]
        
        n_train_left = int(0.2 * classA_left_idx.size)
        n_train_right = int(0.8 * classA_right_idx.size)
        
        train_idx_left = np.random.choice(classA_left_idx, n_train_left, replace=False)
        train_idx_right = np.random.choice(classA_right_idx, n_train_right, replace=False)
        
        val_idx_left = np.setdiff1d(classA_left_idx, train_idx_left)
        val_idx_right = np.setdiff1d(classA_right_idx, train_idx_right)
        
        train_idx_A = np.concatenate((train_idx_left, train_idx_right))
        val_idx_A = np.concatenate((val_idx_left, val_idx_right))
        
        X_train = np.hstack((classA[:, train_idx_A], classB))
        T_train = np.hstack((np.ones(train_idx_A.size) * (-1), np.ones(n_B)))
        
        X_val = classA[:, val_idx_A]
        T_val = np.ones(val_idx_A.size) * (-1)

    else:
        raise ValueError("Invalid scenario: {scenario}. Choose 1, 2, or 3.")

    return X_train, T_train, X_val, T_val