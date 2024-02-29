import numpy as np


def normalize_input(X):

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X_normalized = (X - mean) / std_dev
    return X_normalized


# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
print("Original X:")
print(X)

X_normalized = normalize_input(X)
print("\nNormalized X:")
print(X_normalized)
