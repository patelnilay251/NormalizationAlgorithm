import numpy as np


def mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=32, epochs=100):
    # Add bias term to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Random initialization of theta
    theta = np.random.randn(X_b.shape[1], 1)

    m = len(y)  # Number of training examples

    for epoch in range(epochs):
        # Shuffle the data
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        # Mini-batch gradient descent
        for i in range(0, m, batch_size):
            xi = X_shuffled[i : i + batch_size]
            yi = y_shuffled[i : i + batch_size]
            gradients = 2 / batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradients

    return theta


# Define your dataset
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([[3], [7], [11], [15]])

# Set hyperparameters
learning_rate = 0.01
batch_size = 2
epochs = 100

# Call the mini_batch_gradient_descent function
optimized_parameters = mini_batch_gradient_descent(
    X, y, learning_rate, batch_size, epochs
)

# Print the optimized parameters
print("Optimized Parameters:")
print(optimized_parameters)
