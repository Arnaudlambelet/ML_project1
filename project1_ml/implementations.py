"""Required ML methods for EPFL CS-433 Project 1 (Table 1 of the project description).

Every public function returns ``(w, loss)`` where ``w`` is the LAST weight vector
and ``loss`` is the loss evaluated at that ``w``.

Conventions (from the project description):
    * all vectors are 1D arrays of shape (D,), never (D, 1)
    * MSE uses the 1/2 factor:  L(w) = 1/(2N) * ||y - Xw||^2
    * logistic methods expect y in {0, 1}
    * regularised methods return the loss WITHOUT the penalty term
    * SGD uses mini-batch size 1
    * only NumPy is used (no np.linalg.lstsq)
"""

import numpy as np

# Losses and gradients


def compute_mse(y, tx, w):
    """MSE loss with the 1/2 factor: 1/(2N) * sum (y - tx @ w)^2."""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE loss: -1/N * tx^T (y - tx w)."""
    e = y - tx @ w
    return -tx.T @ e / y.shape[0]


def sigmoid(t):
    """Numerically stable logistic function 1 / (1 + exp(-t))."""
    # 0.5 * (1 + tanh(t/2)) is mathematically identical and never overflows.
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def compute_logistic_loss(y, tx, w):
    """Mean negative log-likelihood for y in {0, 1}.

    L(w) = 1/N * sum [ log(1 + exp(x_n^T w)) - y_n x_n^T w ]
    np.logaddexp(0, z) = log(1 + exp(z)) computed without overflow.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the mean logistic loss: 1/N * tx^T (sigmoid(tx w) - y)."""
    return tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]


# 1. Linear regression with (full) gradient descent


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: shape (N,) targets
        tx: shape (N, D) features
        initial_w: shape (D,) starting weights
        max_iters: number of GD steps
        gamma: step size
    Returns:
        (w, loss): last weights and their MSE loss
    """
    w = np.asarray(initial_w, dtype=float).copy()
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


# 2. Linear regression with stochastic gradient descent (batch size 1)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD with mini-batch size 1.

    At each step one sample n is drawn uniformly at random and
    w <- w - gamma * grad L_n(w).
    """
    w = np.asarray(initial_w, dtype=float).copy()
    n_samples = y.shape[0]
    for _ in range(max_iters):
        n = np.random.randint(n_samples)
        # keep 2D slice for tx so the gradient helper works unchanged
        w = w - gamma * compute_mse_gradient(y[n : n + 1], tx[n : n + 1], w)
    return w, compute_mse(y, tx, w)


# 3. Least squares (normal equations)


def least_squares(y, tx):
    """Least squares via the normal equations  (X^T X) w = X^T y."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


# 4. Ridge regression (normal equations)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via (X^T X + 2N*lambda I) w = X^T y.

    Objective (lecture convention): 1/(2N)||y - Xw||^2 + lambda ||w||^2.
    The returned loss is the plain MSE (no penalty term).
    """
    n_samples, n_features = tx.shape
    a = tx.T @ tx + 2 * n_samples * lambda_ * np.eye(n_features)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


# 5. Logistic regression with gradient descent


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using full gradient descent (y in {0, 1})."""
    w = np.asarray(initial_w, dtype=float).copy()
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


# 6. Regularised logistic regression with gradient descent


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularised logistic regression using GD (y in {0, 1}).

    Objective: logistic loss + lambda ||w||^2  ->  gradient adds 2*lambda*w.
    The returned loss does NOT include the penalty term.
    """
    w = np.asarray(initial_w, dtype=float).copy()
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
