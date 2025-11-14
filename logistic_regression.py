import numpy as np


def sigmoid(z):
    """Compute the sigmoid of z."""
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b, lambda_):
    """ 
    Compute the regularized logistic regression cost function.
    Args:
        X (ndarray): Input features, shape (m, n) where m is the number of examples and n is the number of features.
        y (ndarray): True labels, shape (m,).
        w (ndarray): Weights, shape (n,).
        b (float): Bias term.
        lambda_ (float): Regularization parameter.
    Returns:
        total_cost (scalar): cost.
    """

    m, n = X.shape

    total_cost = 0.0
    reg_cost = 0.0

    z = X @ w + b
    f_wb = sigmoid(z)

    epsilon = 1e-15
    f_wb = np.clip(f_wb, epsilon, 1 - epsilon)

    total_cost = 1/m * np.sum((-y * np.log(f_wb)) - (1 - y) * np.log(1 - f_wb))
    reg_cost = (lambda_ / (2*m)) * np.sum(np.square(w))

    total_cost = total_cost + reg_cost

    return total_cost

def compute_gradient(X, y, w, b, lambda_):
    """
    compute the gradient for regularized logistic regression.

    Args:
        X (ndarray): Input features, shape (m, n) where m is the number of examples and n is the number of features.
        y (ndarray): True labels, shape (m,).
        w (ndarray): Weights, shape (n,).
        b (float): Bias term.
        lambda_ (float): Regularization parameter.
    """

    m, n = X.shape

    dj_dw, dj_db = np.zeros((n,)), 0.0

    z = X @ w + b
    f_wb = sigmoid(z)
    error = f_wb - y

    dj_dw = (1/m) * (X.T @ error) + (lambda_/m) * w
    dj_db = (1/m) * np.sum(error) 

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_, cost_function, gradient_function):
    """
    Perform gradient descent to learn w and b.

    Args:
        X (ndarray): Input features, shape (m, n) where m is the number of examples and n is the number of features.
        y (ndarray): True labels, shape (m,).
        w_in (ndarray): Initial weights, shape (n,).
        b_in (float): Initial bias term.
        alpha (float): Learning rate.
        num_iters (int): Number of iterations for gradient descent.
        lambda_ (float): Regularization parameter.
        cost_function (function): Function to compute the cost.
        gradient_function (function): Function to compute the gradient.

    Returns:
        w (ndarray): Updated weights after gradient descent.
        b (float): Updated bias term after gradient descent.
        J_history (list): History of cost function values.
    """

    w = w_in.copy()
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)

        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            cost = cost_function(X, y, w, b, lambda_)
            J_history.append(cost)

    return w, b, J_history