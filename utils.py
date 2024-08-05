import numpy as np


def matrix_gaussian_weights(center, sigma, x_indices: np.ndarray, y_indices: np.ndarray):
    # TODO: refactor to prevent code similarity to minisom
    """Returns a Gaussian centered in center."""
    d = 2 * sigma * sigma
    ax = np.exp(-np.power(x_indices - x_indices.T[center], 2) / d)
    ay = np.exp(-np.power(y_indices - y_indices.T[center], 2) / d)
    return (ax * ay).T
