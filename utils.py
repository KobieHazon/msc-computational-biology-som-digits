import math
from typing import Dict, Tuple, Counter

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def matrix_gaussian_weights(center, sigma, x_indices: np.ndarray, y_indices: np.ndarray):
    # TODO: refactor to prevent code similarity to minisom
    """Returns a Gaussian centered in center."""
    d = 2 * sigma * sigma
    return np.exp((-np.power(x_indices - x_indices.T[center], 2) / d) +
                  (-np.power(y_indices - y_indices.T[center], 2) / d)).T


def plot_frequency_matrix(location_to_frequency: Dict[Tuple[int, int], Counter[int]], matrix_shape: Tuple[int, int]):
    neurons_common_digit_string = np.full(matrix_shape, f"None (0%)", dtype=object)
    percentage_matrix = np.zeros(matrix_shape)

    for (row, col), counter in location_to_frequency.items():
        most_common_element, count = counter.most_common(1)[0]
        dominant_digit_prob = count / sum(counter.values())
        neurons_common_digit_string[row, col] = f"{int(most_common_element)}({math.ceil(100 * dominant_digit_prob)}%)"
        percentage_matrix[row, col] = dominant_digit_prob

    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=neurons_common_digit_string, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    cell_colormap = matplotlib.colormaps['Blues']
    for (i, j), cell in table.get_celld().items():
        cell.set_facecolor(cell_colormap(percentage_matrix[i, j]))

    plt.show()
