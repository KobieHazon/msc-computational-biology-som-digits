import math
from typing import Dict, Tuple, Counter

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def matrix_gaussian_weights(center, sigma, x_indices: np.ndarray, y_indices: np.ndarray):
    # TODO: refactor to prevent code similarity to minisom
    """Returns a Gaussian centered in center."""
    d = 2 * sigma * sigma
    ax = np.exp(-np.power(x_indices - x_indices.T[center], 2) / d)
    ay = np.exp(-np.power(y_indices - y_indices.T[center], 2) / d)
    return (ax * ay).T


def plot_frequency_matrix(location_to_frequency: Dict[Tuple[int, int], Counter[int]], matrix_shape: Tuple[int, int]):
    neurons_common_digit_string = np.full(matrix_shape, f"None (0%)", dtype=object)
    percentage_matrix = np.zeros(matrix_shape)

    for (row, col), counter in location_to_frequency.items():
        most_common_element, count = counter.most_common(1)[0]
        dominant_digit_percentage = math.ceil(100 * count / sum(counter.values()))
        neurons_common_digit_string[row, col] = f"{int(most_common_element)}({dominant_digit_percentage}%)"
        percentage_matrix[row, col] = dominant_digit_percentage

    normalized_percentages = plt.Normalize(percentage_matrix.min(), percentage_matrix.max())
    cell_colormap = matplotlib.colormaps['Blues']

    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=neurons_common_digit_string, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == -1:
            continue  # Skip the table header or index cells
        cell.set_facecolor(cell_colormap(normalized_percentages(percentage_matrix[i - 1, j])))

    plt.show()
