"""
Contains utils to parse and plot different stats on the result of the SOM algorithm on the problem
"""
import math
from typing import Counter as CounterType, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


class _DigitsImagesSOMResult:
    """
    Class returned as a result of the SOM algorithm on the problem
    """

    def __init__(self,
                 neurons: np.ndarray,
                 neuron_frequencies: Dict[Tuple[int, int], CounterType[int]],
                 quantization_errors: List[float],
                 topographical_errors: List[float]):
        """
        :param neurons: neurons at the end of running the SOM algorithm
        :param neuron_frequencies: frequency of the input data mapped to a certain neuron
        :param quantization_errors: quantization errors at the end of each epoch
        :param topographical_errors: topographical errors at the end of each epoch
        """
        self._neurons = neurons
        self._neuron_frequencies = neuron_frequencies
        self._quantization_errors = quantization_errors
        self._topographical_errors = topographical_errors

    @staticmethod
    def plot_matrix(cells_content: np.ndarray,
                    is_image: bool,
                    colormap_name: Optional[str] = None,
                    colormap_values: Optional[np.ndarray] = None):
        """
        Plots a grid for different visualizations on the neuron
        :param cells_content: textual content of the matrix cells
        :param is_image: True for showing images in the matrix cells
        :param colormap_name: name of the colormap, for coloring the cell
        :param colormap_values: values to give the background of the colormap
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.axis('off')

        hex_radius = 1
        hex_height = np.sqrt(3) * hex_radius
        hex_width = 2 * hex_radius
        x_offset = hex_width * 0.75
        y_offset = hex_height * 0.5

        rows, cols = cells_content.shape[:2]
        for i in range(rows):
            for j in range(cols):
                x = j * x_offset
                y = i * hex_height + (j % 2) * y_offset  # Offset every other column
                hexagon = patches.RegularPolygon((x, y), numVertices=6, radius=hex_radius,
                                                 orientation=np.radians(30), edgecolor='black', facecolor='none')
                ax.add_patch(hexagon)

                if not is_image:
                    ax.text(x, y, str(cells_content[i, j]), ha='center', va='center', fontsize=6)
                    if colormap_name is not None:
                        cell_colormap = matplotlib.colormaps[colormap_name]
                        hexagon.set_facecolor(cell_colormap(colormap_values[i, j]))
                else:
                    extent = [x - hex_radius, x + hex_radius, y - hex_height / 2, y + hex_height / 2]
                    ax.imshow(
                        cells_content[i, j], cmap='gray', extent=extent, clip_path=hexagon, clip_on=True, zorder=-1
                    )

        ax.set_xlim(-hex_width, cols * x_offset + hex_width)
        ax.set_ylim(-hex_height, rows * hex_height + hex_height)
        plt.show()

    def show_digit_confidence_matrix(self):
        """
        graphically shows the confidence of the dominant digit in each neuron
        """
        matrix_shape = self._neurons.shape[:2]
        neurons_common_digit_string = np.full(matrix_shape, f"None (0%)", dtype=object)
        percentage_matrix = np.zeros(matrix_shape)

        for (row, col), counter in self._neuron_frequencies.items():
            most_common_element, count = counter.most_common(1)[0]
            dominant_digit_prob = count / sum(counter.values())
            neurons_common_digit_string[
                row, col] = f"{int(most_common_element)} ({math.ceil(100 * dominant_digit_prob)}%)"
            percentage_matrix[row, col] = dominant_digit_prob

        self.plot_matrix(neurons_common_digit_string, False, 'Blues', percentage_matrix)

    def show_dominant_digit_matrix(self):
        """
        graphically shows the dominant digit in each neuron
        """
        neuron_digits = np.zeros(self._neurons.shape[:2], dtype=np.uint8)
        for (row, col), counter in self._neuron_frequencies.items():
            most_common_element, _ = counter.most_common(1)[0]
            neuron_digits[row, col] = most_common_element

        self.plot_matrix(neuron_digits, False, colormap_name='viridis', colormap_values=neuron_digits / 10.)

    def show_neurons_matrix_graphically(self):
        """
        shows a graphical representation of the neurons, as grayscale images
        """

        def reshape_to_image(cell):
            return cell[:28 * 28].reshape(28, 28) * 255.0

        neurons_images = np.apply_along_axis(reshape_to_image, axis=2, arr=self._neurons)
        self.plot_matrix(neurons_images, True)

    def plot_quantization_errors(self):
        """
        Plots the quantization error value by epoch graph
        """
        plt.plot(self._quantization_errors)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Quantization Errors by Iteration')
        plt.show()

    def plot_topographical_errors(self):
        """
        Plots the topographical error value by epoch graph
        """
        plt.plot(self._topographical_errors)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title('Topographical Errors by Iteration')
        plt.show()
