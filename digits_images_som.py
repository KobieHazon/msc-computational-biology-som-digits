import functools
import math
from collections import Counter
from typing import Counter as CounterType, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from src.digits_images_reader import DigitsImagesReader
from src.utils import matrix_gaussian_weights


class _DigitsImagesSOMResult:
    def __init__(self,
                 neurons: np.ndarray,
                 som_instance: 'DigitsImagesSOM',
                 digits_images_reader: DigitsImagesReader):
        self._neurons = neurons
        self._som_instance = som_instance
        self._digits_images_reader = digits_images_reader

    @functools.lru_cache
    def _get_neuron_frequencies(self) -> Dict[Tuple[int, int], CounterType[int]]:
        neuron_to_digits: Dict[Tuple[int, int], Counter[int]] = {}
        for digit_image_key, digit_image in self._digits_images_reader:
            closest_neuron = self._som_instance.get_closest_neuron(digit_image)
            if closest_neuron not in neuron_to_digits:
                neuron_to_digits[closest_neuron] = Counter()
            neuron_to_digits[closest_neuron][digit_image_key] += 1

        return neuron_to_digits

    @staticmethod
    def plot_matrix(cells_content: np.ndarray,
                    is_image: bool,
                    colormap_name: Optional[str] = None,
                    colormap_values: Optional[np.ndarray] = None):  # TODO: migrate to hexagonal instead of square
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.axis('off')

        hex_radius = 5  # Set the size of the hexagons
        hex_height = np.sqrt(3) * hex_radius
        hex_width = 2 * hex_radius
        x_offset = hex_width * 0.75  # Horizontal distance between the centers of adjacent hexagons
        y_offset = hex_height * 0.5  # Vertical distance between the centers of adjacent hexagons

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
        neuron_to_digits = self._get_neuron_frequencies()

        matrix_shape = self._neurons.shape[:2]
        neurons_common_digit_string = np.full(matrix_shape, f"None (0%)", dtype=object)
        percentage_matrix = np.zeros(matrix_shape)

        for (row, col), counter in neuron_to_digits.items():
            most_common_element, count = counter.most_common(1)[0]
            dominant_digit_prob = count / sum(counter.values())
            neurons_common_digit_string[
                row, col] = f"{int(most_common_element)} ({math.ceil(100 * dominant_digit_prob)}%)"
            percentage_matrix[row, col] = dominant_digit_prob

        self.plot_matrix(neurons_common_digit_string, False, 'Blues', percentage_matrix)

    def show_dominant_digit_matrix(self):
        neuron_to_digits = self._get_neuron_frequencies()
        neuron_digits = np.zeros(self._neurons.shape[:2], dtype=np.uint8)

        for (row, col), counter in neuron_to_digits.items():
            most_common_element, _ = counter.most_common(1)[0]
            neuron_digits[row, col] = most_common_element

        self.plot_matrix(neuron_digits, False, colormap_name='viridis', colormap_values=neuron_digits / 10.)

    def show_neurons_matrix_graphically(self):
        def reshape_to_image(cell):
            return cell[:28 * 28].reshape(28, 28) * 255.0

        neurons_images = np.apply_along_axis(reshape_to_image, axis=2, arr=self._neurons)
        self.plot_matrix(neurons_images, True)


class DigitsImagesSOM:
    # TODO: optimize parameters
    NEURON_MESH_WIDTH: int = 10
    NEURON_MESH_HEIGHT: int = 10
    NEIGHBORHOOD_SIZE: float = 5  # TODO: parameter for report

    # TODO: find out if our implementation is suitable for epochs or not
    TRAIN_ITERATIONS: int = 20  # TODO: parameter for report
    LEARNING_RATE: float = 0.5  # TODO: parameter for report

    def __init__(self, digits_images_reader: DigitsImagesReader):
        self._digits_images_reader = digits_images_reader

        self._neurons: Optional[np.ndarray] = None
        self._neuron_x_indices, self._neuron_y_indices = np.meshgrid(
            np.arange(self.NEURON_MESH_WIDTH), np.arange(self.NEURON_MESH_HEIGHT)
        )
        self._neuron_x_indices, self._neuron_y_indices = (
            self._neuron_x_indices.astype(float), self._neuron_y_indices.astype(float)
        )
        self._neuron_x_indices[::-2] -= 0.5
        self._neuron_y_indices *= (3.0 / 2.0) / np.sqrt(3)

        self._quantization_error_history: List[float] = []
        self._topological_error_history: List[float] = []

    def _init_clustering(self):
        sample_size = self._digits_images_reader.get_flattened_image_size()
        self._neurons = np.random.rand(self.NEURON_MESH_WIDTH, self.NEURON_MESH_HEIGHT, sample_size)
        # TODO: should we normalize the weights?

    def get_closest_neuron(self, to_image: np.array) -> Tuple[int, int]:
        neurons_distance = self._neurons - to_image
        squared_distances = np.einsum('ijk,ijk->ij', neurons_distance, neurons_distance)
        closest_neuron_flattened_index = np.argmin(squared_distances)

        return divmod(closest_neuron_flattened_index, self.NEURON_MESH_HEIGHT)

    def _clustering_step(self, digit_image: np.array, iteration_decay: float, neighborhood_decay: float):
        closest_neuron = self.get_closest_neuron(digit_image)
        neighborhood_weights = iteration_decay * matrix_gaussian_weights(closest_neuron,
                                                                         neighborhood_decay,
                                                                         self._neuron_x_indices,
                                                                         self._neuron_y_indices)  # TODO: parameter for report(gaussian)

        self._neurons += neighborhood_weights[:, :, np.newaxis] * (
                digit_image - self._neurons)  # TODO: parameter for report

    def run_clustering(self) -> _DigitsImagesSOMResult:
        self._init_clustering()

        for iteration in range(self.TRAIN_ITERATIONS):
            print(f"starting train iteration {iteration}")
            iteration_decay = self.LEARNING_RATE / (
                    1 + iteration * (100 / self.TRAIN_ITERATIONS))  # TODO: parameter for report
            neighborhood_decay = (self.NEIGHBORHOOD_SIZE /
                                  (1 + (iteration * (
                                          self.NEIGHBORHOOD_SIZE - 1) / self.TRAIN_ITERATIONS)))  # TODO: parameter for report

            for _, digit_image in self._digits_images_reader:
                self._clustering_step(digit_image, iteration_decay, neighborhood_decay)

        return _DigitsImagesSOMResult(self._neurons, self, self._digits_images_reader)

    def _quantization_error(self) -> float:
        winners_coords = argmin(self._distance_from_weights(data), axis=1)
        quantization = self._weights[unravel_index(winners_coords,
                                           self._weights.shape[:2])]
        return np.norm(data - quantization, axis=1).mean()
    def report_error(self) -> float:  # TODO: implement with both quantization and topological errors
        pass
